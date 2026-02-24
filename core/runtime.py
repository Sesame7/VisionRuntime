"""Core runtime: BaseWorker, async loop helpers, and SystemRuntime."""

from __future__ import annotations

import asyncio
import contextlib
import logging
import os
import queue
import threading
import time
from concurrent.futures import TimeoutError
from dataclasses import dataclass
from datetime import datetime, timezone
from typing import (
    Any,
    Callable,
    Coroutine,
    Optional,
    Protocol,
    Tuple,
    TypeVar,
    cast,
    TYPE_CHECKING,
)

from core.contracts import OutputRecord
from trigger import TriggerConfig, TriggerGateway

if TYPE_CHECKING:  # pragma: no cover
    from .worker import CameraWorker, DetectQueueManager, DetectWorker
    from output.manager import OutputManager, ResultStore

L = logging.getLogger("vision_runtime.runtime")


class BaseWorker:
    def __init__(self, name: str = ""):
        self.name = name or self.__class__.__name__
        self._stop_evt = threading.Event()
        self._thread: threading.Thread | None = None

    def start(self):
        if self._thread:
            return
        self._thread = threading.Thread(target=self._run, daemon=True)
        self._thread.start()

    def stop(self, timeout: float = 2.0):
        self._stop_evt.set()
        if self._thread:
            self._thread.join(timeout=timeout)
            if self._thread.is_alive():
                L.warning("%s worker thread did not exit cleanly", self.name)

    def _run(self):
        try:
            self.run()
        except Exception:
            L.exception("%s worker error", self.name)

    @property
    def is_alive(self) -> bool:
        return bool(self._thread and self._thread.is_alive())

    def run(self):
        """Override in subclasses with the worker loop."""
        raise NotImplementedError


_loop: Optional[asyncio.AbstractEventLoop] = None
_thread: Optional[threading.Thread] = None
_stopped = False
_lock = threading.Lock()
_loop_ready: Optional[threading.Event] = None
_loop_thread_ident: Optional[int] = None


def _ensure_loop():
    global _loop, _thread, _stopped, _loop_ready, _loop_thread_ident
    with _lock:
        if _stopped:
            raise RuntimeError("Async loop already stopped")
        if _loop and _thread and _thread.is_alive():
            return _loop
        _loop = asyncio.new_event_loop()
        _loop_ready = threading.Event()
        _loop_thread_ident = None

        def _runner():
            global _loop_thread_ident
            loop = _loop
            if loop is None:
                return
            asyncio.set_event_loop(loop)
            _loop_thread_ident = threading.get_ident()
            if _loop_ready:
                _loop_ready.set()
            loop.run_forever()

        _thread = threading.Thread(target=_runner, daemon=True)
        _thread.start()
        if _loop_ready:
            _loop_ready.wait(timeout=0.5)
        return _loop


T = TypeVar("T")


def run_async(coro: Coroutine[Any, Any, T], timeout: float | None = 0.5) -> T:
    """Submit coroutine to shared loop from a non-loop thread and wait for result."""
    loop = _ensure_loop()
    if _loop_thread_ident is not None and threading.get_ident() == _loop_thread_ident:
        raise RuntimeError(
            "run_async must not be called from the loop thread; await directly"
        )
    fut = asyncio.run_coroutine_threadsafe(coro, loop)
    try:
        return fut.result(timeout=timeout)
    except TimeoutError:
        fut.cancel()
        L.warning("run_async timeout after %.2fs", timeout or 0)
        raise


def spawn_background_task(coro: Coroutine[Any, Any, Any]):
    """Fire-and-forget task on shared loop; returns the Task handle."""
    loop = _ensure_loop()
    if _loop_thread_ident is not None and threading.get_ident() == _loop_thread_ident:
        return loop.create_task(coro)
    return asyncio.run_coroutine_threadsafe(coro, loop)


def shutdown_loop(timeout: float = 1.0):
    """Cancel pending tasks and stop the shared loop."""
    global _loop, _thread, _stopped, _loop_ready, _loop_thread_ident
    if _loop_thread_ident is not None and threading.get_ident() == _loop_thread_ident:
        raise RuntimeError("shutdown_loop must not be called from the loop thread")
    with _lock:
        loop = _loop
        thread = _thread
        if not loop or not thread:
            _stopped = True
            return
        if loop.is_closed():
            _stopped = True
            return
        _stopped = True

    async def _shutdown():
        current = asyncio.current_task()
        tasks = [t for t in asyncio.all_tasks() if t is not current and not t.done()]
        for t in tasks:
            t.cancel()
        if tasks:
            await asyncio.gather(*tasks, return_exceptions=True)
        await loop.shutdown_asyncgens()
        loop.stop()

    try:
        fut = asyncio.run_coroutine_threadsafe(_shutdown(), loop)
        try:
            fut.result(timeout=timeout)
        except TimeoutError:
            fut.cancel()
            with contextlib.suppress(Exception):
                loop.call_soon_threadsafe(loop.stop)
        except Exception:
            L.warning("shutdown_loop failed", exc_info=True)
    finally:
        if thread.is_alive():
            thread.join(timeout=timeout)
        if not loop.is_closed():
            loop.close()
        _loop = None
        _thread = None
        _loop_ready = None
        _loop_thread_ident = None


@dataclass
class AppContext:
    trigger_gateway: TriggerGateway
    result_store: OutputManager | ResultStore
    queue_mgr: DetectQueueManager
    result_sink: Callable[[OutputRecord, Optional[Tuple[bytes, str]]], None]
    modbus_io: Optional[object] = None


class TriggerHandle(Protocol):
    def start(self) -> None: ...
    def stop(self) -> None: ...


class SystemRuntime:
    """Coordinates worker lifecycles, outputs, trigger, and health monitoring."""

    def __init__(
        self,
        app_context: AppContext,
        camera_worker: CameraWorker,
        detect_worker: DetectWorker,
        output_mgr: OutputManager,
    ):
        self.app_context = app_context
        self.camera_worker = camera_worker
        self.detect_worker = detect_worker
        self.output_mgr = output_mgr
        self.triggers: list[TriggerHandle] = []

        self._stop_evt = threading.Event()
        self._heartbeat_stop = threading.Event()
        self._heartbeat_thread: threading.Thread | None = None
        self._detect_started = False
        self._camera_started = False
        self._stopped = False

    def start(
        self,
        trigger: Optional[TriggerHandle] = None,
        triggers: Optional[list[TriggerHandle]] = None,
    ):
        all_triggers: list[TriggerHandle] = []
        if trigger:
            all_triggers.append(trigger)
        if triggers:
            all_triggers.extend(triggers)
        self.triggers = all_triggers
        self.camera_worker.start()
        self._camera_started = True
        self.detect_worker.start()
        self._detect_started = True

        self.output_mgr.start()
        self._start_heartbeat()

        for t in list(self.triggers):
            try:
                t.start()
            except Exception:
                L.warning("Failed to start trigger", exc_info=True)

    def request_stop(self):
        self._stop_evt.set()

    def run(self, runtime_limit_s: float | None = None):
        start_ts = time.perf_counter()
        try:
            while not self._stop_evt.wait(0.1):
                if (
                    runtime_limit_s is not None
                    and (time.perf_counter() - start_ts) >= runtime_limit_s
                ):
                    L.info(
                        "Runtime limit reached (%ss); shutting down service",
                        runtime_limit_s,
                    )
                    self.request_stop()
        finally:
            self.stop()

    def stop(self):
        if self._stopped:
            return
        self._stopped = True

        try:
            for t in list(self.triggers):
                try:
                    t.stop()
                except Exception:
                    L.warning("Failed to stop trigger", exc_info=True)
        finally:
            try:
                if self._camera_started:
                    self.camera_worker.stop()
            finally:
                if self._detect_started:
                    with contextlib.suppress(Exception):
                        self.detect_worker.stop()
                self._drain_pending_detects(reason="SERVICE_STOP")

        self._stop_heartbeat()
        with contextlib.suppress(Exception):
            self.output_mgr.stop()
        with contextlib.suppress(Exception):
            if self.app_context.result_store is not self.output_mgr:
                self.app_context.result_store.stop()
        with contextlib.suppress(Exception):
            shutdown_loop()

    def reset_system(self):
        """Reset queues, counters, and cached outputs in response to CMD_RESET."""
        try:
            self.app_context.trigger_gateway.reset()
        except Exception:
            L.warning("TriggerGateway reset failed", exc_info=True)
        try:
            self._drain_trigger_queue()
        except Exception:
            L.warning("Trigger queue drain failed", exc_info=True)
        try:
            self.app_context.queue_mgr.clear()
        except Exception:
            L.warning("Detect queue clear failed", exc_info=True)
        try:
            if hasattr(self.output_mgr, "reset"):
                reset_fn = cast(Callable[[], None], getattr(self.output_mgr, "reset"))
                reset_fn()
        except Exception:
            L.warning("OutputManager reset failed", exc_info=True)
        try:
            modbus_io = getattr(self.app_context, "modbus_io", None)
            if modbus_io and hasattr(modbus_io, "reset_outputs"):
                modbus_io.reset_outputs()
        except Exception:
            L.warning("ModbusIO reset failed", exc_info=True)
        try:
            id_mgr = getattr(self.camera_worker, "id_manager", None)
            if hasattr(id_mgr, "reset"):
                reset_fn = cast(Callable[[], None], getattr(id_mgr, "reset"))
                reset_fn()
        except Exception:
            L.warning("ID manager reset failed", exc_info=True)

    def _drain_pending_detects(self, reason: str = "SERVICE_STOP"):
        q = getattr(self.app_context.queue_mgr, "queue", None)
        result_sink = self.app_context.result_sink
        if q is None or result_sink is None:
            return
        while True:
            try:
                task = q.get_nowait()
            except queue.Empty:
                break
            try:
                now = datetime.now(timezone.utc)
                rec = OutputRecord(
                    trigger_seq=task.frame_id,
                    source=getattr(task, "source", ""),
                    device_id=getattr(task, "device_id", ""),
                    result="ERROR",
                    triggered_at=task.triggered_at,
                    captured_at=task.captured_at,
                    detected_at=now,
                    message=reason,
                    result_code=reason,
                    duration_ms=(time.perf_counter() - task.t0) * 1000,
                    remark=reason.lower(),
                )
                result_sink(rec, None)
            except Exception:
                L.exception("Failed to mark pending detect task as NG on stop")
            finally:
                with contextlib.suppress(Exception):
                    q.task_done()

    def _drain_trigger_queue(self):
        q = getattr(self.app_context.trigger_gateway, "trigger_queue", None)
        if q is None:
            return
        while True:
            try:
                q.get_nowait()
            except queue.Empty:
                break
            finally:
                with contextlib.suppress(Exception):
                    q.task_done()

    def _start_heartbeat(self):
        if self._heartbeat_thread:
            return
        self._heartbeat_stop.clear()
        self._heartbeat_thread = threading.Thread(
            target=self._heartbeat_loop, daemon=True
        )
        self._heartbeat_thread.start()

    def _stop_heartbeat(self):
        self._heartbeat_stop.set()
        thread = self._heartbeat_thread
        self._heartbeat_thread = None
        if thread:
            thread.join(timeout=1.0)

    def _heartbeat_loop(self):
        while not self._heartbeat_stop.wait(1.0):
            try:
                self.output_mgr.tick()
            except Exception:
                L.warning("Heartbeat tick failed", exc_info=True)


def build_runtime(
    camera,
    save_dir: str,
    history_size: int = 10,
    debounce_ms: float = 10.0,
    http_host: str = "0.0.0.0",
    http_port: int = 8080,
    enable_http: bool = True,
    max_pending_triggers: int = 50,
    max_pending_trigger_events: int = 1,
    enable_modbus: bool = False,
    enable_modbus_io: bool | None = None,
    modbus_host: str = "0.0.0.0",
    modbus_port: int = 5020,
    coil_offset: int = 800,
    di_offset: int = 800,
    ir_offset: int = 50,
    modbus_heartbeat_ms: int = 1000,
    write_csv: bool = True,
    detector=None,
    detect_timeout_ms: int = 2000,
    enable_preview: bool = True,
    trigger_cfg: TriggerConfig | None = None,
):
    from .worker import (
        CameraWorker,
        DetectQueueManager,
        DetectWorker,
        GlobalIdManager,
        make_queue_overflow_record,
    )
    from output.manager import OutputManager, ResultStore

    # Trigger events are intentionally kept almost non-buffered by default (1-slot queue).
    trigger_queue_capacity = max(1, int(max_pending_trigger_events))
    detect_queue_capacity = max(1, int(max_pending_triggers))
    trigger_queue: queue.Queue = queue.Queue(maxsize=trigger_queue_capacity)
    id_mgr = GlobalIdManager()
    result_store = ResultStore(
        base_dir=save_dir, max_records=history_size, write_csv=write_csv
    )
    output_mgr = OutputManager(result_store)
    queue_mgr = DetectQueueManager(maxsize=detect_queue_capacity)
    if detector is None:
        raise ValueError("detector is required")
    trigger_cfg = trigger_cfg or TriggerConfig()
    ip_whitelist = set(trigger_cfg.ip_whitelist) if trigger_cfg.ip_whitelist else None

    def on_trigger_overflow(dropped_event):
        """Record NG for triggers dropped before acquisition."""
        now_dt = datetime.now(timezone.utc)
        source = getattr(dropped_event, "source", "")
        triggered_at = getattr(dropped_event, "triggered_at", now_dt)
        t0 = None
        try:
            t0_val = getattr(dropped_event, "monotonic_ms", None)
            t0 = (t0_val / 1000.0) if isinstance(t0_val, (int, float)) else None
        except Exception:
            t0 = None
        frame_id = getattr(dropped_event, "trigger_seq", None) or id_mgr.next_id()
        rec = make_queue_overflow_record(
            frame_id=frame_id,
            source=source,
            device_id="",
            triggered_at=triggered_at,
            captured_at=triggered_at,
            detected_at=now_dt,
            t0=t0,
            remark=f"trigger_queue_overflow source={source}",
        )
        output_mgr.publish(rec, None)

    app_context = AppContext(
        trigger_gateway=TriggerGateway(
            trigger_queue,
            debounce_ms=debounce_ms,
            min_interval_ms=float(
                getattr(trigger_cfg, "global_min_interval_ms", 0.0) or 0.0
            ),
            high_priority_cooldown_ms=float(
                getattr(trigger_cfg, "high_priority_cooldown_ms", 0.0) or 0.0
            ),
            high_priority_sources=set(
                getattr(trigger_cfg, "high_priority_sources", []) or []
            ),
            low_priority_sources=set(
                getattr(trigger_cfg, "low_priority_sources", []) or []
            ),
            ip_whitelist=ip_whitelist,
            on_overflow=on_trigger_overflow,
        ),
        result_store=output_mgr,
        queue_mgr=queue_mgr,
        result_sink=output_mgr.publish,
    )

    modbus_io = None
    if enable_http:
        from output.hmi import HmiOutput

        project_root = os.path.dirname(os.path.dirname(__file__))
        index_path = os.path.join(project_root, "output", "web", "index.html")
        hmi_output = HmiOutput(
            http_host,
            http_port,
            app_context,
            index_path=index_path,
            task_reg=output_mgr.register_task,
        )
        output_mgr.add_channel(hmi_output)
    modbus_io_enabled = bool(
        enable_modbus if enable_modbus_io is None else enable_modbus_io
    )
    if modbus_io_enabled:
        from core.modbus_io import ModbusIO

        modbus_io = ModbusIO(
            modbus_host,
            modbus_port,
            coil_offset=coil_offset,
            di_offset=di_offset,
            ir_offset=ir_offset,
            heartbeat_ms=modbus_heartbeat_ms,
            task_reg=output_mgr.register_task,
        )
    if enable_modbus and modbus_io:
        from output.modbus import ModbusOutput

        modbus_output = ModbusOutput(modbus_io)
        output_mgr.add_channel(modbus_output)

    camera_worker = CameraWorker(
        camera, id_mgr, trigger_queue, queue_mgr, result_sink=output_mgr.publish
    )
    detect_worker = DetectWorker(
        queue_mgr,
        result_sink=output_mgr.publish,
        detector=detector,
        timeout_ms=detect_timeout_ms,
        enable_preview=enable_preview,
    )
    runtime = SystemRuntime(app_context, camera_worker, detect_worker, output_mgr)
    if modbus_io:
        runtime.app_context.modbus_io = modbus_io
    return runtime


__all__ = [
    "BaseWorker",
    "run_async",
    "spawn_background_task",
    "shutdown_loop",
    "AppContext",
    "SystemRuntime",
    "build_runtime",
]
