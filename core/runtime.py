"""Core runtime: BaseWorker and SystemRuntime orchestration."""

from __future__ import annotations

import logging
import os
import queue
import threading
import time
from dataclasses import dataclass
from datetime import datetime, timezone
from typing import (
    Any,
    Callable,
    Optional,
    Protocol,
    Tuple,
    TYPE_CHECKING,
)

from core.contracts import OutputRecord
from core.lifecycle import LoopRunner
from core.queue_utils import drain_queue_nowait
from trigger import TriggerConfig, TriggerGateway

if TYPE_CHECKING:  # pragma: no cover
    from core.modbus_io import ModbusIO
    from .worker import CameraWorker, DetectQueueManager, DetectWorker
    from output.manager import OutputManager

L = logging.getLogger("vision_runtime.runtime")

DEFAULT_TRIGGER_QUEUE_CAPACITY = 2


@dataclass
class RuntimeBuildConfig:
    save_dir: str
    history_size: int = 10
    debounce_ms: float = 10.0
    http_host: str = "0.0.0.0"
    http_port: int = 8080
    enable_http: bool = True
    detect_queue_capacity: int = 50
    # "off" | "trigger" | "output" | "both"
    modbus_mode: str = "off"
    modbus_host: str = "0.0.0.0"
    modbus_port: int = 5020
    coil_offset: int = 800
    di_offset: int = 800
    ir_offset: int = 50
    modbus_heartbeat_ms: int = 1000
    write_csv: bool = True
    detect_timeout_ms: int = 2000
    preview_enabled: bool = True


def build_runtime_config_from_loaded_config(cfg) -> RuntimeBuildConfig:
    modbus_trigger_enabled = bool(cfg.trigger.modbus.enabled)
    modbus_output_enabled = bool(cfg.output.modbus.enabled)
    if modbus_trigger_enabled and modbus_output_enabled:
        modbus_mode = "both"
    elif modbus_output_enabled:
        modbus_mode = "output"
    elif modbus_trigger_enabled:
        modbus_mode = "trigger"
    else:
        modbus_mode = "off"
    return RuntimeBuildConfig(
        save_dir=cfg.runtime.save_dir,
        history_size=cfg.output.hmi.history_size,
        debounce_ms=cfg.trigger.debounce_ms,
        http_host=cfg.comm.http.host,
        http_port=cfg.comm.http.port,
        enable_http=cfg.output.hmi.enabled,
        detect_queue_capacity=cfg.runtime.detect_queue_capacity,
        modbus_mode=modbus_mode,
        modbus_host=cfg.comm.modbus.host,
        modbus_port=cfg.comm.modbus.port,
        coil_offset=cfg.comm.modbus.coil_offset,
        di_offset=cfg.comm.modbus.di_offset,
        ir_offset=cfg.comm.modbus.ir_offset,
        modbus_heartbeat_ms=cfg.comm.modbus.heartbeat_ms,
        write_csv=cfg.output.write_csv,
        detect_timeout_ms=cfg.detect.timeout_ms,
        preview_enabled=bool(cfg.detect.preview_enabled),
    )


class BaseWorker:
    def __init__(self, name: str = ""):
        self.name = name or self.__class__.__name__
        self._stop_evt = threading.Event()
        self._thread: threading.Thread | None = None
        self._last_error: Exception | None = None

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
        except Exception as e:
            self._last_error = e
            L.exception("%s worker error", self.name)

    @property
    def is_alive(self) -> bool:
        return bool(self._thread and self._thread.is_alive())

    @property
    def last_error(self) -> Exception | None:
        return self._last_error

    def run(self):
        """Override in subclasses with the worker loop."""
        raise NotImplementedError


class TriggerHandle(Protocol):
    def start(self) -> None: ...
    def stop(self) -> None: ...


class ResultReadApi(Protocol):
    @property
    def latest_records(self) -> list[OutputRecord]: ...

    @property
    def max_records(self) -> int: ...

    def latest_overlay(self) -> Optional[Tuple[bytes, str]]: ...

    def stats(self) -> dict[str, Any]: ...

    def heartbeat_seq(self) -> int | None: ...


@dataclass
class AppContext:
    trigger_gateway: TriggerGateway
    results: ResultReadApi
    queue_mgr: DetectQueueManager
    result_sink: Callable[[OutputRecord, Optional[Tuple[bytes, str]]], None]
    modbus_io: Optional["ModbusIO"] = None


class SystemRuntime:
    """Coordinates worker lifecycles, outputs, trigger, and health monitoring."""

    def __init__(
        self,
        app_context: AppContext,
        camera_worker: CameraWorker,
        detect_worker: DetectWorker,
        output_mgr: OutputManager,
        loop_runner: LoopRunner,
    ):
        self.app_context = app_context
        self.camera_worker = camera_worker
        self.detect_worker = detect_worker
        self.output_mgr = output_mgr
        self.loop_runner = loop_runner
        self.triggers: list[TriggerHandle] = []

        self._stop_evt = threading.Event()
        self._heartbeat_stop = threading.Event()
        self._heartbeat_thread: threading.Thread | None = None
        self._detect_started = False
        self._camera_started = False
        self._camera_session_cm: Any | None = None
        self._camera_session_entered = False
        self._stopped = False

    def start(
        self,
        triggers: Optional[list[TriggerHandle]] = None,
    ):
        self.triggers = list(triggers or [])
        self._enter_camera_session()

        self.camera_worker.start()
        self._camera_started = True
        self.detect_worker.start()
        self._detect_started = True

        self.output_mgr.start()
        self._start_heartbeat()

        for t in list(self.triggers):
            t.start()

    def request_stop(self):
        self._stop_evt.set()

    def run(self, runtime_limit_s: float | None = None):
        start_ts = time.perf_counter()
        try:
            while not self._stop_evt.wait(0.1):
                self._raise_if_worker_stopped()
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

    def _raise_if_worker_stopped(self):
        if self._camera_started and not self.camera_worker.is_alive:
            err = self.camera_worker.last_error
            if err is not None:
                raise RuntimeError("CameraWorker stopped unexpectedly") from err
            raise RuntimeError("CameraWorker stopped unexpectedly")
        if self._detect_started and not self.detect_worker.is_alive:
            err = self.detect_worker.last_error
            if err is not None:
                raise RuntimeError("DetectWorker stopped unexpectedly") from err
            raise RuntimeError("DetectWorker stopped unexpectedly")

    def stop(self):
        if self._stopped:
            return
        self._stopped = True
        stop_t0 = time.perf_counter()
        stage_t0 = stop_t0

        def _log_stage(name: str):
            nonlocal stage_t0
            now = time.perf_counter()
            L.debug("Shutdown stage=%s elapsed=%.1fms", name, (now - stage_t0) * 1000)
            stage_t0 = now

        for t in list(self.triggers):
            t.stop()
        _log_stage("triggers")

        if self._camera_started:
            self.camera_worker.stop()
        if self._detect_started:
            self.detect_worker.stop()
        self._drain_pending_detects(reason="SERVICE_STOP")
        _log_stage("workers_and_pending_detects")

        self._stop_heartbeat()
        _log_stage("heartbeat")
        self.output_mgr.stop()
        _log_stage("output_manager")
        self.loop_runner.shutdown_loop()
        _log_stage("async_loop")
        self._exit_camera_session()
        _log_stage("camera_session")
        L.debug(
            "Shutdown stage=total elapsed=%.1fms",
            (time.perf_counter() - stop_t0) * 1000,
        )

    def reset_system(self):
        """Reset queues, counters, and cached outputs in response to CMD_RESET."""
        self.app_context.trigger_gateway.reset()
        self._drain_trigger_queue()
        self.app_context.queue_mgr.clear()
        self.output_mgr.reset()
        modbus_io = self.app_context.modbus_io
        if modbus_io:
            modbus_io.reset_outputs()
        self.camera_worker.id_manager.reset()

    def _drain_pending_detects(self, reason: str = "SERVICE_STOP"):
        q = self.app_context.queue_mgr.queue
        result_sink = self.app_context.result_sink

        def _mark_pending(task):
            now = datetime.now(timezone.utc)
            rec = OutputRecord(
                trigger_seq=task.frame_id,
                source=task.source,
                device_id=task.device_id,
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

        drain_queue_nowait(q, on_item=_mark_pending)

    def _drain_trigger_queue(self):
        drain_queue_nowait(self.app_context.trigger_gateway.trigger_queue)

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
            self.output_mgr.tick()

    def _enter_camera_session(self):
        if self._camera_session_entered:
            return
        cm = self.camera_worker.camera.session()
        self._camera_session_cm = cm
        cm.__enter__()
        self._camera_session_entered = True

    def _exit_camera_session(self):
        if not self._camera_session_entered:
            return
        cm = self._camera_session_cm
        if cm is None:
            raise RuntimeError("camera session context missing during shutdown")
        self._camera_session_cm = None
        self._camera_session_entered = False
        cm.__exit__(None, None, None)


def build_runtime(
    camera,
    *,
    config: RuntimeBuildConfig,
    detector,
    trigger_cfg: TriggerConfig | None = None,
    loop_runner: LoopRunner | None = None,
):
    from .worker import (
        CameraWorker,
        DetectQueueManager,
        DetectWorker,
        GlobalIdManager,
        make_queue_overflow_record,
    )
    from output.manager import OutputManager, ResultStore

    loop_runner = loop_runner or LoopRunner()
    cfg = config
    modbus_mode = str(getattr(cfg, "modbus_mode", "off") or "off").strip().lower()
    if modbus_mode not in {"off", "trigger", "output", "both"}:
        raise ValueError(
            "config.modbus_mode must be one of: off, trigger, output, both"
        )
    modbus_io_enabled = modbus_mode in {"trigger", "output", "both"}
    modbus_output_enabled = modbus_mode in {"output", "both"}
    # Trigger events are intentionally kept lightly buffered by default (2-slot queue).
    trigger_queue_capacity = DEFAULT_TRIGGER_QUEUE_CAPACITY
    detect_queue_capacity = max(1, int(cfg.detect_queue_capacity))
    trigger_queue: queue.Queue = queue.Queue(maxsize=trigger_queue_capacity)
    id_mgr = GlobalIdManager()
    result_store = ResultStore(
        base_dir=cfg.save_dir, max_records=cfg.history_size, write_csv=cfg.write_csv
    )
    output_mgr = OutputManager(result_store, loop_runner=loop_runner)
    queue_mgr = DetectQueueManager(maxsize=detect_queue_capacity)
    if detector is None:
        raise ValueError("detector is required")
    trigger_cfg = trigger_cfg or TriggerConfig()
    ip_whitelist = set(trigger_cfg.ip_whitelist) if trigger_cfg.ip_whitelist else None

    def on_trigger_overflow(dropped_event):
        """Record NG for triggers dropped before acquisition."""
        now_dt = datetime.now(timezone.utc)
        source = dropped_event.source
        triggered_at = dropped_event.triggered_at or now_dt
        t0_val = dropped_event.monotonic_ms
        t0 = (t0_val / 1000.0) if isinstance(t0_val, (int, float)) else None
        frame_id = dropped_event.trigger_seq or id_mgr.next_id()
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
            debounce_ms=cfg.debounce_ms,
            min_interval_ms=float(trigger_cfg.global_min_interval_ms or 0.0),
            high_priority_cooldown_ms=float(
                trigger_cfg.high_priority_cooldown_ms or 0.0
            ),
            high_priority_sources=set(trigger_cfg.high_priority_sources or []),
            low_priority_sources=set(trigger_cfg.low_priority_sources or []),
            ip_whitelist=ip_whitelist,
            on_overflow=on_trigger_overflow,
        ),
        results=output_mgr,
        queue_mgr=queue_mgr,
        result_sink=output_mgr.publish,
    )

    modbus_io = None
    if cfg.enable_http:
        from output.hmi import HmiOutput

        project_root = os.path.dirname(os.path.dirname(__file__))
        index_path = os.path.join(project_root, "output", "web", "index.html")
        hmi_output = HmiOutput(
            cfg.http_host,
            cfg.http_port,
            app_context,
            index_path=index_path,
            task_reg=output_mgr.adopt_task,
            loop_runner=loop_runner,
        )
        output_mgr.add_channel(hmi_output)
    if modbus_io_enabled:
        from core.modbus_io import ModbusIO

        modbus_io = ModbusIO(
            cfg.modbus_host,
            cfg.modbus_port,
            coil_offset=cfg.coil_offset,
            di_offset=cfg.di_offset,
            ir_offset=cfg.ir_offset,
            heartbeat_ms=cfg.modbus_heartbeat_ms,
            task_reg=output_mgr.adopt_task,
            loop_runner=loop_runner,
        )
    if modbus_output_enabled and modbus_io:
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
        timeout_ms=cfg.detect_timeout_ms,
        preview_enabled=cfg.preview_enabled,
    )
    runtime = SystemRuntime(
        app_context,
        camera_worker,
        detect_worker,
        output_mgr,
        loop_runner=loop_runner,
    )
    if modbus_io:
        runtime.app_context.modbus_io = modbus_io
    return runtime


def build_runtime_from_loaded_config(
    camera,
    cfg,
    *,
    detector,
    trigger_cfg: TriggerConfig | None = None,
    loop_runner: LoopRunner | None = None,
):
    runtime_cfg = build_runtime_config_from_loaded_config(cfg)
    return build_runtime(
        camera,
        config=runtime_cfg,
        detector=detector,
        trigger_cfg=trigger_cfg,
        loop_runner=loop_runner,
    )


__all__ = [
    "BaseWorker",
    "AppContext",
    "ResultReadApi",
    "RuntimeBuildConfig",
    "build_runtime_config_from_loaded_config",
    "build_runtime_from_loaded_config",
    "SystemRuntime",
    "build_runtime",
]
