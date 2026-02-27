"""Core runtime: SystemRuntime orchestration and runtime assembly."""

from __future__ import annotations

import logging
import threading
import time
from contextlib import ExitStack
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
from utils.lifecycle import LoopRunner, drain_queue_nowait_with_task_done
from trigger import TriggerConfig, TriggerGateway

if TYPE_CHECKING:  # pragma: no cover
    from utils.modbus.modbus_server_io import ModbusIO
    from .worker import CameraWorker, DetectQueueManager, DetectWorker
    from output.manager import OutputManager

L = logging.getLogger("vision_runtime.runtime")


@dataclass
class RuntimeBuildConfig:
    save_dir: str
    history_size: int = 10
    debounce_ms: float = 10.0
    http_host: str = "0.0.0.0"
    http_port: int = 8080
    enable_http: bool = True
    detect_queue_capacity: int = 50
    enable_modbus_trigger: bool = False
    enable_modbus_output: bool = False
    modbus_host: str = "0.0.0.0"
    modbus_port: int = 5020
    coil_offset: int = 800
    di_offset: int = 800
    ir_offset: int = 50
    modbus_heartbeat_ms: int = 1000
    write_csv: bool = True
    detect_timeout_ms: int = 2000
    preview_enabled: bool = True
    preview_max_edge: int = 1280


def build_runtime_config_from_loaded_config(cfg) -> RuntimeBuildConfig:
    from .runtime_assembly import build_runtime_config_from_loaded_config as _impl

    return _impl(cfg)


class TriggerHandle(Protocol):
    def start(self) -> None: ...
    def stop(self) -> None: ...
    def raise_if_failed(self) -> None: ...


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
    detect_queue_mgr: DetectQueueManager
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
        self._camera_session_stack: ExitStack | None = None
        self._started = False
        self._stopped = False

    def start(
        self,
        triggers: Optional[list[TriggerHandle]] = None,
    ):
        if self._started:
            raise RuntimeError(
                "SystemRuntime is single-use; start() may only be called once"
            )
        if self._stopped:
            raise RuntimeError("SystemRuntime is stopped and cannot be started again")
        self._started = True
        self.triggers = list(triggers or [])
        try:
            self._enter_camera_session()

            self.camera_worker.start()
            self.detect_worker.start()

            self.output_mgr.start()

            for t in list(self.triggers):
                t.start()
        except Exception:
            L.exception("Runtime start failed; rolling back partial startup")
            try:
                self.stop()
            except Exception:
                L.exception("Runtime rollback stop failed")
            raise

    def request_stop(self):
        self._stop_evt.set()

    def run(self, runtime_limit_s: float | None = None):
        if not self._started:
            raise RuntimeError("SystemRuntime.run() requires start() first")
        start_ts = time.perf_counter()
        next_heartbeat_ts = start_ts + 1.0
        try:
            while not self._stop_evt.wait(0.1):
                now_ts = time.perf_counter()
                if now_ts >= next_heartbeat_ts:
                    self.output_mgr.tick()
                    next_heartbeat_ts = now_ts + 1.0
                self._raise_if_worker_stopped()
                self._raise_if_trigger_stopped()
                self._raise_if_output_stopped()
                if (
                    runtime_limit_s is not None
                    and (now_ts - start_ts) >= runtime_limit_s
                ):
                    L.info(
                        "Runtime limit reached (%ss); shutting down service",
                        runtime_limit_s,
                    )
                    self.request_stop()
        finally:
            self.stop()

    def _raise_if_worker_stopped(self):
        if self.camera_worker.has_started and not self.camera_worker.is_alive:
            err = self.camera_worker.last_error
            if err is not None:
                raise RuntimeError(
                    f"CameraWorker stopped unexpectedly ({type(err).__name__})"
                ) from err
            raise RuntimeError("CameraWorker stopped unexpectedly")
        if self.detect_worker.has_started and not self.detect_worker.is_alive:
            err = self.detect_worker.last_error
            if err is not None:
                raise RuntimeError(
                    f"DetectWorker stopped unexpectedly ({type(err).__name__})"
                ) from err
            raise RuntimeError("DetectWorker stopped unexpectedly")

    def _raise_if_trigger_stopped(self):
        for trig in self.triggers:
            trig.raise_if_failed()

    def _raise_if_output_stopped(self):
        self.output_mgr.raise_if_failed()

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

        def _run_stage(name: str, fn: Callable[[], None]):
            try:
                fn()
            except Exception:
                L.exception("Shutdown stage failed: %s", name)
            finally:
                _log_stage(name)

        def _stop_triggers():
            for t in list(self.triggers):
                try:
                    t.stop()
                except Exception:
                    L.exception("Trigger stop failed: %r", t)

        def _stop_workers_and_pending():
            if self.camera_worker.has_started:
                self.camera_worker.stop()
            if self.detect_worker.has_started:
                self.detect_worker.stop()
            self._drain_pending_detects(reason="SERVICE_STOP")

        _run_stage("triggers", _stop_triggers)
        _run_stage("workers_and_pending_detects", _stop_workers_and_pending)
        _run_stage("output_manager", self.output_mgr.stop)
        _run_stage("async_loop", self.loop_runner.shutdown_loop)
        _run_stage("camera_session", self._exit_camera_session)
        L.debug(
            "Shutdown stage=total elapsed=%.1fms",
            (time.perf_counter() - stop_t0) * 1000,
        )

    def reset_system(self):
        """Reset queues, counters, and cached outputs in response to CMD_RESET."""
        self.app_context.trigger_gateway.reset()
        self._drain_trigger_queue()
        self.app_context.detect_queue_mgr.clear()
        self.output_mgr.reset()
        modbus_io = self.app_context.modbus_io
        if modbus_io:
            modbus_io.reset_outputs()

    def _drain_pending_detects(self, reason: str = "SERVICE_STOP"):
        q = self.app_context.detect_queue_mgr.queue

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
                duration_ms=(time.perf_counter() - task.start_monotonic_s) * 1000,
                remark=reason.lower(),
            )
            self.output_mgr.publish(rec, None)

        drain_queue_nowait_with_task_done(q, on_item=_mark_pending)

    def _drain_trigger_queue(self):
        drain_queue_nowait_with_task_done(
            self.app_context.trigger_gateway.trigger_queue
        )

    def _enter_camera_session(self):
        if self._camera_session_stack is not None:
            return
        stack = ExitStack()
        stack.enter_context(self.camera_worker.camera.session())
        self._camera_session_stack = stack

    def _exit_camera_session(self):
        stack = self._camera_session_stack
        if stack is None:
            return
        self._camera_session_stack = None
        stack.close()


def build_runtime(
    camera,
    *,
    config: RuntimeBuildConfig,
    detector,
    trigger_cfg: TriggerConfig | None = None,
    loop_runner: LoopRunner | None = None,
):
    from .runtime_assembly import build_runtime as _impl

    return _impl(
        camera,
        config=config,
        detector=detector,
        trigger_cfg=trigger_cfg,
        loop_runner=loop_runner,
    )


def build_runtime_from_loaded_config(
    camera,
    cfg,
    *,
    detector,
    trigger_cfg: TriggerConfig | None = None,
    loop_runner: LoopRunner | None = None,
):
    from .runtime_assembly import build_runtime_from_loaded_config as _impl

    return _impl(
        camera,
        cfg,
        detector=detector,
        trigger_cfg=trigger_cfg,
        loop_runner=loop_runner,
    )


__all__ = [
    "AppContext",
    "ResultReadApi",
    "RuntimeBuildConfig",
    "build_runtime_config_from_loaded_config",
    "build_runtime_from_loaded_config",
    "SystemRuntime",
    "build_runtime",
]
