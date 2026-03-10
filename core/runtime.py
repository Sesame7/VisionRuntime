"""Core runtime: SystemRuntime orchestration and runtime wiring."""

from __future__ import annotations

import logging
import os
import queue
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
    enable_modbus_trigger: bool = False
    enable_modbus_output: bool = False
    modbus_host: str = "0.0.0.0"
    modbus_port: int = 5020
    coil_offset: int = 800
    di_offset: int = 800
    ir_offset: int = 50
    hr_offset: int = 50
    modbus_heartbeat_ms: int = 1000
    write_csv: bool = True
    detect_timeout_ms: int = 2000
    preview_enabled: bool = True
    preview_max_edge: int = 1280
    enable_overlay_archive: bool = False
    overlay_archive_base_dir: str = "overlay_archive"
    overlay_archive_default_batch_id: str = "=@NO-BATCH_+"
    overlay_archive_only_ng: bool = True


def build_runtime_config_from_loaded_config(cfg) -> RuntimeBuildConfig:
    return RuntimeBuildConfig(
        save_dir=cfg.runtime.save_dir,
        history_size=cfg.output.hmi.history_size,
        debounce_ms=cfg.trigger.debounce_ms,
        http_host=cfg.comm.http.host,
        http_port=cfg.comm.http.port,
        enable_http=cfg.output.hmi.enabled,
        detect_queue_capacity=cfg.runtime.detect_queue_capacity,
        enable_modbus_trigger=bool(cfg.trigger.modbus.enabled),
        enable_modbus_output=bool(cfg.output.modbus.enabled),
        modbus_host=cfg.comm.modbus.host,
        modbus_port=cfg.comm.modbus.port,
        coil_offset=cfg.comm.modbus.coil_offset,
        di_offset=cfg.comm.modbus.di_offset,
        ir_offset=cfg.comm.modbus.ir_offset,
        hr_offset=cfg.comm.modbus.hr_offset,
        modbus_heartbeat_ms=cfg.comm.modbus.heartbeat_ms,
        write_csv=cfg.output.write_csv,
        detect_timeout_ms=cfg.detect.timeout_ms,
        preview_enabled=bool(cfg.detect.preview_enabled),
        preview_max_edge=int(cfg.detect.preview_max_edge),
        enable_overlay_archive=bool(cfg.output.overlay_archive.enabled),
        overlay_archive_base_dir=str(cfg.output.overlay_archive.base_dir),
        overlay_archive_default_batch_id=str(
            cfg.output.overlay_archive.default_batch_id
        ),
        overlay_archive_only_ng=bool(cfg.output.overlay_archive.only_ng),
    )


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


class RecipeControlApi(Protocol):
    def recipe_status(self) -> dict[str, Any]: ...
    def switch_recipe_slot(self, slot_command: int) -> tuple[bool, str]: ...


class BatchControlApi(Protocol):
    def batch_status(self) -> dict[str, Any]: ...
    def set_batch(self, batch_id: str) -> tuple[bool, str]: ...


@dataclass
class AppContext:
    trigger_gateway: TriggerGateway
    results: ResultReadApi
    detect_queue_mgr: DetectQueueManager
    modbus_io: Optional["ModbusIO"] = None
    recipe_api: Optional[RecipeControlApi] = None
    batch_api: Optional[BatchControlApi] = None


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
        self._recipe_manager: Any | None = None
        self._recipe_switch_guard_ms: int = 0
        self._recipe_active: str = ""
        self._recipe_switching = False
        self._recipe_lock = threading.Lock()

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
        """Reset queues, counters, and cached outputs with best-effort cleanup."""
        self.app_context.trigger_gateway.reset()
        self._drain_trigger_queue()
        self.app_context.detect_queue_mgr.clear()
        self.output_mgr.reset()
        modbus_io = self.app_context.modbus_io
        if modbus_io:
            modbus_io.reset_outputs()

    def configure_recipe_switching(
        self, recipe_manager, *, switch_guard_ms: int
    ) -> None:
        self._recipe_manager = recipe_manager
        with self._recipe_lock:
            self._recipe_switch_guard_ms = max(0, int(switch_guard_ms))
            self._recipe_active = str(recipe_manager.active_recipe())
            self._recipe_switching = False

    def recipe_status(self) -> dict[str, Any]:
        manager = self._recipe_manager
        with self._recipe_lock:
            active = self._recipe_active
            switching = bool(self._recipe_switching)
        payload = {
            "current_slot": 0,
            "total_slots": 0,
            "slots": [],
            "switching": switching,
        }
        if manager is not None:
            slots = manager.list_slots()
            payload["slots"] = slots
            payload["total_slots"] = len(slots)
            if active:
                try:
                    payload["current_slot"] = int(manager.slot_of_recipe(active))
                except Exception:
                    payload["current_slot"] = 0
        return payload

    def switch_recipe_slot(self, slot_command: int) -> tuple[bool, str]:
        manager = self._recipe_manager
        if manager is None:
            return False, "recipe switching is not configured"
        try:
            command = int(slot_command)
        except Exception:
            return False, "recipe slot command must be an integer"
        if command < 0:
            return False, "recipe slot command must be >= 0"
        if command == 0:
            return True, "recipe switch ignored: slot command is 0"

        total_slots = int(manager.recipe_count())
        if total_slots <= 0:
            return False, "no recipes are available"
        target_slot = ((command - 1) % total_slots) + 1
        try:
            target = str(manager.recipe_name_at_slot(target_slot))
        except Exception as exc:
            return False, str(exc)
        try:
            ok, message = self.switch_recipe(target)
        except Exception as exc:
            return False, str(exc) or type(exc).__name__
        if ok:
            return True, f"{message} slot={target_slot}"
        return False, message

    def switch_recipe(self, recipe_name: str) -> tuple[bool, str]:
        target = str(recipe_name or "").strip()
        if not target:
            return False, "recipe name is required"
        manager = self._recipe_manager
        if manager is None:
            return False, "recipe switching is not configured"

        with self._recipe_lock:
            if self._recipe_switching:
                return False, "recipe switch already in progress"
            current = self._recipe_active or str(manager.active_recipe())
            self._recipe_switching = True
            guard_ms = self._recipe_switch_guard_ms

        settle_timeout_s = self._recipe_switch_settle_timeout_s(guard_ms)
        t0 = time.perf_counter()
        L.info(
            "Recipe switch start from=%s to=%s guard_ms=%d settle_timeout_s=%.2f",
            current,
            target,
            guard_ms,
            settle_timeout_s,
        )
        self.app_context.trigger_gateway.set_accepting(False)
        try:
            if guard_ms > 0:
                time.sleep(guard_ms / 1000.0)
            self._drain_trigger_queue()
            if not self.camera_worker.wait_idle(timeout_s=settle_timeout_s):
                raise TimeoutError("camera worker did not become idle before switch")
            self.app_context.detect_queue_mgr.clear()
            if not self.detect_worker.wait_idle(timeout_s=settle_timeout_s):
                raise TimeoutError("detect worker did not become idle before switch")
            detector = manager.build_detector_for(target)
            self.detect_worker.replace_detector(detector)
            manager.set_active_recipe(target)
            self.app_context.trigger_gateway.reset()
            self.output_mgr.reset()
            modbus_io = self.app_context.modbus_io
            if modbus_io is not None:
                modbus_io.reset_outputs()
            elapsed_ms = (time.perf_counter() - t0) * 1000.0
            with self._recipe_lock:
                self._recipe_active = target
            L.info(
                "Recipe switch success from=%s to=%s elapsed=%.1fms",
                current,
                target,
                elapsed_ms,
            )
            return True, f"switched to {target}"
        finally:
            self.app_context.trigger_gateway.set_accepting(True)
            with self._recipe_lock:
                self._recipe_switching = False

    def _recipe_switch_settle_timeout_s(self, guard_ms: int) -> float:
        guard_s = max(0.0, float(guard_ms) / 1000.0)
        detect_timeout_s = (
            max(0.0, float(getattr(self.detect_worker, "timeout_ms", 0.0) or 0.0))
            / 1000.0
        )
        cam_cfg = getattr(getattr(self.camera_worker, "camera", None), "cfg", None)
        camera_grab_s = (
            max(0.0, float(getattr(cam_cfg, "grab_timeout_ms", 0.0) or 0.0)) / 1000.0
        )
        # Allow one guarded switch window plus the slower in-flight stage and margin.
        return max(2.0, guard_s + max(detect_timeout_s, camera_grab_s) + 1.0)

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
    from .worker import make_queue_overflow_record

    loop_runner = loop_runner or LoopRunner()
    cfg = config
    if detector is None:
        raise ValueError("detector is required")
    trigger_cfg = trigger_cfg or TriggerConfig()
    trigger_queue = _build_trigger_queue()
    output_mgr = _build_output_manager(cfg)
    detect_queue_mgr = _build_detect_queue_manager(
        cfg, output_mgr_publish=output_mgr.publish
    )
    app_context = _build_app_context(
        cfg,
        trigger_cfg=trigger_cfg,
        trigger_queue=trigger_queue,
        detect_queue_mgr=detect_queue_mgr,
        output_mgr=output_mgr,
        make_queue_overflow_record_fn=make_queue_overflow_record,
    )
    modbus_io = _wire_output_channels(
        cfg,
        app_context=app_context,
        output_mgr=output_mgr,
        loop_runner=loop_runner,
    )
    camera_worker, detect_worker = _build_workers(
        camera,
        detector=detector,
        trigger_queue=trigger_queue,
        detect_queue_mgr=detect_queue_mgr,
        output_mgr_publish=output_mgr.publish,
        detect_timeout_ms=cfg.detect_timeout_ms,
        preview_enabled=cfg.preview_enabled,
        preview_max_edge=cfg.preview_max_edge,
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


def _build_trigger_queue() -> queue.Queue:
    # Trigger events are intentionally kept lightly buffered by default.
    return queue.Queue(maxsize=DEFAULT_TRIGGER_QUEUE_CAPACITY)


def _build_output_manager(cfg: RuntimeBuildConfig):
    from output.manager import OutputManager, ResultStore

    result_store = ResultStore(
        base_dir=cfg.save_dir,
        max_records=cfg.history_size,
        write_csv=cfg.write_csv,
    )
    return OutputManager(result_store)


def _build_detect_queue_manager(
    cfg: RuntimeBuildConfig,
    *,
    output_mgr_publish: Callable[[OutputRecord, Optional[Tuple[bytes, str]]], None],
):
    from .worker import DetectQueueManager

    return DetectQueueManager(
        maxsize=max(1, int(cfg.detect_queue_capacity)),
        result_sink=output_mgr_publish,
    )


def _build_app_context(
    cfg: RuntimeBuildConfig,
    *,
    trigger_cfg: TriggerConfig,
    trigger_queue: queue.Queue,
    detect_queue_mgr,
    output_mgr,
    make_queue_overflow_record_fn,
) -> AppContext:
    ip_whitelist = set(trigger_cfg.ip_whitelist) if trigger_cfg.ip_whitelist else None

    def on_trigger_overflow(dropped_event):
        """Record ERROR for triggers dropped before acquisition."""
        now_dt = datetime.now(timezone.utc)
        source = dropped_event.source
        triggered_at = dropped_event.triggered_at or now_dt
        start_monotonic_val = dropped_event.monotonic_ms
        start_monotonic_s = (
            (start_monotonic_val / 1000.0)
            if isinstance(start_monotonic_val, (int, float))
            else None
        )
        frame_id = int(getattr(dropped_event, "trigger_seq", 0) or 0)
        if frame_id <= 0:
            raise RuntimeError("Trigger overflow event missing trigger_seq")
        rec = make_queue_overflow_record_fn(
            frame_id=frame_id,
            source=source,
            device_id="",
            triggered_at=triggered_at,
            captured_at=triggered_at,
            detected_at=now_dt,
            start_monotonic_s=start_monotonic_s,
            remark=f"trigger_queue_overflow source={source}",
        )
        output_mgr.publish(rec, None)

    return AppContext(
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
        detect_queue_mgr=detect_queue_mgr,
    )


def _wire_output_channels(
    cfg: RuntimeBuildConfig,
    *,
    app_context: AppContext,
    output_mgr,
    loop_runner: LoopRunner,
):
    modbus_io_enabled = bool(cfg.enable_modbus_trigger or cfg.enable_modbus_output)
    modbus_output_enabled = bool(cfg.enable_modbus_output)
    modbus_io = None
    from output.overlay_archive import BatchState

    batch_state = BatchState(default_batch_id=cfg.overlay_archive_default_batch_id)
    app_context.batch_api = batch_state

    if cfg.enable_http:
        from output.hmi import HmiOutput

        project_root = os.path.dirname(os.path.dirname(__file__))
        index_path = os.path.join(project_root, "output", "web", "index.html")
        output_mgr.add_channel(
            HmiOutput(
                cfg.http_host,
                cfg.http_port,
                app_context,
                index_path=index_path,
                loop_runner=loop_runner,
            )
        )

    if cfg.enable_overlay_archive and cfg.preview_enabled:
        from output.overlay_archive import OverlayArchiveOutput

        archive_base_dir = str(cfg.overlay_archive_base_dir or "").strip()
        if not archive_base_dir:
            raise ValueError("overlay_archive_base_dir must be a non-empty path")
        if not os.path.isabs(archive_base_dir):
            archive_base_dir = os.path.join(cfg.save_dir, archive_base_dir)
        overlay_archive = OverlayArchiveOutput(
            base_dir=archive_base_dir,
            batch_state=batch_state,
            only_ng=cfg.overlay_archive_only_ng,
        )
        output_mgr.add_channel(overlay_archive)

    if modbus_io_enabled:
        from output.modbus import ModbusIoLifecycleChannel
        from utils.modbus.modbus_server_io import ModbusIO

        modbus_io = ModbusIO(
            cfg.modbus_host,
            cfg.modbus_port,
            coil_offset=cfg.coil_offset,
            di_offset=cfg.di_offset,
            ir_offset=cfg.ir_offset,
            hr_offset=cfg.hr_offset,
            heartbeat_ms=cfg.modbus_heartbeat_ms,
            loop_runner=loop_runner,
        )
        # Modbus server lifecycle is independent from result channel enablement.
        output_mgr.add_channel(ModbusIoLifecycleChannel(modbus_io))

    if modbus_output_enabled and modbus_io:
        from output.modbus import ModbusOutput

        output_mgr.add_channel(ModbusOutput(modbus_io, stats_provider=output_mgr.stats))

    return modbus_io


def _build_workers(
    camera,
    *,
    detector,
    trigger_queue: queue.Queue,
    detect_queue_mgr,
    output_mgr_publish: Callable[[OutputRecord, Optional[Tuple[bytes, str]]], None],
    detect_timeout_ms: int,
    preview_enabled: bool,
    preview_max_edge: int,
):
    from .worker import CameraWorker, DetectWorker

    camera_worker = CameraWorker(
        camera,
        trigger_queue,
        detect_queue_mgr,
        result_sink=output_mgr_publish,
    )
    detect_worker = DetectWorker(
        detect_queue_mgr,
        result_sink=output_mgr_publish,
        detector=detector,
        timeout_ms=detect_timeout_ms,
        preview_enabled=preview_enabled,
        preview_max_edge=preview_max_edge,
    )
    return camera_worker, detect_worker


__all__ = [
    "AppContext",
    "BatchControlApi",
    "RecipeControlApi",
    "ResultReadApi",
    "RuntimeBuildConfig",
    "build_runtime_config_from_loaded_config",
    "build_runtime_from_loaded_config",
    "SystemRuntime",
    "build_runtime",
]
