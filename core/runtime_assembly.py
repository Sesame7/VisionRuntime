"""Runtime assembly helpers: queues, workers, outputs, and wiring."""

from __future__ import annotations

import os
import queue
from datetime import datetime, timezone
from typing import Callable, Optional, Tuple

from core.contracts import OutputRecord
from trigger import TriggerConfig, TriggerGateway
from utils.lifecycle import LoopRunner

from .runtime import AppContext, RuntimeBuildConfig, SystemRuntime

DEFAULT_TRIGGER_QUEUE_CAPACITY = 2


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
        modbus_heartbeat_ms=cfg.comm.modbus.heartbeat_ms,
        write_csv=cfg.output.write_csv,
        detect_timeout_ms=cfg.detect.timeout_ms,
        preview_enabled=bool(cfg.detect.preview_enabled),
        preview_max_edge=int(cfg.detect.preview_max_edge),
    )


def _build_trigger_queue() -> queue.Queue:
    # Trigger events are intentionally kept lightly buffered by default.
    return queue.Queue(maxsize=DEFAULT_TRIGGER_QUEUE_CAPACITY)


def _build_output_manager(cfg: RuntimeBuildConfig, *, loop_runner: LoopRunner):
    from output.manager import OutputManager, ResultStore

    result_store = ResultStore(
        base_dir=cfg.save_dir,
        max_records=cfg.history_size,
        write_csv=cfg.write_csv,
    )
    return OutputManager(result_store, loop_runner=loop_runner)


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
                task_reg=output_mgr.adopt_task,
                loop_runner=loop_runner,
            )
        )

    if modbus_io_enabled:
        from utils.modbus.modbus_server_io import ModbusIO

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

        output_mgr.add_channel(ModbusOutput(modbus_io))

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
    output_mgr = _build_output_manager(cfg, loop_runner=loop_runner)
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


__all__ = [
    "build_runtime_config_from_loaded_config",
    "build_runtime",
    "build_runtime_from_loaded_config",
]
