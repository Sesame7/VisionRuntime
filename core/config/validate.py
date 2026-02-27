"""Runtime config value validation."""

from __future__ import annotations

from typing import Any

from .schema import ConfigError, LoadedConfig


def validate_config(cfg: LoadedConfig) -> None:
    # runtime
    _require_int(
        "runtime.detect_queue_capacity", cfg.runtime.detect_queue_capacity, min_v=1
    )
    _require_float("runtime.max_runtime_s", cfg.runtime.max_runtime_s, min_v=0.0)
    _require_int("output.hmi.history_size", cfg.output.hmi.history_size, min_v=1)

    # camera
    _require_int("camera.device_index", cfg.camera.device_index, min_v=0)
    _require_int("camera.grab_timeout_ms", cfg.camera.grab_timeout_ms, min_v=1)
    _require_int("camera.max_retry_per_frame", cfg.camera.max_retry_per_frame, min_v=1)
    _require_int("camera.width", cfg.camera.width, min_v=0)
    _require_int("camera.height", cfg.camera.height, min_v=0)
    _require_int("camera.exposure_us", cfg.camera.exposure_us, min_v=0)
    _require_float("camera.analogue_gain", cfg.camera.analogue_gain, min_v=0.0)
    _require_int("camera.frame_duration_us", cfg.camera.frame_duration_us, min_v=0)
    _require_int("camera.settle_ms", cfg.camera.settle_ms, min_v=0)

    # trigger
    _require_float("trigger.debounce_ms", cfg.trigger.debounce_ms, min_v=0.0)
    _require_float(
        "trigger.global_min_interval_ms", cfg.trigger.global_min_interval_ms, min_v=0.0
    )
    _require_float(
        "trigger.high_priority_cooldown_ms",
        cfg.trigger.high_priority_cooldown_ms,
        min_v=0.0,
    )
    _require_str_list(
        "trigger.high_priority_sources", cfg.trigger.high_priority_sources
    )
    _require_str_list("trigger.low_priority_sources", cfg.trigger.low_priority_sources)
    _require_str_list("trigger.ip_whitelist", cfg.trigger.ip_whitelist)
    _require_int("trigger.modbus.poll_ms", cfg.trigger.modbus.poll_ms, min_v=1)

    # comm
    _require_port("comm.http.port", cfg.comm.http.port)
    _require_port("comm.tcp.port", cfg.comm.tcp.port)
    _require_port("comm.modbus.port", cfg.comm.modbus.port)
    _require_int("comm.modbus.coil_offset", cfg.comm.modbus.coil_offset, min_v=0)
    _require_int("comm.modbus.di_offset", cfg.comm.modbus.di_offset, min_v=0)
    _require_int("comm.modbus.ir_offset", cfg.comm.modbus.ir_offset, min_v=0)
    _require_int("comm.modbus.heartbeat_ms", cfg.comm.modbus.heartbeat_ms, min_v=1)

    # detect
    _require_int("detect.timeout_ms", cfg.detect.timeout_ms, min_v=0)
    _require_int("detect.preview_max_edge", cfg.detect.preview_max_edge, min_v=0)


def _require_int(
    name: str, value: Any, *, min_v: int | None = None, max_v: int | None = None
) -> int:
    iv = int(value)
    if min_v is not None and iv < min_v:
        op = ">=" if min_v != 1 else ">"
        threshold = min_v if min_v != 1 else 0
        raise ConfigError(f"{name} must be {op} {threshold}")
    if max_v is not None and iv > max_v:
        raise ConfigError(f"{name} must be <= {max_v}")
    return iv


def _require_float(
    name: str, value: Any, *, min_v: float | None = None, max_v: float | None = None
) -> float:
    fv = float(value)
    if min_v is not None and fv < min_v:
        raise ConfigError(f"{name} must be >= {min_v:g}")
    if max_v is not None and fv > max_v:
        raise ConfigError(f"{name} must be <= {max_v:g}")
    return fv


def _require_port(name: str, value: Any) -> int:
    return _require_int(name, value, min_v=1, max_v=65535)


def _require_str_list(name: str, value: Any) -> list[str]:
    if not isinstance(value, list):
        raise ConfigError(f"{name} must be a list of strings")
    for i, item in enumerate(value):
        if not isinstance(item, str):
            raise ConfigError(f"{name}[{i}] must be a string")
    return value


__all__ = ["validate_config"]
