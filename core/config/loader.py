"""YAML loader and section builders for runtime configuration."""

from __future__ import annotations

import glob
import importlib
import os
from typing import Any

import yaml

from .schema import (
    CameraConfigBlock,
    CommConfigBlock,
    CommHttpConfigBlock,
    CommModbusConfigBlock,
    CommTcpConfigBlock,
    ConfigError,
    DetectConfigBlock,
    LoadedConfig,
    OutputConfigBlock,
    OutputHmiConfigBlock,
    OutputModbusConfigBlock,
    RuntimeConfig,
    TriggerConfigBlock,
    TriggerModbusConfigBlock,
    TriggerTcpConfigBlock,
)


def load_config(config_dir: str = "config") -> LoadedConfig:
    main_path = _find_main_config(config_dir)
    main_data = _read_yaml(main_path)
    imports = main_data.get("imports") or []
    _import_modules(imports, main_path)

    runtime = _build_dataclass(
        RuntimeConfig, main_data.get("runtime", {}), main_path, section="runtime"
    )
    camera = _build_camera_config(main_data.get("camera", {}), main_path)
    trigger = _build_trigger_config(main_data.get("trigger", {}), main_path)
    comm = _build_comm_config(main_data.get("comm", {}), main_path)
    detect = _build_dataclass(
        DetectConfigBlock, main_data.get("detect", {}), main_path, section="detect"
    )
    output = _build_output_config(main_data.get("output", {}), main_path)

    if not detect.config_file:
        raise ConfigError(f"detect.config_file is required in {main_path}")
    detect_path = detect.config_file
    if not os.path.isabs(detect_path):
        detect_path = os.path.join(config_dir, detect_path)
    if not os.path.exists(detect_path):
        raise ConfigError(f"Detect config not found: {detect_path}")

    detect_params = _read_yaml(detect_path)
    if not isinstance(detect_params, dict):
        raise ConfigError(f"Detect config must be a mapping: {detect_path}")
    return LoadedConfig(
        imports=imports,
        runtime=runtime,
        camera=camera,
        trigger=trigger,
        comm=comm,
        detect=detect,
        output=output,
        detect_params=detect_params or {},
        paths={
            "main": main_path,
            "detect": detect_path,
        },
    )


def _find_main_config(config_dir: str) -> str:
    patterns = [
        os.path.join(config_dir, "main_*.yaml"),
        os.path.join(config_dir, "main_*.yml"),
    ]
    candidates: list[str] = []
    for pattern in patterns:
        candidates.extend(glob.glob(pattern))
    if len(candidates) == 0:
        raise ConfigError(f"No main_*.yaml found under {config_dir}")
    if len(candidates) > 1:
        raise ConfigError(
            f"Expected exactly one main_*.yaml, found: {', '.join(candidates)}"
        )
    return candidates[0]


def _read_yaml(path: str) -> dict[str, Any]:
    with open(path, "r", encoding="utf-8") as f:
        data = yaml.safe_load(f) or {}
    if not isinstance(data, dict):
        raise ConfigError(f"YAML root must be a mapping: {path}")
    return data


def _build_dataclass(cls, data: dict[str, Any], main_path: str, section: str):
    obj = cls()
    fields = cls.__dataclass_fields__
    for k, v in (data or {}).items():
        if k in fields:
            setattr(obj, k, v)
        else:
            raise ConfigError(f"Unknown field {section}.{k} in {main_path}")
    return obj


def _validate_allowed_keys(
    data: dict[str, Any], allowed_keys: set[str], section: str, main_path: str
) -> None:
    for key in data.keys():
        if key not in allowed_keys:
            raise ConfigError(f"Unknown field {section}.{key} in {main_path}")


def _apply_scalar_fields(cfg: Any, data: dict[str, Any], keys: tuple[str, ...]) -> None:
    for key in keys:
        if key in data:
            setattr(cfg, key, data[key])


def _apply_subblock_dataclasses(
    cfg: Any,
    data: dict[str, Any],
    main_path: str,
    *,
    parent_section: str,
    block_classes: dict[str, Any],
) -> None:
    for key, cls in block_classes.items():
        block = data.get(key)
        if block is None:
            continue
        if not isinstance(block, dict):
            raise ConfigError(
                f"'{parent_section}.{key}' must be a mapping in {main_path}"
            )
        setattr(
            cfg,
            key,
            _build_dataclass(cls, block, main_path, section=f"{parent_section}.{key}"),
        )


def _build_trigger_config(data: dict[str, Any], main_path: str) -> TriggerConfigBlock:
    if data is None:
        return TriggerConfigBlock()
    if not isinstance(data, dict):
        raise ConfigError(f"'trigger' must be a mapping in {main_path}")
    cfg = TriggerConfigBlock()
    allowed_keys = {
        "debounce_ms",
        "global_min_interval_ms",
        "high_priority_cooldown_ms",
        "high_priority_sources",
        "low_priority_sources",
        "ip_whitelist",
        "tcp",
        "modbus",
    }
    _validate_allowed_keys(data, allowed_keys, "trigger", main_path)
    _apply_scalar_fields(
        cfg,
        data,
        (
            "debounce_ms",
            "global_min_interval_ms",
            "high_priority_cooldown_ms",
            "high_priority_sources",
            "low_priority_sources",
            "ip_whitelist",
        ),
    )
    _apply_subblock_dataclasses(
        cfg,
        data,
        main_path,
        parent_section="trigger",
        block_classes={
            "tcp": TriggerTcpConfigBlock,
            "modbus": TriggerModbusConfigBlock,
        },
    )
    return cfg


def _build_camera_config(data: dict[str, Any], main_path: str) -> CameraConfigBlock:
    if data is None:
        return CameraConfigBlock()
    if not isinstance(data, dict):
        raise ConfigError(f"'camera' must be a mapping in {main_path}")

    cfg = CameraConfigBlock()
    if "type" in data:
        cfg.type = str(data.get("type") or cfg.type)
    selected_type = str(cfg.type or "").strip()
    camera_fields = CameraConfigBlock.__dataclass_fields__

    def _apply_camera_fields(block: dict[str, Any], section: str):
        for k, v in block.items():
            if k in camera_fields:
                setattr(cfg, k, v)
            else:
                raise ConfigError(f"Unknown field {section}.{k} in {main_path}")

    for key, value in data.items():
        if key in {"type", "common"}:
            continue
        if isinstance(value, dict):
            continue
        raise ConfigError(
            f"camera.{key} must be nested under camera.common or camera.{selected_type} in {main_path}"
        )

    common_data = data.get("common")
    if common_data is not None:
        if not isinstance(common_data, dict):
            raise ConfigError(f"'camera.common' must be a mapping in {main_path}")
        _apply_camera_fields(common_data, "camera.common")

    selected_block = data.get(selected_type)
    if selected_block is not None:
        if not isinstance(selected_block, dict):
            raise ConfigError(
                f"'camera.{selected_type}' must be a mapping in {main_path}"
            )
        _apply_camera_fields(selected_block, f"camera.{selected_type}")
    return cfg


def _build_comm_config(data: dict[str, Any], main_path: str) -> CommConfigBlock:
    if data is None:
        return CommConfigBlock()
    if not isinstance(data, dict):
        raise ConfigError(f"'comm' must be a mapping in {main_path}")
    cfg = CommConfigBlock()
    _validate_allowed_keys(data, {"tcp", "modbus", "http"}, "comm", main_path)
    _apply_subblock_dataclasses(
        cfg,
        data,
        main_path,
        parent_section="comm",
        block_classes={
            "tcp": CommTcpConfigBlock,
            "modbus": CommModbusConfigBlock,
            "http": CommHttpConfigBlock,
        },
    )
    return cfg


def _build_output_config(data: dict[str, Any], main_path: str) -> OutputConfigBlock:
    if data is None:
        return OutputConfigBlock()
    if not isinstance(data, dict):
        raise ConfigError(f"'output' must be a mapping in {main_path}")
    cfg = OutputConfigBlock()
    _validate_allowed_keys(data, {"hmi", "modbus", "write_csv"}, "output", main_path)
    _apply_subblock_dataclasses(
        cfg,
        data,
        main_path,
        parent_section="output",
        block_classes={
            "hmi": OutputHmiConfigBlock,
            "modbus": OutputModbusConfigBlock,
        },
    )
    if "write_csv" in data:
        cfg.write_csv = bool(data.get("write_csv"))
    return cfg


def _import_modules(imports: Any, main_path: str):
    if imports is None:
        return
    if not isinstance(imports, list):
        raise ConfigError(f"'imports' must be a list in {main_path}")
    if not imports:
        return
    for path in imports:
        if not isinstance(path, str) or not path:
            raise ConfigError(f"Invalid import path {path!r} in {main_path}")
        importlib.import_module(path)


__all__ = ["load_config"]
