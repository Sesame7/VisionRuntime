"""YAML configuration loader (single main_*.yaml + detect_*.yaml)."""

import glob
import importlib
import os
from dataclasses import dataclass, field
from typing import Any, Dict, List

import yaml


class ConfigError(Exception):
    pass


@dataclass
class RuntimeConfig:
    save_dir: str = "data"
    max_runtime_s: float = 0.0
    detect_queue_capacity: int = 50
    log_level: str = "info"


@dataclass
class CameraConfigBlock:
    type: str = "opt"
    device_index: int = 0
    grab_timeout_ms: int = 2000
    max_retry_per_frame: int = 3
    save_images: bool = True
    save_ext: str = ".bmp"
    capture_output_format: str = "bgr8"
    width: int = 0
    height: int = 0
    ae_enable: bool = True
    awb_enable: bool = True
    exposure_us: int = 0
    analogue_gain: float = 0.0
    frame_duration_us: int = 0
    settle_ms: int = 200
    use_still: bool = True
    image_dir: str = ""
    order: str = "name_asc"
    end_mode: str = "loop"


@dataclass
class TriggerTcpConfigBlock:
    enabled: bool = True
    word: str = "TRIG"


@dataclass
class TriggerModbusConfigBlock:
    enabled: bool = False
    poll_ms: int = 20


@dataclass
class TriggerConfigBlock:
    debounce_ms: float = 10.0
    global_min_interval_ms: float = 0.0
    high_priority_cooldown_ms: float = 0.0
    high_priority_sources: List[str] = field(default_factory=list)
    low_priority_sources: List[str] = field(default_factory=list)
    ip_whitelist: List[str] = field(default_factory=list)
    tcp: TriggerTcpConfigBlock = field(default_factory=TriggerTcpConfigBlock)
    modbus: TriggerModbusConfigBlock = field(default_factory=TriggerModbusConfigBlock)


@dataclass
class CommTcpConfigBlock:
    host: str = "0.0.0.0"
    port: int = 9000


@dataclass
class CommModbusConfigBlock:
    host: str = "0.0.0.0"
    port: int = 5020
    coil_offset: int = 800
    di_offset: int = 800
    ir_offset: int = 50
    heartbeat_ms: int = 1000


@dataclass
class CommHttpConfigBlock:
    host: str = "0.0.0.0"
    port: int = 8000


@dataclass
class CommConfigBlock:
    tcp: CommTcpConfigBlock = field(default_factory=CommTcpConfigBlock)
    modbus: CommModbusConfigBlock = field(default_factory=CommModbusConfigBlock)
    http: CommHttpConfigBlock = field(default_factory=CommHttpConfigBlock)


@dataclass
class DetectConfigBlock:
    impl: str = "overexposure"
    config_file: str = ""
    timeout_ms: int = 2000
    preview_enabled: bool = True


@dataclass
class OutputHmiConfigBlock:
    enabled: bool = True
    history_size: int = 10


@dataclass
class OutputModbusConfigBlock:
    enabled: bool = False


@dataclass
class OutputConfigBlock:
    hmi: OutputHmiConfigBlock = field(default_factory=OutputHmiConfigBlock)
    modbus: OutputModbusConfigBlock = field(default_factory=OutputModbusConfigBlock)
    write_csv: bool = True


@dataclass
class LoadedConfig:
    imports: List[str]
    runtime: RuntimeConfig
    camera: CameraConfigBlock
    trigger: TriggerConfigBlock
    comm: CommConfigBlock
    detect: DetectConfigBlock
    output: OutputConfigBlock
    detect_params: Dict[str, Any]
    paths: Dict[str, str] = field(default_factory=dict)


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


def _find_main_config(config_dir: str) -> str:
    patterns = [
        os.path.join(config_dir, "main_*.yaml"),
        os.path.join(config_dir, "main_*.yml"),
    ]
    candidates: List[str] = []
    for pattern in patterns:
        candidates.extend(glob.glob(pattern))
    if len(candidates) == 0:
        raise ConfigError(f"No main_*.yaml found under {config_dir}")
    if len(candidates) > 1:
        raise ConfigError(
            f"Expected exactly one main_*.yaml, found: {', '.join(candidates)}"
        )
    return candidates[0]


def _read_yaml(path: str) -> Dict[str, Any]:
    with open(path, "r", encoding="utf-8") as f:
        data = yaml.safe_load(f) or {}
    if not isinstance(data, dict):
        raise ConfigError(f"YAML root must be a mapping: {path}")
    return data


def _build_dataclass(cls, data: Dict[str, Any], main_path: str, section: str):
    obj = cls()
    fields = cls.__dataclass_fields__
    for k, v in (data or {}).items():
        if k in fields:
            setattr(obj, k, v)
        else:
            raise ConfigError(f"Unknown field {section}.{k} in {main_path}")
    return obj


def _validate_allowed_keys(
    data: Dict[str, Any], allowed_keys: set[str], section: str, main_path: str
) -> None:
    for key in data.keys():
        if key not in allowed_keys:
            raise ConfigError(f"Unknown field {section}.{key} in {main_path}")


def _apply_scalar_fields(cfg: Any, data: Dict[str, Any], keys: tuple[str, ...]) -> None:
    for key in keys:
        if key in data:
            setattr(cfg, key, data[key])


def _apply_subblock_dataclasses(
    cfg: Any,
    data: Dict[str, Any],
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


def _build_trigger_config(data: Dict[str, Any], main_path: str) -> TriggerConfigBlock:
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


def _build_camera_config(data: Dict[str, Any], main_path: str) -> CameraConfigBlock:
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
        if key == "type":
            continue
        if key == "common":
            continue
        if isinstance(value, dict):
            continue  # camera.<impl> blocks; only selected_type is applied below
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


def _build_comm_config(data: Dict[str, Any], main_path: str) -> CommConfigBlock:
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


def _build_output_config(data: Dict[str, Any], main_path: str) -> OutputConfigBlock:
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


__all__ = [
    "ConfigError",
    "RuntimeConfig",
    "CameraConfigBlock",
    "TriggerConfigBlock",
    "TriggerTcpConfigBlock",
    "TriggerModbusConfigBlock",
    "CommConfigBlock",
    "CommHttpConfigBlock",
    "CommTcpConfigBlock",
    "CommModbusConfigBlock",
    "DetectConfigBlock",
    "OutputHmiConfigBlock",
    "OutputModbusConfigBlock",
    "OutputConfigBlock",
    "LoadedConfig",
    "load_config",
    "validate_config",
]
