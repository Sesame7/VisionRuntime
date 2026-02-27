"""Typed config schema blocks shared by loader/validator/runtime."""

from dataclasses import dataclass, field
from typing import Any, Dict, List


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
    preview_max_edge: int = 1280


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
]
