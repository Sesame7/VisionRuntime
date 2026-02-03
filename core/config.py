"""YAML configuration loader (single main_*.yaml + detect_*.yaml)."""


import glob
import importlib
import logging
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
	history_size: int = 10
	max_pending_triggers: int = 50
	debounce_ms: float = 10.0
	log_level: str = "info"
	opencv_num_threads: int = 0


@dataclass
class CameraConfigBlock:
	type: str = "opt"
	device_index: int = 0
	grab_timeout_ms: int = 2000
	max_retry_per_frame: int = 3
	save_images: bool = True
	ext: str = ".bmp"
	output_pixel_format: str = "bgr8"
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
	enable_preview: bool = True
	generate_overlay: bool = True


@dataclass
class OutputConfigBlock:
	enable_http: bool = True
	enable_modbus: bool = False
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

	runtime = _build_dataclass(RuntimeConfig, main_data.get("runtime", {}), main_path, section="runtime")
	camera = _build_dataclass(CameraConfigBlock, main_data.get("camera", {}), main_path, section="camera")
	trigger = _build_trigger_config(main_data.get("trigger", {}), main_path)
	comm = _build_comm_config(main_data.get("comm", {}), main_path)
	detect = _build_dataclass(DetectConfigBlock, main_data.get("detect", {}), main_path, section="detect")
	output = _build_dataclass(OutputConfigBlock, main_data.get("output", {}), main_path, section="output")

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
	candidates: List[str] = []
	for pattern in patterns:
		candidates.extend(glob.glob(pattern))
	if len(candidates) == 0:
		raise ConfigError(f"No main_*.yaml found under {config_dir}")
	if len(candidates) > 1:
		raise ConfigError(f"Expected exactly one main_*.yaml, found: {', '.join(candidates)}")
	return candidates[0]


def _read_yaml(path: str) -> Dict[str, Any]:
	with open(path, "r", encoding="utf-8") as f:
		data = yaml.safe_load(f) or {}
	if not isinstance(data, dict):
		raise ConfigError(f"YAML root must be a mapping: {path}")
	return data


def _build_dataclass(cls, data: Dict[str, Any], main_path: str, section: str):
	obj = cls()
	for k, v in (data or {}).items():
		if hasattr(obj, k):
			setattr(obj, k, v)
		else:
			logging.warning("Unknown field %s.%s in %s (ignored)", section, k, main_path)
	return obj


def _build_trigger_config(data: Dict[str, Any], main_path: str) -> TriggerConfigBlock:
	if data is None:
		return TriggerConfigBlock()
	if not isinstance(data, dict):
		raise ConfigError(f"'trigger' must be a mapping in {main_path}")
	cfg = TriggerConfigBlock()
	allowed_keys = {
		"global_min_interval_ms",
		"high_priority_cooldown_ms",
		"high_priority_sources",
		"low_priority_sources",
		"ip_whitelist",
		"tcp",
		"modbus",
	}
	for key in data.keys():
		if key not in allowed_keys:
			logging.warning("Unknown field trigger.%s in %s (ignored)", key, main_path)
	# Global filters
	for key in (
		"global_min_interval_ms",
		"high_priority_cooldown_ms",
		"high_priority_sources",
		"low_priority_sources",
		"ip_whitelist",
	):
		if key in data:
			setattr(cfg, key, data.get(key))

	# New structured blocks
	tcp_data = data.get("tcp")
	if isinstance(tcp_data, dict):
		cfg.tcp = _build_dataclass(TriggerTcpConfigBlock, tcp_data, main_path, section="trigger.tcp")
	modbus_data = data.get("modbus")
	if isinstance(modbus_data, dict):
		cfg.modbus = _build_dataclass(TriggerModbusConfigBlock, modbus_data, main_path, section="trigger.modbus")

	return cfg


def _build_comm_config(data: Dict[str, Any], main_path: str) -> CommConfigBlock:
	if data is None:
		return CommConfigBlock()
	if not isinstance(data, dict):
		raise ConfigError(f"'comm' must be a mapping in {main_path}")
	cfg = CommConfigBlock()
	allowed_keys = {"tcp", "modbus", "http"}
	for key in data.keys():
		if key not in allowed_keys:
			logging.warning("Unknown field comm.%s in %s (ignored)", key, main_path)
	tcp_data = data.get("tcp")
	if isinstance(tcp_data, dict):
		cfg.tcp = _build_dataclass(CommTcpConfigBlock, tcp_data, main_path, section="comm.tcp")
	modbus_data = data.get("modbus")
	if isinstance(modbus_data, dict):
		cfg.modbus = _build_dataclass(CommModbusConfigBlock, modbus_data, main_path, section="comm.modbus")
	http_data = data.get("http")
	if isinstance(http_data, dict):
		cfg.http = _build_dataclass(CommHttpConfigBlock, http_data, main_path, section="comm.http")
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
		try:
			importlib.import_module(path)
		except Exception as e:
			raise ConfigError(f"Failed to import {path!r} declared in {main_path}: {e}") from e


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
	"OutputConfigBlock",
	"LoadedConfig",
	"load_config",
]
