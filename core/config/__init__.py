"""Config package facade (compatible with historical `core.config` imports)."""

from . import schema as _schema
from .loader import load_config
from .schema import ConfigError, LoadedConfig
from .validate import validate_config

_LEGACY_SCHEMA_EXPORTS = {
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
}


def __getattr__(name: str):
    # Backward compatibility for callers importing schema blocks from core.config.
    if name in _LEGACY_SCHEMA_EXPORTS:
        return getattr(_schema, name)
    raise AttributeError(f"module '{__name__}' has no attribute '{name}'")


def __dir__():
    return sorted(set(globals()) | _LEGACY_SCHEMA_EXPORTS)


__all__ = [
    "ConfigError",
    "LoadedConfig",
    "load_config",
    "validate_config",
]
