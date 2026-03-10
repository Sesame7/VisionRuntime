"""Config package facade."""

from .loader import load_config
from .schema import ConfigError, LoadedConfig, LoadedPaths
from .validate import validate_config

__all__ = [
    "ConfigError",
    "LoadedConfig",
    "LoadedPaths",
    "load_config",
    "validate_config",
]
