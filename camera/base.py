# -- coding: utf-8 --

import os
import threading
import importlib
from abc import ABC, abstractmethod
from contextlib import contextmanager
from dataclasses import dataclass
from datetime import datetime
from typing import Dict, Type

from core.contracts import CaptureResult

CameraFactory = Dict[str, Type["BaseCamera"]]
_registry: CameraFactory = {}


@dataclass
class CameraConfig:
    save_dir: str
    ext: str = ".bmp"
    device_index: int = 0
    timeout_ms: int = 2000
    max_retry_per_frame: int = 3
    save_images: bool = True
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


class BaseCamera(ABC):
    def __init__(self, cfg: CameraConfig):
        self.cfg = cfg
        self.lock = threading.Lock()

    @abstractmethod
    def capture_once(self, idx, triggered_at: datetime | None = None) -> CaptureResult:
        """Perform a single capture, returning CaptureResult."""

    @contextmanager
    @abstractmethod
    def session(self):
        """Manage camera lifecycle."""
        yield


def register_camera(name: str):
    def decorator(cls: Type[BaseCamera]):
        _registry[name] = cls
        return cls

    return decorator


def create_camera(name: str, cfg: CameraConfig) -> BaseCamera:
    if name not in _registry:
        import_err: Exception | None = None
        try:
            importlib.import_module(f"{__package__}.{name}")
        except Exception as e:
            import_err = e
    if name not in _registry:
        hint = (
            f" (import failed: {import_err})"
            if "import_err" in locals() and import_err
            else ""
        )
        raise ValueError(
            f"Unknown camera type '{name}'. Available: {', '.join(_registry.keys()) or 'none'}{hint}"
        )
    return _registry[name](cfg)


def ensure_dir(path: str):
    os.makedirs(path, exist_ok=True)


__all__ = [
    "CameraConfig",
    "CaptureResult",
    "BaseCamera",
    "register_camera",
    "create_camera",
    "ensure_dir",
]
