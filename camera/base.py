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


def build_camera_config(
    cfg_block,
    *,
    save_dir: str,
    ext: str | None = None,
    output_pixel_format: str | None = None,
) -> CameraConfig:
    return CameraConfig(
        save_dir=save_dir,
        ext=str(ext if ext is not None else getattr(cfg_block, "ext", ".bmp")),
        device_index=int(getattr(cfg_block, "device_index", 0)),
        timeout_ms=int(getattr(cfg_block, "grab_timeout_ms", 2000)),
        max_retry_per_frame=int(getattr(cfg_block, "max_retry_per_frame", 3)),
        save_images=bool(getattr(cfg_block, "save_images", True)),
        output_pixel_format=str(
            output_pixel_format
            if output_pixel_format is not None
            else getattr(cfg_block, "output_pixel_format", "bgr8")
        ),
        width=int(getattr(cfg_block, "width", 0)),
        height=int(getattr(cfg_block, "height", 0)),
        ae_enable=bool(getattr(cfg_block, "ae_enable", True)),
        awb_enable=bool(getattr(cfg_block, "awb_enable", True)),
        exposure_us=int(getattr(cfg_block, "exposure_us", 0)),
        analogue_gain=float(getattr(cfg_block, "analogue_gain", 0.0)),
        frame_duration_us=int(getattr(cfg_block, "frame_duration_us", 0)),
        settle_ms=int(getattr(cfg_block, "settle_ms", 200)),
        use_still=bool(getattr(cfg_block, "use_still", True)),
        image_dir=str(getattr(cfg_block, "image_dir", "")),
        order=str(getattr(cfg_block, "order", "name_asc")),
        end_mode=str(getattr(cfg_block, "end_mode", "loop")),
    )


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
    "build_camera_config",
    "BaseCamera",
    "register_camera",
    "create_camera",
    "ensure_dir",
]
