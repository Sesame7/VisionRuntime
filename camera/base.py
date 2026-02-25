# -- coding: utf-8 --

import os
import threading
from abc import ABC, abstractmethod
from contextlib import contextmanager
from dataclasses import dataclass
from datetime import datetime
from typing import Dict, Type

from core.contracts import CaptureResult
from core.registry import register_named, resolve_registered

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
        ext=_normalize_ext(ext if ext is not None else cfg_block.save_ext),
        device_index=int(cfg_block.device_index),
        timeout_ms=int(cfg_block.grab_timeout_ms),
        max_retry_per_frame=int(cfg_block.max_retry_per_frame),
        save_images=bool(cfg_block.save_images),
        output_pixel_format=_normalize_pixel_format(
            output_pixel_format
            if output_pixel_format is not None
            else cfg_block.capture_output_format
        ),
        width=int(cfg_block.width),
        height=int(cfg_block.height),
        ae_enable=bool(cfg_block.ae_enable),
        awb_enable=bool(cfg_block.awb_enable),
        exposure_us=int(cfg_block.exposure_us),
        analogue_gain=float(cfg_block.analogue_gain),
        frame_duration_us=int(cfg_block.frame_duration_us),
        settle_ms=int(cfg_block.settle_ms),
        use_still=bool(cfg_block.use_still),
        image_dir=str(cfg_block.image_dir),
        order=str(cfg_block.order),
        end_mode=str(cfg_block.end_mode),
    )


def _normalize_ext(ext: object) -> str:
    raw = str(ext or "")
    if not raw:
        return ".bmp"
    return raw if raw.startswith(".") else f".{raw}"


def _normalize_pixel_format(value: object) -> str:
    fmt = str(value or "").strip().lower()
    return fmt or "bgr8"


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
    return register_named(_registry, name)


def create_camera(name: str, cfg: CameraConfig) -> BaseCamera:
    cls = resolve_registered(
        _registry,
        name,
        package=__package__ or "camera",
        unknown_label="camera type",
    )
    return cls(cfg)


def create_camera_from_loaded_config(cfg) -> BaseCamera:
    image_root = os.path.join(cfg.runtime.save_dir, "images")
    if bool(cfg.camera.save_images):
        ensure_dir(image_root)
    cam_cfg = build_camera_config(cfg.camera, save_dir=image_root)
    return create_camera(cfg.camera.type, cam_cfg)


def ensure_dir(path: str):
    os.makedirs(path, exist_ok=True)


__all__ = [
    "CameraConfig",
    "CaptureResult",
    "build_camera_config",
    "BaseCamera",
    "register_camera",
    "create_camera",
    "create_camera_from_loaded_config",
    "ensure_dir",
]
