# -- coding: utf-8 --

import logging
import os
import random
import re
import time
from contextlib import contextmanager
from datetime import datetime, timezone

import numpy as np
import cv2

from camera.base import BaseCamera, CameraConfig, CaptureResult, register_camera

L = logging.getLogger("vision_runtime.camera.mock")

_SUPPORTED_EXTS = {".jpg", ".jpeg", ".png", ".bmp"}
_ORDER_CHOICES = {
    "name_asc",
    "name_desc",
    "name_natural",
    "mtime_asc",
    "mtime_desc",
    "random",
}
_END_CHOICES = {"loop", "stop", "hold"}


def _natural_key(name: str):
    parts = re.split(r"(\d+)", name)
    key = []
    for part in parts:
        if part.isdigit():
            key.append(int(part))
        else:
            key.append(part.lower())
    return key


def _resolve_image_dir(path: str) -> str:
    base = str(path or "").strip()
    if not base:
        raise RuntimeError("mock image_dir is required")
    if not os.path.isabs(base):
        base = os.path.abspath(os.path.join(os.getcwd(), base))
    if not os.path.isdir(base):
        raise RuntimeError(f"mock image_dir not found: {base}")
    return base


def _list_images(root_dir: str) -> list[str]:
    files = []
    for name in os.listdir(root_dir):
        full = os.path.join(root_dir, name)
        if not os.path.isfile(full):
            continue
        ext = os.path.splitext(name)[1].lower()
        if ext in _SUPPORTED_EXTS:
            files.append(full)
    return files


def _sort_images(paths: list[str], order: str) -> list[str]:
    if order == "name_asc":
        return sorted(paths, key=lambda p: os.path.basename(p).lower())
    if order == "name_desc":
        return sorted(paths, key=lambda p: os.path.basename(p).lower(), reverse=True)
    if order == "name_natural":
        return sorted(paths, key=lambda p: _natural_key(os.path.basename(p)))
    if order == "mtime_asc":
        return sorted(paths, key=lambda p: os.path.getmtime(p))
    if order == "mtime_desc":
        return sorted(paths, key=lambda p: os.path.getmtime(p), reverse=True)
    if order == "random":
        shuffled = list(paths)
        random.shuffle(shuffled)
        return shuffled
    return paths


def _imread_any(path: str, flags: int) -> np.ndarray | None:
    arr = cv2.imread(path, flags)
    if arr is not None:
        return arr
    try:
        data = np.fromfile(path, dtype=np.uint8)
    except Exception:
        return None
    if data.size == 0:
        return None
    return cv2.imdecode(data, flags)


def _load_image_bgr(path: str, output_pixel_format: str) -> np.ndarray:
    fmt = str(output_pixel_format).lower()
    if fmt == "mono8":
        arr = _imread_any(path, cv2.IMREAD_GRAYSCALE)
    else:
        arr = _imread_any(path, cv2.IMREAD_COLOR)
    if arr is None:
        raise RuntimeError("opencv_imread_failed")
    if fmt == "mono8":
        return arr.astype(np.uint8, copy=False)
    return arr.astype(np.uint8, copy=False)


@register_camera("mock")
class MockCamera(BaseCamera):
    def __init__(self, cfg: CameraConfig):
        super().__init__(cfg)
        self._paths: list[str] = []
        self._pos = 0
        self._root_dir = ""
        self._order = str(cfg.order or "name_asc").strip().lower()
        self._end_mode = str(cfg.end_mode or "loop").strip().lower()

    def _scan(self):
        if self._order not in _ORDER_CHOICES:
            raise RuntimeError(
                f"mock order must be one of {sorted(_ORDER_CHOICES)}, got {self._order!r}"
            )
        if self._end_mode not in _END_CHOICES:
            raise RuntimeError(
                f"mock end_mode must be one of {sorted(_END_CHOICES)}, got {self._end_mode!r}"
            )
        self._paths = _sort_images(_list_images(self._root_dir), self._order)
        if not self._paths:
            raise RuntimeError(f"no images found in {self._root_dir}")
        self._pos = 0

    def _next_path(self) -> str | None:
        if not self._paths:
            return None
        if self._pos < len(self._paths):
            path = self._paths[self._pos]
            self._pos += 1
            return path
        if self._end_mode == "loop":
            if self._order == "random":
                self._paths = _sort_images(self._paths, self._order)
            self._pos = 0
            path = self._paths[self._pos]
            self._pos += 1
            return path
        if self._end_mode == "hold":
            return self._paths[-1]
        return None

    def capture_once(self, idx, triggered_at=None):
        path = self._next_path()
        if not path:
            return CaptureResult(
                success=False,
                trigger_seq=idx,
                device_id="mock",
                error="no_more_images",
            )

        start = time.perf_counter()
        try:
            arr = _load_image_bgr(path, self.cfg.output_pixel_format)
        except Exception as e:
            return CaptureResult(
                success=False,
                trigger_seq=idx,
                device_id="mock",
                error=f"read_failed: {e}",
            )
        read_ms = (time.perf_counter() - start) * 1000
        captured_at = datetime.now(timezone.utc)
        try:
            rel_path = os.path.relpath(path)
        except Exception:
            rel_path = path
        L.info("[%5s] mock @ %s", idx, rel_path)
        return CaptureResult(
            success=True,
            trigger_seq=idx,
            source="",
            device_id="mock",
            path=None,
            image=arr,
            timings={"grab_ms": read_ms},
            captured_at=captured_at,
        )

    @contextmanager
    def session(self):
        self._root_dir = _resolve_image_dir(self.cfg.image_dir)
        self._scan()
        yield self


__all__ = ["MockCamera"]
