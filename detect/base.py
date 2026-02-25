import logging
from typing import Callable, Dict, Protocol, Tuple

import cv2
import numpy as np
from core.registry import register_named, resolve_registered

L = logging.getLogger("vision_runtime.detection")


class Detector(Protocol):
    def detect(
        self, img: np.ndarray
    ) -> Tuple[bool, str, np.ndarray | None, str | None]:
        """Return ok flag, message, overlay image (or None), and result_code."""
        ...


_registry: Dict[str, Callable[..., Detector]] = {}


def register_detector(name: str):
    return register_named(_registry, name)


def create_detector(
    name: str,
    params: dict,
    *,
    generate_overlay: bool = True,
    input_pixel_format: str | None = None,
) -> Detector:
    # Lazy import: allow configs to reference detectors without requiring explicit
    # import registration elsewhere (and avoid importing optional heavy deps unless needed).
    factory = resolve_registered(
        _registry,
        name,
        package=__package__ or "detect",
        unknown_label="detector impl",
    )
    return factory(
        params or {}, generate_overlay, input_pixel_format=input_pixel_format
    )


def create_detector_from_loaded_config(
    cfg,
    *,
    input_pixel_format: str | None = None,
) -> Detector:
    preview_enabled = bool(cfg.detect.preview_enabled)
    pixel_format = (
        input_pixel_format
        if input_pixel_format is not None
        else str(cfg.camera.capture_output_format)
    )
    return create_detector(
        cfg.detect.impl,
        cfg.detect_params or {},
        generate_overlay=preview_enabled,
        input_pixel_format=pixel_format,
    )


def encode_image_jpeg(
    img: np.ndarray, quality: int = 50, subsampling: int = 2
) -> Tuple[bytes, str]:
    """
    Encode image to JPEG bytes with speed-friendly params.
    Returns (bytes, content_type).
    """
    bgr = img.astype(np.uint8, copy=False)
    params = [int(cv2.IMWRITE_JPEG_QUALITY), int(quality)]
    # Optional subsampling control if supported by OpenCV build.
    if hasattr(cv2, "IMWRITE_JPEG_SAMPLING_FACTOR"):
        factor_map = {
            0: "IMWRITE_JPEG_SAMPLING_FACTOR_444",
            1: "IMWRITE_JPEG_SAMPLING_FACTOR_422",
            2: "IMWRITE_JPEG_SAMPLING_FACTOR_420",
        }
        factor_name = factor_map.get(int(subsampling))
        factor = getattr(cv2, factor_name, None) if factor_name else None
        if factor is not None:
            params += [int(cv2.IMWRITE_JPEG_SAMPLING_FACTOR), int(factor)]
    ok, buf = cv2.imencode(".jpg", bgr, params)
    if not ok:
        raise RuntimeError("opencv_imencode_failed")
    return buf.tobytes(), "image/jpeg"


__all__ = [
    "Detector",
    "register_detector",
    "create_detector",
    "create_detector_from_loaded_config",
    "encode_image_jpeg",
]
