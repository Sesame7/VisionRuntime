import logging
from typing import Tuple

import numpy as np

from .base import register_detector

L = logging.getLogger("sci_cam.detection.overexposure")


def detect_overexposure(
    img: np.ndarray,
    threshold: int = 245,
    ratio_threshold: float = 0.02,
    return_overlay: bool = True,
) -> Tuple[float, bool, np.ndarray | None]:
    """
    Detect overexposed regions.
    Returns (overexp_ratio, is_ng, overlay_image or None).
    Overlay draws half-transparent red on overexposed pixels; can be skipped for low-spec devices.
    """
    # Use integer luminance approximation to avoid temporary float arrays and lower peak memory.
    if img.ndim == 2:
        gray = img.astype(np.uint8, copy=False)
    else:
        img_u16 = img.astype(np.uint16, copy=False)
        gray_u16 = (
            img_u16[:, :, 0] * 77 + img_u16[:, :, 1] * 150 + img_u16[:, :, 2] * 29
        ) >> 8
        gray = gray_u16.astype(np.uint8, copy=False)
    mask = gray >= threshold
    ratio = float(mask.mean())
    is_ng = ratio > ratio_threshold

    if not return_overlay:
        return ratio, is_ng, None

    if img.ndim == 2:
        overlay_base = np.repeat(gray[:, :, None], 3, axis=2)
    else:
        overlay_base = img
    if not mask.any():
        return ratio, is_ng, overlay_base

    overlay = overlay_base.copy()
    over_idx = mask
    # Blend with red (50%) using integer math to avoid temporary float arrays.
    overlay[..., 2][over_idx] = (
        overlay[..., 2][over_idx].astype(np.uint16) + 255
    ) // 2  # R channel in BGR -> index 2
    overlay[..., 1][over_idx] = overlay[..., 1][over_idx] // 2  # G
    overlay[..., 0][over_idx] = overlay[..., 0][over_idx] // 2  # B
    return ratio, is_ng, overlay


@register_detector("overexposure")
class OverExposureDetector:
    def __init__(
        self,
        params: dict,
        generate_overlay: bool = True,
        input_pixel_format: str | None = None,
    ):
        self.threshold = int(params.get("overexp_threshold", 245))
        self.ratio_threshold = float(params.get("overexp_ratio", 0.02))
        self.downscale_factor = float(params.get("downscale_factor", 1.0))
        self.generate_overlay = generate_overlay
        self._validate()

    def detect(self, img: np.ndarray):
        target = img
        if self.downscale_factor < 0.999:
            step = max(1, int(round(1.0 / self.downscale_factor)))
            target = img[::step, ::step]
        ratio, is_ng, overlay = detect_overexposure(
            target,
            threshold=self.threshold,
            ratio_threshold=self.ratio_threshold,
            return_overlay=self.generate_overlay,
        )
        prefix = "NG" if is_ng else "OK"
        message = f"{prefix}: overexp_ratio={ratio:.4f} thr={self.threshold} ratio_thr={self.ratio_threshold:.4f}"
        result_code = "DETECT_OVEREXPOSE" if is_ng else "OK"
        return (not is_ng), message, overlay, result_code

    def _validate(self):
        if not (0 <= self.threshold <= 255):
            raise ValueError("detect overexp_threshold must be 0..255")
        if not (0 < self.ratio_threshold <= 1.0):
            raise ValueError("detect overexp_ratio must be in (0, 1]")
        if not (0 < self.downscale_factor <= 1.0):
            raise ValueError("detect downscale_factor must be in (0, 1]")


__all__ = ["OverExposureDetector", "detect_overexposure"]
