from typing import Tuple

import numpy as np

from .base import register_detector


def detect_overexposure(
    img: np.ndarray,
    threshold: int = 245,
    ratio_threshold: float = 0.02,
    return_overlay: bool = True,
) -> Tuple[float, bool, np.ndarray | None]:
    if img.ndim == 2:
        gray = img.astype(np.uint8, copy=False)
    else:
        img_u16 = img.astype(np.uint16, copy=False)
        gray = (
            img_u16[:, :, 0] * 77 + img_u16[:, :, 1] * 150 + img_u16[:, :, 2] * 29
        ) >> 8
        gray = gray.astype(np.uint8, copy=False)
    mask = gray >= threshold
    ratio = float(mask.mean())
    is_ng = ratio > ratio_threshold

    if not return_overlay:
        return ratio, is_ng, None

    if img.ndim == 2:
        overlay = np.repeat(gray[:, :, None], 3, axis=2)
    else:
        overlay = img.copy()
    if mask.any():
        overlay[..., 2][mask] = (overlay[..., 2][mask].astype(np.uint16) + 255) // 2
        overlay[..., 1][mask] = overlay[..., 1][mask] // 2
        overlay[..., 0][mask] = overlay[..., 0][mask] // 2
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


__all__ = ["OverExposureDetector", "detect_overexposure"]
