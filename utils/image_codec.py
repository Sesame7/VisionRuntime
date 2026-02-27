from __future__ import annotations

import cv2
import numpy as np


def resize_image_max_edge(img: np.ndarray, max_edge: int) -> np.ndarray:
    """Downscale image to keep the longest edge <= max_edge (no upscaling)."""
    limit = int(max_edge or 0)
    if limit <= 0:
        return img
    h, w = img.shape[:2]
    if h <= 0 or w <= 0:
        return img
    longest = max(h, w)
    if longest <= limit:
        return img
    scale = float(limit) / float(longest)
    new_w = max(1, int(round(w * scale)))
    new_h = max(1, int(round(h * scale)))
    if new_w == w and new_h == h:
        return img
    return cv2.resize(img, (new_w, new_h), interpolation=cv2.INTER_AREA)


def encode_image_jpeg(
    img: np.ndarray, quality: int = 50, subsampling: int = 2
) -> tuple[bytes, str]:
    """Encode an image to JPEG bytes and MIME type."""
    bgr = img.astype(np.uint8, copy=False)
    params = [int(cv2.IMWRITE_JPEG_QUALITY), int(quality)]
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


__all__ = ["encode_image_jpeg", "resize_image_max_edge"]
