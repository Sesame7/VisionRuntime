import inspect
import logging
from typing import Callable, Dict, Protocol, Tuple

import numpy as np
from utils.registry import register_named, resolve_registered

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
    preview_max_edge: int = 1280,
) -> Detector:
    # Lazy import: allow configs to reference detectors without requiring explicit
    # import registration elsewhere (and avoid importing optional heavy deps unless needed).
    factory = resolve_registered(
        _registry,
        name,
        package=__package__ or "detect",
        unknown_label="detector impl",
    )
    kwargs = {"input_pixel_format": input_pixel_format}
    if _factory_accepts_kwarg(factory, "preview_max_edge"):
        kwargs["preview_max_edge"] = int(preview_max_edge)
    return factory(params or {}, generate_overlay, **kwargs)


def create_detector_from_loaded_config(
    cfg,
    *,
    input_pixel_format: str | None = None,
) -> Detector:
    preview_enabled = bool(cfg.detect.preview_enabled)
    preview_max_edge = int(cfg.detect.preview_max_edge)
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
        preview_max_edge=preview_max_edge,
    )


def _factory_accepts_kwarg(factory: Callable[..., Detector], name: str) -> bool:
    try:
        signature = inspect.signature(factory)
    except (TypeError, ValueError):
        return False
    for param in signature.parameters.values():
        if param.kind == inspect.Parameter.VAR_KEYWORD:
            return True
    return name in signature.parameters


__all__ = [
    "Detector",
    "register_detector",
    "create_detector",
    "create_detector_from_loaded_config",
]
