import math
from dataclasses import dataclass, field
from typing import Any, List, Optional, Tuple

import cv2
import numpy as np

from .base import register_detector


# Numeric and geometry helpers
def _round_and_clamp_int(v: Any, lo: int, hi: int) -> int:
    x = int(round(float(v)))
    return lo if x < lo else hi if x > hi else x


def _clamp_u8(v: Any) -> int:
    return max(0, min(255, int(v)))


def _clamp_span_inclusive(lo: Any, hi: Any, vmin: int, vmax: int) -> Tuple[int, int]:
    lo_i = _round_and_clamp_int(lo, vmin, vmax)
    hi_i = _round_and_clamp_int(hi, vmin, vmax)
    if hi_i <= lo_i:
        hi_i = min(vmax, lo_i + 1)
    return lo_i, hi_i


def _clamp_span_exclusive(
    lo: Any, hi: Any, vmin: int, vmax_excl: int
) -> Tuple[int, int]:
    lo_i = _round_and_clamp_int(lo, vmin, max(vmin, vmax_excl - 1))
    hi_i = _round_and_clamp_int(hi, vmin, vmax_excl)
    if hi_i <= lo_i:
        hi_i = min(vmax_excl, lo_i + 1)
    return lo_i, hi_i


def _rect_from_inclusive(
    x0: int, y0: int, x1: int, y1: int
) -> Tuple[int, int, int, int]:
    x0 = int(x0)
    y0 = int(y0)
    x1 = int(x1)
    y1 = int(y1)
    return (
        x0,
        y0,
        max(1, x1 - x0 + 1),
        max(1, y1 - y0 + 1),
    )


# Array and signal helpers
def _hue_in_range(h: np.ndarray, lo: int, hi: int) -> np.ndarray:
    lo = int(lo) % 180
    hi = int(hi) % 180
    if lo <= hi:
        return (h >= lo) & (h <= hi)
    return (h >= lo) | (h <= hi)


def _find_longest_true_span(b: np.ndarray) -> Optional[Tuple[int, int]]:
    arr = np.asarray(b, dtype=bool)
    if arr.size == 0:
        return None
    best: Optional[Tuple[int, int]] = None
    start: Optional[int] = None
    for i, flag in enumerate(arr):
        if flag and start is None:
            start = i
        elif not flag and start is not None:
            end = i - 1
            if best is None or (end - start) > (best[1] - best[0]):
                best = (start, end)
            start = None
    if start is not None:
        end = int(arr.size - 1)
        if best is None or (end - start) > (best[1] - best[0]):
            best = (start, end)
    return best


def _row_foreground_ratio(mask_u8: np.ndarray, n_rows: int) -> np.ndarray:
    if mask_u8.size == 0:
        return np.zeros(n_rows, dtype=np.float32)
    return (mask_u8.mean(axis=1) / 255.0).astype(np.float32)


def _smooth_1d(values: np.ndarray, window: int, axis: int) -> np.ndarray:
    arr = np.asarray(values, dtype=np.float32)
    window = int(window)
    if window <= 1 or arr.size <= 1:
        return arr
    if axis == 0:
        return cv2.blur(
            arr.reshape(-1, 1),
            (1, window),
            borderType=cv2.BORDER_REPLICATE,
        ).reshape(-1)
    if axis == 1:
        return cv2.blur(
            arr.reshape(1, -1),
            (window, 1),
            borderType=cv2.BORDER_REPLICATE,
        ).reshape(-1)
    raise ValueError("axis must be 0 or 1")


# Config parsing helpers
def _parse_xy_point(value: Any, field_name: str) -> Tuple[int, int]:
    if not isinstance(value, dict):
        raise ValueError(
            f"{field_name} entries must be objects like {{x: ..., y: ...}}"
        )
    if "x" not in value or "y" not in value:
        raise ValueError(f"{field_name} entries must contain x and y")
    return int(value["x"]), int(value["y"])


def _normalize_odd_window_size(value: int, field_name: str) -> int:
    window_size = int(value)
    if window_size < 0:
        raise ValueError(f"{field_name} must be >= 0")
    if (window_size % 2) == 0:
        window_size += 1
    return window_size


def _as_bgr_triplet(value: Any, default: Tuple[int, int, int]) -> Tuple[int, int, int]:
    if isinstance(value, (list, tuple)) and len(value) == 3:
        b, g, r = value
        return (int(b), int(g), int(r))
    return default


@dataclass
class _SlotRoi:
    row: int
    col: int
    x: int
    y: int
    w: int
    h: int

    @property
    def rect(self) -> Tuple[int, int, int, int]:
        return (self.x, self.y, self.w, self.h)


@dataclass
class _SlotInspectionState:
    row: int
    col: int
    roi_rect: Tuple[int, int, int, int]  # x,y,w,h
    x_stem_center: float  # full-image x (float)
    y_stem_bottom_anchor: int  # full-image y
    roi_hsv: np.ndarray  # ROI HSV image
    y_junction_line: Optional[int] = None


@dataclass
class _FailTracker:
    fail_stage_priority: dict[str, int]
    first_fail_entry: Optional[Tuple[str, int, int]] = None
    first_fail_sort_key: Optional[Tuple[int, int, int]] = None

    def consider(self, code: str, row: int, col: int) -> None:
        sort_key = (
            int(row),
            int(col),
            int(self.fail_stage_priority.get(code, 99)),
        )
        if self.first_fail_sort_key is None or sort_key < self.first_fail_sort_key:
            self.first_fail_sort_key = sort_key
            self.first_fail_entry = (str(code), int(row), int(col))


@dataclass
class _OverlayRecorder:
    detector: Any
    enabled: bool
    ops: List[Tuple[str, tuple]] = field(default_factory=list)

    def _push(self, kind: str, *args: Any) -> None:
        if self.enabled:
            self.ops.append((kind, args))

    def rect(
        self,
        rect: Tuple[int, int, int, int],
        color: Tuple[int, int, int],
        thickness: int | None = None,
    ) -> None:
        self._push("rect", rect, color, thickness)

    def center(self, cx: int, cy: int, color: Tuple[int, int, int]) -> None:
        self._push("center", int(cx), int(cy), color)

    def cross(self, cx: int, cy: int, color: Tuple[int, int, int]) -> None:
        self._push("cross", int(cx), int(cy), color)

    def junction_line(
        self, x: int, w: int, y: int, color: Tuple[int, int, int]
    ) -> None:
        self._push("junction_line", int(x), int(w), int(y), color)

    def mask_overlay(
        self,
        x: int,
        y: int,
        mask_u8: np.ndarray,
        color: Tuple[int, int, int],
        alpha: float = 0.35,
    ) -> None:
        self._push("mask_overlay", int(x), int(y), mask_u8, color, float(alpha))

    def render(self, img: Optional[np.ndarray]) -> None:
        if not self.enabled or img is None or not self.ops:
            return
        draw_ops = {
            "rect": self.detector._draw_rect,
            "center": self.detector._draw_center,
            "cross": self.detector._draw_cross,
            "junction_line": self.detector._draw_junction_line,
            "mask_overlay": self.detector._draw_mask_overlay,
        }
        for kind, args in self.ops:
            drawer = draw_ops.get(kind)
            if drawer is not None:
                drawer(img, *args)


@dataclass
class _SlotLayoutCfg:
    rows: int
    slots_per_row: List[int]
    left_anchors: List[Tuple[int, int]]
    right_anchors: List[Tuple[int, int]]
    roi_width_px: int
    roi_height_px: int


@dataclass
class _StemStageCfg:
    hsv_lower: np.ndarray
    hsv_upper: np.ndarray
    min_area_px: int
    bottom_exclude_percentile: float


@dataclass
class _JunctionLineWhiteRowCfg:
    s_max: int
    v_min: int
    lower: np.ndarray
    upper: np.ndarray


@dataclass
class _JunctionLineStageCfg:
    y_from_anchor_up_px: int
    y_from_anchor_down_px: int
    outer_radius_px: int
    inner_exclude_radius_px: int
    white_row_cfgs: List[_JunctionLineWhiteRowCfg]
    split_smooth_window_rows: int
    split_min_rows_above: int
    split_min_rows_below: int
    height_check_max_deviation_px: int


@dataclass
class _ColorBandStageCfg:
    y_below_junction_line_from_px: int
    y_below_junction_line_to_px: int
    x_radius_from_stem_center_px: int
    hue_low: int
    hue_high: int
    saturation_threshold: int
    saturation_smooth_window_cols: int
    width_check_min_px: int
    width_check_max_px: int
    alignment_reference_x_shift_px: float
    alignment_tolerance_px: float


@dataclass
class _ResultOverlayCfg:
    ok_bgr: Tuple[int, int, int]
    ng_bgr: Tuple[int, int, int]
    stroke_width_px: int


@dataclass
class _DebugOverlayCfg:
    slot_layout: bool
    stem_mask: bool
    junction_line: bool
    color_band: bool
    guide_bgr: Tuple[int, int, int]
    mask_overlay_bgr: Tuple[int, int, int]


@register_detector("weigao_tray")
class WeigaoTrayDetector:
    """
    Weigao tray inspection (configurable rows/columns), adapted for Smart_Camera.

    - Overlay uses input image as the base.
    - Cumulative drawing: green marks keep even if later stages fail.
    """

    _FAIL_STAGE_PRIORITY = {
        "STEM_NOT_FOUND_NG": 0,
        "JUNCTION_LINE_NOT_FOUND_NG": 1,
        "JUNCTION_LINE_HEIGHT_NG": 2,
        "COLOR_BAND_NOT_FOUND_NG": 3,
        "COLOR_BAND_ALIGNMENT_NG": 4,
    }

    # Public API and pipeline
    def __init__(
        self,
        params: dict,
        generate_overlay: bool = True,
        input_pixel_format: str | None = None,
    ):
        self.params = params or {}
        self.generate_overlay = bool(generate_overlay)
        if input_pixel_format and input_pixel_format.lower() != "bgr8":
            raise ValueError("weigao_tray requires camera.capture_output_format=bgr8")

        self._load_params()

    def detect(self, img: np.ndarray):
        return self._detect_impl(img)

    def _detect_impl(self, img: np.ndarray):
        if img is None or img.size == 0:
            return False, "Error: empty image", None, "ERROR"
        if img.ndim != 3 or img.shape[2] != 3:
            return (
                False,
                f"Error: expected BGR image HxWx3, got shape={img.shape}",
                None,
                "ERROR",
            )

        img_h, img_w = img.shape[:2]
        slot_rois = self._generate_slot_rois(img_w, img_h)

        overlay_img = img.copy() if self.generate_overlay else None
        overlay = _OverlayRecorder(detector=self, enabled=overlay_img is not None)
        fail_tracker = _FailTracker(fail_stage_priority=self._FAIL_STAGE_PRIORITY)
        slots_stem_ok = self._locate_stem_stage(img, slot_rois, overlay, fail_tracker)
        slots_junction_line_ok = self._process_junction_line_stage(
            slots_stem_ok, overlay, fail_tracker
        )
        self._process_color_band_stage(slots_junction_line_ok, overlay, fail_tracker)
        overlay.render(overlay_img)
        return self._finalize_result(overlay_img, fail_tracker)

    def _finalize_result(
        self, overlay_img: Optional[np.ndarray], fail_tracker: _FailTracker
    ):
        if fail_tracker.first_fail_entry is None:
            return True, "OK", overlay_img, "OK"
        code, rr, cc = fail_tracker.first_fail_entry
        return False, f"NG: {code} r={rr} c={cc}", overlay_img, code

    # Config parsing and validation
    def _load_params(self) -> None:
        p = self.params
        self.slot_layout_cfg = self._parse_slot_layout_cfg(p)
        self.stem_cfg = self._parse_stem_cfg(p)

        junction_line_params = p.get("junction_line") or {}
        self.junction_line_cfg = self._parse_junction_line_cfg(junction_line_params)

        self.color_band_cfg = self._parse_color_band_cfg(p)
        self.result_overlay_cfg = self._parse_result_overlay_cfg(p)
        self.debug_overlay_cfg = self._parse_debug_overlay_cfg(p)

        self._validate()

    def _parse_slot_layout_cfg(self, p: dict) -> _SlotLayoutCfg:
        slot_layout = p.get("slot_layout") or {}
        slot_roi = slot_layout.get("slot_roi") or {}
        left_anchors_raw = list(slot_layout.get("left_anchors") or [])
        right_anchors_raw = list(slot_layout.get("right_anchors") or [])
        slots_per_row = list(slot_layout.get("slots_per_row") or [17, 17, 17])
        rows = len(slots_per_row)
        return _SlotLayoutCfg(
            rows=rows,
            slots_per_row=slots_per_row,
            left_anchors=[
                _parse_xy_point(v, "slot_layout.left_anchors") for v in left_anchors_raw
            ],
            right_anchors=[
                _parse_xy_point(v, "slot_layout.right_anchors")
                for v in right_anchors_raw
            ],
            roi_width_px=int(slot_roi.get("width_px", 100)),
            roi_height_px=int(slot_roi.get("height_px", 200)),
        )

    def _parse_stem_cfg(self, p: dict) -> _StemStageCfg:
        stem = p.get("stem") or {}
        stem_hsv = stem.get("hsv") or {}
        return _StemStageCfg(
            hsv_lower=np.array(stem_hsv.get("lower") or [10, 140, 30], dtype=np.uint8),
            hsv_upper=np.array(stem_hsv.get("upper") or [50, 255, 255], dtype=np.uint8),
            min_area_px=int(stem.get("min_area_px", 50)),
            bottom_exclude_percentile=float(
                stem.get("bottom_exclude_percentile", 90.0)
            ),
        )

    def _parse_junction_line_cfg(self, junction_line: dict) -> _JunctionLineStageCfg:
        search_band = junction_line.get("search_band") or {}
        y_from_anchor_px = search_band.get("y_from_anchor_px") or {}
        x_radius_from_stem_center_px = (
            search_band.get("x_radius_from_stem_center_px") or {}
        )
        split = junction_line.get("split") or {}
        min_split_rows = split.get("min_split_rows") or {}
        smooth_window_rows = _normalize_odd_window_size(
            int(split.get("smooth_window_rows", 9)),
            "junction_line.split.smooth_window_rows",
        )
        junction_line_height_check = junction_line.get("height_check") or {}

        white_by_row = junction_line.get("white_by_row")
        if white_by_row is None:
            white_rows_raw: List[Any] = [{} for _ in range(self.slot_layout_cfg.rows)]
        elif isinstance(white_by_row, list):
            white_rows_raw = list(white_by_row)
        else:
            raise ValueError("junction_line.white_by_row must be a list of row configs")

        white_rows: List[_JunctionLineWhiteRowCfg] = []
        for i, item in enumerate(white_rows_raw):
            if item is None:
                item = {}
            if not isinstance(item, dict):
                raise ValueError(f"junction_line.white_by_row[{i}] must be an object")
            s_max = int(item.get("s_max", 50))
            v_min = int(item.get("v_min", 50))
            white_rows.append(
                _JunctionLineWhiteRowCfg(
                    s_max=s_max,
                    v_min=v_min,
                    lower=np.array([0, 0, _clamp_u8(v_min)], dtype=np.uint8),
                    upper=np.array([179, _clamp_u8(s_max), 255], dtype=np.uint8),
                )
            )
        return _JunctionLineStageCfg(
            y_from_anchor_up_px=int(y_from_anchor_px.get("up", 20)),
            y_from_anchor_down_px=int(y_from_anchor_px.get("down", 40)),
            outer_radius_px=int(x_radius_from_stem_center_px.get("outer", 34)),
            inner_exclude_radius_px=int(
                x_radius_from_stem_center_px.get("inner_exclude", 12)
            ),
            white_row_cfgs=white_rows,
            split_smooth_window_rows=int(smooth_window_rows),
            split_min_rows_above=int(min_split_rows.get("above", 5)),
            split_min_rows_below=int(min_split_rows.get("below", 5)),
            height_check_max_deviation_px=int(
                junction_line_height_check.get("max_deviation_px", 15)
            ),
        )

    def _parse_color_band_cfg(self, p: dict) -> _ColorBandStageCfg:
        color_band = p.get("color_band") or {}
        search_window = color_band.get("search_window") or {}
        y_below_junction_line_px = search_window.get("y_below_junction_line_px") or {}
        color_response = color_band.get("color_response") or {}
        hue_range = color_response.get("hue_range") or {}
        saturation = color_response.get("saturation") or {}
        width_check_px = color_band.get("width_check_px") or {}
        alignment_check = color_band.get("alignment_check") or {}

        saturation_smooth_window_cols = _normalize_odd_window_size(
            int(saturation.get("smooth_window_cols", 5)),
            "color_band.color_response.saturation.smooth_window_cols",
        )
        return _ColorBandStageCfg(
            y_below_junction_line_from_px=int(y_below_junction_line_px.get("from", 10)),
            y_below_junction_line_to_px=int(y_below_junction_line_px.get("to", 30)),
            x_radius_from_stem_center_px=int(
                search_window.get("x_radius_from_stem_center_px", 25)
            ),
            hue_low=int(hue_range.get("low", 0)),
            hue_high=int(hue_range.get("high", 179)),
            saturation_threshold=int(saturation.get("threshold", 30)),
            saturation_smooth_window_cols=int(saturation_smooth_window_cols),
            width_check_min_px=int(width_check_px.get("min", 10)),
            width_check_max_px=int(width_check_px.get("max", 0)),
            alignment_reference_x_shift_px=float(
                alignment_check.get("reference_x_shift_px", 8)
            ),
            alignment_tolerance_px=float(alignment_check.get("tolerance_px", 5)),
        )

    def _parse_result_overlay_cfg(self, p: dict) -> _ResultOverlayCfg:
        result_overlay = p.get("result_overlay") or {}
        return _ResultOverlayCfg(
            ok_bgr=_as_bgr_triplet(result_overlay.get("ok_bgr"), (0, 200, 0)),
            ng_bgr=_as_bgr_triplet(result_overlay.get("ng_bgr"), (0, 0, 220)),
            stroke_width_px=max(1, int(result_overlay.get("stroke_width_px", 4))),
        )

    def _parse_debug_overlay_cfg(self, p: dict) -> _DebugOverlayCfg:
        debug_overlay = p.get("debug_overlay") or {}
        return _DebugOverlayCfg(
            slot_layout=bool(debug_overlay.get("slot_layout", False)),
            stem_mask=bool(debug_overlay.get("stem_mask", False)),
            junction_line=bool(debug_overlay.get("junction_line", False)),
            color_band=bool(debug_overlay.get("color_band", False)),
            guide_bgr=(220, 0, 0),  # blue in BGR
            mask_overlay_bgr=(140, 60, 0),  # darker orange in BGR
        )

    def _validate(self) -> None:
        slot_layout_cfg = self.slot_layout_cfg
        if slot_layout_cfg.rows <= 0:
            raise ValueError("slot_layout.slots_per_row must not be empty")
        if len(slot_layout_cfg.slots_per_row) != slot_layout_cfg.rows:
            raise ValueError("internal slot_layout row count mismatch")
        if any(int(c) <= 0 for c in slot_layout_cfg.slots_per_row):
            raise ValueError("slot_layout.slots_per_row values must be > 0")
        if (
            len(slot_layout_cfg.left_anchors) != slot_layout_cfg.rows
            or len(slot_layout_cfg.right_anchors) != slot_layout_cfg.rows
        ):
            raise ValueError(
                "slot_layout.left_anchors/right_anchors length must equal "
                "len(slot_layout.slots_per_row)"
            )
        if slot_layout_cfg.roi_width_px <= 0 or slot_layout_cfg.roi_height_px <= 0:
            raise ValueError("slot_layout.slot_roi.width_px and height_px must be > 0")
        if self.stem_cfg.hsv_lower.shape != (3,) or self.stem_cfg.hsv_upper.shape != (
            3,
        ):
            raise ValueError("stem.hsv.lower/upper must be 3-element lists")
        if not (0.0 <= self.stem_cfg.bottom_exclude_percentile <= 100.0):
            raise ValueError("stem.bottom_exclude_percentile must be in [0,100]")
        if len(self.junction_line_cfg.white_row_cfgs) != slot_layout_cfg.rows:
            raise ValueError(
                "junction_line.white_by_row length must equal "
                "len(slot_layout.slots_per_row)"
            )
        if self.junction_line_cfg.height_check_max_deviation_px < 0:
            raise ValueError("junction_line.height_check.max_deviation_px must be >= 0")
        if self.junction_line_cfg.outer_radius_px <= 0:
            raise ValueError(
                "junction_line.search_band.x_radius_from_stem_center_px.outer "
                "must be > 0"
            )
        if self.junction_line_cfg.inner_exclude_radius_px < 0:
            raise ValueError(
                "junction_line.search_band.x_radius_from_stem_center_px."
                "inner_exclude must be >= 0"
            )
        if (
            self.junction_line_cfg.outer_radius_px
            <= self.junction_line_cfg.inner_exclude_radius_px
        ):
            raise ValueError(
                "junction_line.search_band.x_radius_from_stem_center_px.outer "
                "must be > inner_exclude"
            )
        if (
            self.color_band_cfg.width_check_min_px < 0
            or self.color_band_cfg.width_check_max_px < 0
        ):
            raise ValueError("color_band.width_check_px.min/max must be >= 0")
        if self.color_band_cfg.x_radius_from_stem_center_px <= 0:
            raise ValueError(
                "color_band.search_window.x_radius_from_stem_center_px must be > 0"
            )

    # Shared geometry and overlay helpers
    def _keep_largest_component_inplace(self, mask_u8: np.ndarray) -> None:
        num, labels, stats, _ = cv2.connectedComponentsWithStats(
            mask_u8, connectivity=8
        )
        if num <= 1:
            mask_u8.fill(0)
            return
        if num == 2:
            return
        # Skip label 0 (background)
        areas = stats[1:, cv2.CC_STAT_AREA]
        if areas.size == 0:
            mask_u8.fill(0)
            return
        largest_label = int(1 + int(np.argmax(areas)))
        mask_u8.fill(0)
        mask_u8[labels == largest_label] = 255

    def _generate_slot_rois(self, img_w: int, img_h: int) -> List[_SlotRoi]:
        slot_layout_cfg = self.slot_layout_cfg
        slot_rois: List[_SlotRoi] = []
        roi_w = max(1, min(int(slot_layout_cfg.roi_width_px), int(img_w)))
        roi_h = max(1, min(int(slot_layout_cfg.roi_height_px), int(img_h)))
        max_x = max(0, img_w - roi_w)
        max_y = max(0, img_h - roi_h)
        for row_idx in range(slot_layout_cfg.rows):
            cols = int(slot_layout_cfg.slots_per_row[row_idx])
            x_left, y_top = slot_layout_cfg.left_anchors[row_idx]
            x_right, _y_top2 = slot_layout_cfg.right_anchors[row_idx]
            x_left = _round_and_clamp_int(x_left, 0, max_x)
            x_right = _round_and_clamp_int(x_right, 0, max_x)
            y_top = _round_and_clamp_int(y_top, 0, max_y)
            if x_right < x_left:
                x_right = x_left
            x_step = 0.0 if cols <= 1 else (x_right - x_left) / float(cols - 1)
            for col_idx in range(cols):
                x = _round_and_clamp_int(x_left + col_idx * x_step, 0, max_x)
                slot_rois.append(
                    _SlotRoi(
                        row=row_idx,
                        col=col_idx,
                        x=x,
                        y=y_top,
                        w=roi_w,
                        h=roi_h,
                    )
                )
        return slot_rois

    def _junction_line_search_strip_spans(
        self, cx_in_roi: float, roi_w: int
    ) -> Tuple[int, int, int, int]:
        junction_line_cfg = self.junction_line_cfg
        cx_i = _round_and_clamp_int(cx_in_roi, 0, roi_w - 1)
        outer_radius = int(junction_line_cfg.outer_radius_px)
        inner_exclude_radius = int(junction_line_cfg.inner_exclude_radius_px)
        lx0, lx1 = _clamp_span_exclusive(
            cx_i - outer_radius, cx_i - inner_exclude_radius, 0, roi_w
        )
        rx0, rx1 = _clamp_span_exclusive(
            cx_i + inner_exclude_radius, cx_i + outer_radius, 0, roi_w
        )
        return lx0, lx1, rx0, rx1

    def _draw_rect(
        self,
        img: np.ndarray,
        rect: Tuple[int, int, int, int],
        color: Tuple[int, int, int],
        thickness: int | None = None,
    ) -> None:
        x, y, w, h = rect
        cv2.rectangle(
            img,
            (int(x), int(y)),
            (int(x + w - 1), int(y + h - 1)),
            color,
            thickness=max(
                1,
                int(
                    self.result_overlay_cfg.stroke_width_px
                    if thickness is None
                    else thickness
                ),
            ),
        )

    def _draw_center(
        self, img: np.ndarray, cx: int, cy: int, color: Tuple[int, int, int]
    ) -> None:
        r = max(1, int(self.result_overlay_cfg.stroke_width_px))
        cv2.circle(img, (int(cx), int(cy)), r, color, thickness=-1)

    def _draw_cross(
        self, img: np.ndarray, cx: int, cy: int, color: Tuple[int, int, int]
    ) -> None:
        arm = max(2, int(self.result_overlay_cfg.stroke_width_px) * 2)
        t = max(1, int(self.result_overlay_cfg.stroke_width_px))
        cv2.line(
            img,
            (int(cx - arm), int(cy - arm)),
            (int(cx + arm), int(cy + arm)),
            color,
            thickness=t,
        )
        cv2.line(
            img,
            (int(cx - arm), int(cy + arm)),
            (int(cx + arm), int(cy - arm)),
            color,
            thickness=t,
        )

    def _draw_junction_line(
        self, img: np.ndarray, x: int, w: int, y: int, color: Tuple[int, int, int]
    ) -> None:
        cv2.line(
            img,
            (int(x), int(y)),
            (int(x + w), int(y)),
            color,
            thickness=self.result_overlay_cfg.stroke_width_px,
        )

    def _draw_mask_overlay(
        self,
        img: np.ndarray,
        x: int,
        y: int,
        mask_u8: np.ndarray,
        color: Tuple[int, int, int],
        alpha: float = 0.35,
    ) -> None:
        if mask_u8 is None or mask_u8.size == 0:
            return
        h, w = mask_u8.shape[:2]
        roi = img[y : y + h, x : x + w]
        if roi.shape[:2] != (h, w):
            return
        m = mask_u8 > 0
        if not bool(np.any(m)):
            return
        color_img = np.empty_like(roi)
        color_img[:] = color
        blended = cv2.addWeighted(roi, 1.0 - float(alpha), color_img, float(alpha), 0.0)
        roi[m] = blended[m]

    def _record_slot_failure(
        self,
        overlay: _OverlayRecorder,
        fail_tracker: _FailTracker,
        code: str,
        row: int,
        col: int,
        rect: Optional[Tuple[int, int, int, int]] = None,
        rects: Optional[List[Tuple[int, int, int, int]]] = None,
    ) -> None:
        if rect is not None:
            overlay.rect(rect, self.result_overlay_cfg.ng_bgr)
        if rects:
            for r in rects:
                overlay.rect(r, self.result_overlay_cfg.ng_bgr)
        fail_tracker.consider(code, row, col)

    # Detection stages: stem
    def _locate_stem_stage(
        self,
        img: np.ndarray,
        slot_rois: List[_SlotRoi],
        overlay: _OverlayRecorder,
        fail_tracker: _FailTracker,
    ) -> List[_SlotInspectionState]:
        stem_cfg = self.stem_cfg
        debug_overlay_cfg = self.debug_overlay_cfg
        overlay_cfg = self.result_overlay_cfg
        stem_min_area_px = int(stem_cfg.min_area_px)
        bottom_exclude_percentile = (
            max(0.0, min(100.0, float(stem_cfg.bottom_exclude_percentile))) / 100.0
        )
        slots_stem_ok: List[_SlotInspectionState] = []

        for slot_roi in slot_rois:
            r = int(slot_roi.row)
            c = int(slot_roi.col)
            x = int(slot_roi.x)
            y = int(slot_roi.y)
            w = int(slot_roi.w)
            h = int(slot_roi.h)
            roi_bgr = img[y : y + h, x : x + w]
            roi_hsv = cv2.cvtColor(roi_bgr, cv2.COLOR_BGR2HSV)
            stem_mask_raw = cv2.inRange(roi_hsv, stem_cfg.hsv_lower, stem_cfg.hsv_upper)
            self._keep_largest_component_inplace(stem_mask_raw)

            if debug_overlay_cfg.stem_mask:
                overlay.mask_overlay(
                    x,
                    y,
                    stem_mask_raw,
                    debug_overlay_cfg.mask_overlay_bgr,
                    alpha=0.55,
                )

            stem_mask_raw_area = int(cv2.countNonZero(stem_mask_raw))
            if stem_mask_raw_area < stem_min_area_px:
                self._record_slot_failure(
                    overlay, fail_tracker, "STEM_NOT_FOUND_NG", r, c, rect=(x, y, w, h)
                )
                continue

            raw_mask_row_counts = (stem_mask_raw > 0).sum(axis=1).astype(np.int32)
            raw_mask_foreground_total = int(raw_mask_row_counts.sum())
            raw_mask_row_cumsum = np.cumsum(raw_mask_row_counts, dtype=np.int64)
            bottom_exclude_rank = (
                int(
                    math.floor(
                        bottom_exclude_percentile * (raw_mask_foreground_total - 1)
                    )
                )
                if raw_mask_foreground_total > 1
                else 0
            )
            y_stem_exclude_cut = int(
                np.searchsorted(
                    raw_mask_row_cumsum, bottom_exclude_rank + 1, side="left"
                )
            )

            stem_mask_trimmed = stem_mask_raw.copy()
            if y_stem_exclude_cut + 1 < h:
                stem_mask_trimmed[y_stem_exclude_cut + 1 :, :] = 0

            trimmed_moments = cv2.moments(stem_mask_trimmed, binaryImage=True)
            stem_mask_trimmed_area = float(trimmed_moments.get("m00", 0.0))
            if stem_mask_trimmed_area <= 0.0:
                self._record_slot_failure(
                    overlay, fail_tracker, "STEM_NOT_FOUND_NG", r, c, rect=(x, y, w, h)
                )
                continue

            stem_cx_roi = float(trimmed_moments["m10"] / stem_mask_trimmed_area)
            stem_cy_roi = float(trimmed_moments["m01"] / stem_mask_trimmed_area)
            stem_cx_img = int(round(x + stem_cx_roi))
            stem_cy_img = int(round(y + stem_cy_roi))
            overlay.center(stem_cx_img, stem_cy_img, overlay_cfg.ok_bgr)

            trimmed_mask_row_counts = (
                (stem_mask_trimmed > 0).sum(axis=1).astype(np.int32)
            )
            trimmed_nonzero_rows = np.flatnonzero(trimmed_mask_row_counts > 0)
            if trimmed_nonzero_rows.size <= 0:
                self._record_slot_failure(
                    overlay, fail_tracker, "STEM_NOT_FOUND_NG", r, c, rect=(x, y, w, h)
                )
                continue
            y_stem_bottom_anchor = int(trimmed_nonzero_rows[-1])

            slots_stem_ok.append(
                _SlotInspectionState(
                    row=r,
                    col=c,
                    roi_rect=(x, y, w, h),
                    x_stem_center=float(x + stem_cx_roi),
                    y_stem_bottom_anchor=int(y + y_stem_bottom_anchor),
                    roi_hsv=roi_hsv,
                )
            )
        return slots_stem_ok

    # Detection stages: junction line
    def _locate_junction_line_candidates(
        self,
        slots_stem_ok: List[_SlotInspectionState],
        overlay: _OverlayRecorder,
        fail_tracker: _FailTracker,
    ) -> Tuple[List[_SlotInspectionState], List[List[int]]]:
        junction_line_cfg = self.junction_line_cfg
        debug_overlay_cfg = self.debug_overlay_cfg
        slots_junction_line_located: List[_SlotInspectionState] = []
        junction_line_y_by_row: List[List[int]] = [
            [] for _ in range(self.slot_layout_cfg.rows)
        ]

        for slot in slots_stem_ok:
            x, y, w, h = slot.roi_rect
            junction_line_white_row_cfg = junction_line_cfg.white_row_cfgs[slot.row]
            search_y0, search_y1 = _clamp_span_inclusive(
                slot.y_stem_bottom_anchor - junction_line_cfg.y_from_anchor_up_px,
                slot.y_stem_bottom_anchor + junction_line_cfg.y_from_anchor_down_px,
                y,
                y + h - 1,
            )
            lx0, lx1, rx0, rx1 = self._junction_line_search_strip_spans(
                slot.x_stem_center - x, w
            )
            search_strip_rects = [
                _rect_from_inclusive(x + lx0, search_y0, x + lx1 - 1, search_y1),
                _rect_from_inclusive(x + rx0, search_y0, x + rx1 - 1, search_y1),
            ]

            if debug_overlay_cfg.junction_line:
                for rect in search_strip_rects:
                    overlay.rect(rect, debug_overlay_cfg.guide_bgr, thickness=1)

            search_y0_roi, search_y1_roi = _clamp_span_inclusive(
                search_y0 - y, search_y1 - y, 0, h - 1
            )
            junction_window_hsv = slot.roi_hsv[search_y0_roi : search_y1_roi + 1, :]
            junction_white_mask = cv2.inRange(
                junction_window_hsv,
                junction_line_white_row_cfg.lower,
                junction_line_white_row_cfg.upper,
            )
            # Build a single virtual search region by concatenating the left/right
            # strips. All later row-wise operations work on this merged region.
            merged_search_strips_white_mask = np.concatenate(
                (junction_white_mask[:, lx0:lx1], junction_white_mask[:, rx0:rx1]),
                axis=1,
            )
            white_row_response = _row_foreground_ratio(
                merged_search_strips_white_mask,
                int(merged_search_strips_white_mask.shape[0]),
            )

            white_row_response_smoothed = _smooth_1d(
                white_row_response,
                junction_line_cfg.split_smooth_window_rows,
                axis=0,
            )

            window_rows = int(white_row_response.size)
            split_row_min = max(1, int(junction_line_cfg.split_min_rows_above))
            split_row_max = window_rows - max(
                1, int(junction_line_cfg.split_min_rows_below)
            )
            if window_rows < 2 or split_row_min > split_row_max:
                self._record_slot_failure(
                    overlay,
                    fail_tracker,
                    "JUNCTION_LINE_NOT_FOUND_NG",
                    slot.row,
                    slot.col,
                    rects=search_strip_rects,
                )
                continue

            # Statistical split line: choose the row that maximizes
            # (mean white response below) - (mean white response above).
            # This matches the process assumption that the region below the
            # junction is whiter than the region above it.
            white_row_response_prefix_sums = np.concatenate(
                ([0.0], np.cumsum(white_row_response_smoothed, dtype=np.float32))
            )
            candidate_split_rows = np.arange(
                split_row_min, split_row_max + 1, dtype=np.int32
            )
            white_sum_above = white_row_response_prefix_sums[candidate_split_rows]
            white_sum_below = (
                white_row_response_prefix_sums[window_rows] - white_sum_above
            )
            white_mean_above = white_sum_above / candidate_split_rows
            white_mean_below = white_sum_below / (window_rows - candidate_split_rows)
            split_contrast = white_mean_below - white_mean_above
            best_split_idx = int(np.argmax(split_contrast))
            best_split_row_in_window = int(candidate_split_rows[best_split_idx])

            slot.y_junction_line = int(search_y0 + best_split_row_in_window)
            slots_junction_line_located.append(slot)
            junction_line_y_by_row[slot.row].append(int(slot.y_junction_line))

        return slots_junction_line_located, junction_line_y_by_row

    def _validate_junction_line_height_stage(
        self,
        slots_junction_line_located: List[_SlotInspectionState],
        junction_line_y_by_row: List[List[int]],
        overlay: _OverlayRecorder,
        fail_tracker: _FailTracker,
    ) -> List[_SlotInspectionState]:
        junction_line_cfg = self.junction_line_cfg
        overlay_cfg = self.result_overlay_cfg
        junction_line_height_baseline_by_row: List[Optional[float]] = [
            float(np.median(ys)) if ys else None for ys in junction_line_y_by_row
        ]

        slots_junction_line_ok: List[_SlotInspectionState] = []
        for slot in slots_junction_line_located:
            junction_line_height_baseline = junction_line_height_baseline_by_row[
                slot.row
            ]
            x, y, w, h = slot.roi_rect
            if junction_line_height_baseline is None:
                self._record_slot_failure(
                    overlay,
                    fail_tracker,
                    "JUNCTION_LINE_NOT_FOUND_NG",
                    slot.row,
                    slot.col,
                    rect=(x, y, w, h),
                )
                continue

            lx0, lx1, rx0, rx1 = self._junction_line_search_strip_spans(
                slot.x_stem_center - x, w
            )
            # Draw only on left/right strips to avoid covering the center feature.
            line_segments = [
                (x + lx0, max(0, lx1 - lx0 - 1)),
                (x + rx0, max(0, rx1 - rx0 - 1)),
            ]

            y_junction_line = (
                int(slot.y_junction_line) if slot.y_junction_line is not None else y
            )
            junction_line_height_abs_deviation_px = abs(
                float(y_junction_line) - float(junction_line_height_baseline)
            )
            junction_line_height_ok = junction_line_height_abs_deviation_px <= float(
                junction_line_cfg.height_check_max_deviation_px
            )
            line_color = (
                overlay_cfg.ok_bgr if junction_line_height_ok else overlay_cfg.ng_bgr
            )
            for seg_x, seg_w in line_segments:
                overlay.junction_line(seg_x, seg_w, y_junction_line, line_color)
            if not junction_line_height_ok:
                fail_tracker.consider("JUNCTION_LINE_HEIGHT_NG", slot.row, slot.col)
                continue
            slots_junction_line_ok.append(slot)

        return slots_junction_line_ok

    def _process_junction_line_stage(
        self,
        slots_stem_ok: List[_SlotInspectionState],
        overlay: _OverlayRecorder,
        fail_tracker: _FailTracker,
    ) -> List[_SlotInspectionState]:
        slots_junction_line_located, junction_line_y_by_row = (
            self._locate_junction_line_candidates(slots_stem_ok, overlay, fail_tracker)
        )
        return self._validate_junction_line_height_stage(
            slots_junction_line_located,
            junction_line_y_by_row,
            overlay,
            fail_tracker,
        )

    # Detection stages: color band
    def _color_band_search_window(
        self, slot: _SlotInspectionState
    ) -> Tuple[int, int, int, int, Tuple[int, int, int, int]]:
        color_band_cfg = self.color_band_cfg
        x, y, w, h = slot.roi_rect
        y_junction_line = slot.y_junction_line
        if y_junction_line is None:
            raise ValueError("slot.y_junction_line must be set before color band stage")
        x1_max = x + w - 1
        y1_max = y + h - 1
        y0f, y1f = _clamp_span_inclusive(
            int(y_junction_line) + color_band_cfg.y_below_junction_line_from_px,
            int(y_junction_line) + color_band_cfg.y_below_junction_line_to_px,
            y,
            y1_max,
        )
        xc = _round_and_clamp_int(slot.x_stem_center, x, x1_max)
        x0f, x1f = _clamp_span_inclusive(
            xc - color_band_cfg.x_radius_from_stem_center_px,
            xc + color_band_cfg.x_radius_from_stem_center_px,
            x,
            x1_max,
        )
        return x0f, y0f, x1f, y1f, _rect_from_inclusive(x0f, y0f, x1f, y1f)

    def _find_color_band_span_in_window(
        self,
        slot: _SlotInspectionState,
        x0f: int,
        y0f: int,
        x1f: int,
        y1f: int,
    ) -> Optional[Tuple[int, int]]:
        color_band_cfg = self.color_band_cfg
        x, y, w, h = slot.roi_rect
        lx0, lx1 = _clamp_span_inclusive(x0f - x, x1f - x, 0, w - 1)
        ly0, ly1 = _clamp_span_inclusive(y0f - y, y1f - y, 0, h - 1)
        roi_hsv = slot.roi_hsv[ly0 : ly1 + 1, lx0 : lx1 + 1]
        Hc = roi_hsv[:, :, 0]
        Sc = roi_hsv[:, :, 1].astype(np.float32)
        hue_mask = _hue_in_range(Hc, color_band_cfg.hue_low, color_band_cfg.hue_high)
        s_masked = np.where(hue_mask, Sc, np.nan)
        s_col = np.nanmedian(s_masked, axis=0).astype(np.float32)
        s_col = np.nan_to_num(s_col, nan=0.0)
        s_col = _smooth_1d(
            s_col,
            color_band_cfg.saturation_smooth_window_cols,
            axis=1,
        )

        run = _find_longest_true_span(
            s_col >= float(color_band_cfg.saturation_threshold)
        )
        if run is None:
            return None

        left, right = run
        width = int(right - left + 1)
        width_too_narrow = width < int(color_band_cfg.width_check_min_px)
        width_too_wide = int(color_band_cfg.width_check_max_px) > 0 and width > int(
            color_band_cfg.width_check_max_px
        )
        if width_too_narrow or width_too_wide:
            return None
        return int(x0f + left), int(x0f + right)

    def _color_band_reference_x(
        self, slot: _SlotInspectionState, slot_count: int
    ) -> float:
        color_band_cfg = self.color_band_cfg
        t = 0.0 if slot_count <= 1 else float(slot.col) / float(slot_count - 1)
        edge_shift_px = float(color_band_cfg.alignment_reference_x_shift_px)
        base_shift = (-edge_shift_px) + (2.0 * edge_shift_px) * t
        return float(slot.x_stem_center) + base_shift

    def _is_color_band_aligned(
        self, slot: _SlotInspectionState, x_color_band_center: float, slot_count: int
    ) -> bool:
        x_ref = self._color_band_reference_x(slot, slot_count)
        tol = float(self.color_band_cfg.alignment_tolerance_px)
        return (x_ref - tol) <= x_color_band_center <= (x_ref + tol)

    def _process_color_band_stage(
        self,
        slots_junction_line_ok: List[_SlotInspectionState],
        overlay: _OverlayRecorder,
        fail_tracker: _FailTracker,
    ) -> None:
        debug_overlay_cfg = self.debug_overlay_cfg
        overlay_cfg = self.result_overlay_cfg
        slots_per_row = self.slot_layout_cfg.slots_per_row
        for slot in slots_junction_line_ok:
            if slot.y_junction_line is None:
                fail_tracker.consider("JUNCTION_LINE_NOT_FOUND_NG", slot.row, slot.col)
                continue
            x0f, y0f, x1f, y1f, color_band_search_rect = self._color_band_search_window(
                slot
            )

            if debug_overlay_cfg.color_band:
                overlay.rect(
                    color_band_search_rect,
                    debug_overlay_cfg.guide_bgr,
                    thickness=1,
                )

            if y1f <= y0f or x1f <= x0f:
                self._record_slot_failure(
                    overlay,
                    fail_tracker,
                    "COLOR_BAND_NOT_FOUND_NG",
                    slot.row,
                    slot.col,
                    rect=color_band_search_rect,
                )
                continue

            span = self._find_color_band_span_in_window(
                slot,
                x0f,
                y0f,
                x1f,
                y1f,
            )
            if span is None:
                self._record_slot_failure(
                    overlay,
                    fail_tracker,
                    "COLOR_BAND_NOT_FOUND_NG",
                    slot.row,
                    slot.col,
                    rect=color_band_search_rect,
                )
                continue

            xL, xR = span
            x_color_band_center = float((xL + xR) / 2.0)
            y_color_band_center = int(round((y0f + y1f) / 2.0))
            x_color_band_center_i = int(round(x_color_band_center))

            if debug_overlay_cfg.color_band:
                overlay.rect(
                    _rect_from_inclusive(xL, y0f, xL, y1f),
                    debug_overlay_cfg.guide_bgr,
                    thickness=1,
                )
                overlay.rect(
                    _rect_from_inclusive(xR, y0f, xR, y1f),
                    debug_overlay_cfg.guide_bgr,
                    thickness=1,
                )

            overlay.center(
                x_color_band_center_i,
                int(y_color_band_center),
                overlay_cfg.ok_bgr,
            )
            slot_count = int(slots_per_row[slot.row])
            color_band_alignment_ok = self._is_color_band_aligned(
                slot, x_color_band_center, slot_count
            )
            if not color_band_alignment_ok:
                overlay.cross(
                    x_color_band_center_i,
                    int(y_color_band_center),
                    overlay_cfg.ng_bgr,
                )
                fail_tracker.consider("COLOR_BAND_ALIGNMENT_NG", slot.row, slot.col)


__all__ = ["WeigaoTrayDetector"]
