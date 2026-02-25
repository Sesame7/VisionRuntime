import math
from dataclasses import dataclass, field
from typing import Any, List, Optional, Tuple

import cv2
import numpy as np

from .base import register_detector


def _clamp_int(v: Any, lo: int, hi: int) -> int:
    x = int(round(float(v)))
    return lo if x < lo else hi if x > hi else x


def _h_in_range(h: np.ndarray, lo: int, hi: int) -> np.ndarray:
    lo = int(lo) % 180
    hi = int(hi) % 180
    if lo <= hi:
        return (h >= lo) & (h <= hi)
    return (h >= lo) | (h <= hi)


def _longest_true_run(b: np.ndarray) -> Optional[Tuple[int, int]]:
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


def _clamp_span_inclusive(lo: Any, hi: Any, vmin: int, vmax: int) -> Tuple[int, int]:
    lo_i = _clamp_int(lo, vmin, vmax)
    hi_i = _clamp_int(hi, vmin, vmax)
    if hi_i <= lo_i:
        hi_i = min(vmax, lo_i + 1)
    return lo_i, hi_i


def _clamp_span_exclusive(
    lo: Any, hi: Any, vmin: int, vmax_excl: int
) -> Tuple[int, int]:
    lo_i = _clamp_int(lo, vmin, max(vmin, vmax_excl - 1))
    hi_i = _clamp_int(hi, vmin, vmax_excl)
    if hi_i <= lo_i:
        hi_i = min(vmax_excl, lo_i + 1)
    return lo_i, hi_i


def _row_mean(mask_u8: np.ndarray, n_rows: int) -> np.ndarray:
    if mask_u8.size == 0:
        return np.zeros(n_rows, dtype=np.float32)
    return (mask_u8.mean(axis=1) / 255.0).astype(np.float32)


def _parse_xy_point(value: Any, field_name: str) -> Tuple[int, int]:
    if not isinstance(value, dict):
        raise ValueError(
            f"{field_name} entries must be objects like {{x: ..., y: ...}}"
        )
    if "x" not in value or "y" not in value:
        raise ValueError(f"{field_name} entries must contain x and y")
    return int(value["x"]), int(value["y"])


def _clamp_u8(v: Any) -> int:
    return int(max(0, min(255, int(v))))


@dataclass
class _SlotD:
    row: int
    col: int
    roi: Tuple[int, int, int, int]  # x,y,w,h
    x_center: float  # full-image x (float)
    y_junction: int  # full-image y
    hsv: np.ndarray  # ROI HSV image


@dataclass
class _FailTracker:
    detector: Any
    first_fail: Optional[Tuple[str, int, int]] = None
    first_fail_key: Optional[Tuple[int, int, int]] = None

    def consider(self, code: str, row: int, col: int) -> None:
        key = (
            int(row),
            int(col),
            int(self.detector._FAIL_STAGE.get(code, 99)),
        )
        if self.first_fail_key is None or key < self.first_fail_key:
            self.first_fail_key = key
            self.first_fail = (str(code), int(row), int(col))


@dataclass
class _OverlayRecorder:
    detector: Any
    enabled: bool
    ops: List[Tuple[str, tuple]] = field(default_factory=list)

    def rect(
        self,
        rect: Tuple[int, int, int, int],
        color: Tuple[int, int, int],
        thickness: int | None = None,
    ) -> None:
        if self.enabled:
            self.ops.append(("rect", (rect, color, thickness)))

    def center(self, cx: int, cy: int, color: Tuple[int, int, int]) -> None:
        if self.enabled:
            self.ops.append(("center", (int(cx), int(cy), color)))

    def junction_line(
        self, x: int, w: int, y: int, color: Tuple[int, int, int]
    ) -> None:
        if self.enabled:
            self.ops.append(("junction_line", (int(x), int(w), int(y), color)))

    def mask_overlay(
        self,
        x: int,
        y: int,
        mask_u8: np.ndarray,
        color: Tuple[int, int, int],
        alpha: float = 0.35,
    ) -> None:
        if self.enabled:
            self.ops.append(
                ("mask_overlay", (int(x), int(y), mask_u8, color, float(alpha)))
            )

    def render(self, img: Optional[np.ndarray]) -> None:
        if not self.enabled or img is None or not self.ops:
            return
        for kind, args in self.ops:
            if kind == "rect":
                rect, color, thickness = args
                self.detector._draw_rect(img, rect, color, thickness)
            elif kind == "center":
                cx, cy, color = args
                self.detector._draw_center(img, cx, cy, color)
            elif kind == "junction_line":
                x, w, y, color = args
                self.detector._draw_junction_line(img, x, w, y, color)
            elif kind == "mask_overlay":
                x, y, mask_u8, color, alpha = args
                self.detector._draw_mask_overlay(img, x, y, mask_u8, color, alpha)


@dataclass
class _RodStageCfg:
    min_mask_area: float


@dataclass
class _JunctionStageCfg:
    bottom_anchor_percentile: float
    y_window_up_px: int
    y_window_down_px: int
    strip_center_offset_x_px: int
    strip_width_px: int


@dataclass
class _SplitStageCfg:
    smooth_half_window_rows: int
    min_rows_above: int
    min_rows_below: int
    kernel_size: int


@dataclass
class _HeightStageCfg:
    max_upward_deviation_px: int


@dataclass
class _BandStageCfg:
    y_from_junction_top_px: int
    y_from_junction_bottom_px: int
    x_half_width_px: int
    hue_low: int
    hue_high: int
    saturation_smooth_half_window_cols: int
    saturation_threshold: int
    width_min_px: int
    width_max_px: int
    kernel_size: int


@dataclass
class _OffsetStageCfg:
    reference_edge_shift_px: float
    tolerance_px: float


@dataclass
class _GridCfg:
    rows: int
    cols_per_row: List[int]
    anchor_left: List[Tuple[int, int]]
    anchor_right: List[Tuple[int, int]]
    roi_w: int
    roi_h: int


@dataclass
class _WhiteRowCfg:
    s_max: int
    v_min: int
    lower: np.ndarray
    upper: np.ndarray


@dataclass
class _WhiteCfg:
    by_row: List[_WhiteRowCfg]


@dataclass
class _OverlayStyleCfg:
    ok_bgr: Tuple[int, int, int]
    ng_bgr: Tuple[int, int, int]
    line_width: int


@dataclass
class _DebugOverlayCfg:
    show_mask: bool
    show_strips: bool
    show_band: bool
    bgr: Tuple[int, int, int]
    mask_bgr: Tuple[int, int, int]


def _as_bgr_triplet(value: Any, default: Tuple[int, int, int]) -> Tuple[int, int, int]:
    if isinstance(value, (list, tuple)) and len(value) == 3:
        b, g, r = value
        return (int(b), int(g), int(r))
    return default


@register_detector("weigao_tray")
class WeigaoTrayDetector:
    """
    Weigao tray inspection (configurable rows/columns), adapted for Smart_Camera.

    - Overlay uses input image as the base.
    - Cumulative drawing: green marks keep even if later stages fail.
    """

    _FAIL_STAGE = {
        "ROD_NOT_FOUND": 0,
        "JUNCTION_NOT_FOUND": 1,
        "HEIGHT_NG": 2,
        "BAND_NOT_FOUND": 3,
        "OFFSET_NG": 4,
    }

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

    def _load_params(self) -> None:
        p = self.params
        self.grid_cfg = self._parse_grid_cfg(p)
        self.yellow_lower, self.yellow_upper = self._parse_yellow_hsv(p)
        self.rod_cfg = self._parse_rod_cfg(p)

        j = p.get("junction") or {}
        self.junction_cfg = self._parse_junction_cfg(j)
        self.white_cfg = self._parse_white_cfg(j)
        self.split_cfg = self._parse_split_cfg(j)
        self.height_cfg = self._parse_height_cfg(j)

        self.band_cfg = self._parse_band_cfg(p)
        self.offset_cfg = self._parse_offset_cfg(p)
        self.overlay_cfg = self._parse_overlay_cfg(p)
        self.dbg_cfg = self._parse_dbg_cfg(p)

        self._validate()

    @staticmethod
    def _kernel_size_from_half_window(half_window: int) -> int:
        return int(2 * half_window + 1) if int(half_window) > 0 else 0

    def _parse_grid_cfg(self, p: dict) -> _GridCfg:
        grid = p.get("grid") or {}
        roi = grid.get("roi") or {}
        anchor_left_raw = list(grid.get("anchor_left") or [])
        anchor_right_raw = list(grid.get("anchor_right") or [])
        cols_per_row = list(grid.get("cols_per_row") or [17, 17, 17])
        rows = len(cols_per_row)
        return _GridCfg(
            rows=rows,
            cols_per_row=cols_per_row,
            anchor_left=[
                _parse_xy_point(v, "grid.anchor_left") for v in anchor_left_raw
            ],
            anchor_right=[
                _parse_xy_point(v, "grid.anchor_right") for v in anchor_right_raw
            ],
            roi_w=int(roi.get("width", 100)),
            roi_h=int(roi.get("height", 200)),
        )

    def _parse_yellow_hsv(self, p: dict) -> Tuple[np.ndarray, np.ndarray]:
        yh = p.get("yellow_hsv") or {}
        lower = np.array(yh.get("lower") or [10, 160, 40], dtype=np.uint8)
        upper = np.array(yh.get("upper") or [30, 255, 255], dtype=np.uint8)
        return lower, upper

    def _parse_rod_cfg(self, p: dict) -> _RodStageCfg:
        rod = p.get("rod") or {}
        return _RodStageCfg(
            min_mask_area=float(rod.get("min_mask_area", 3500)),
        )

    def _parse_junction_cfg(self, j: dict) -> _JunctionStageCfg:
        search_region = j.get("search_region") or {}
        y_window = search_region.get("y_window") or {}
        strips = search_region.get("strips") or {}
        return _JunctionStageCfg(
            bottom_anchor_percentile=float(j.get("bottom_anchor_percentile", 98.0)),
            y_window_up_px=int(y_window.get("up_px", 60)),
            y_window_down_px=int(y_window.get("down_px", 30)),
            strip_center_offset_x_px=int(strips.get("center_offset_x_px", 24)),
            strip_width_px=int(strips.get("width_px", 24)),
        )

    def _parse_white_cfg(self, j: dict) -> _WhiteCfg:
        white = j.get("white")
        if white is None:
            white_rows_raw: List[Any] = [{} for _ in range(self.grid_cfg.rows)]
        elif isinstance(white, list):
            white_rows_raw = list(white)
        else:
            raise ValueError("junction.white must be a list of row configs")

        white_rows: List[_WhiteRowCfg] = []
        for i, item in enumerate(white_rows_raw):
            if item is None:
                item = {}
            if not isinstance(item, dict):
                raise ValueError(f"junction.white[{i}] must be an object")
            s_max = int(item.get("S_max", 50))
            v_min = int(item.get("V_min", 50))
            white_rows.append(
                _WhiteRowCfg(
                    s_max=s_max,
                    v_min=v_min,
                    lower=np.array([0, 0, _clamp_u8(v_min)], dtype=np.uint8),
                    upper=np.array([179, _clamp_u8(s_max), 255], dtype=np.uint8),
                )
            )
        return _WhiteCfg(by_row=white_rows)

    def _parse_split_cfg(self, j: dict) -> _SplitStageCfg:
        split = j.get("split") or {}
        split_min_rows = split.get("min_rows") or {}
        smooth_half_window_rows = int(split.get("smooth_half_window_rows", 3))
        return _SplitStageCfg(
            smooth_half_window_rows=smooth_half_window_rows,
            min_rows_above=int(split_min_rows.get("above", 2)),
            min_rows_below=int(split_min_rows.get("below", 2)),
            kernel_size=self._kernel_size_from_half_window(smooth_half_window_rows),
        )

    def _parse_height_cfg(self, j: dict) -> _HeightStageCfg:
        junction_height = j.get("height") or {}
        return _HeightStageCfg(
            max_upward_deviation_px=int(
                junction_height.get("max_upward_deviation_px", 12)
            )
        )

    def _parse_band_cfg(self, p: dict) -> _BandStageCfg:
        band = p.get("band") or {}
        band_search_region = band.get("search_region") or {}
        band_y_from_junction = band_search_region.get("y_from_junction_px") or {}
        band_x_window = band_search_region.get("x_window") or {}
        band_color = band.get("color") or {}
        band_hue_range = band_color.get("hue_range") or {}
        band_saturation = band_color.get("saturation") or {}
        band_width_px = band.get("width_px") or {}

        smooth_half_window_cols = int(band_saturation.get("smooth_half_window_cols", 2))
        return _BandStageCfg(
            y_from_junction_top_px=int(band_y_from_junction.get("top", 3)),
            y_from_junction_bottom_px=int(band_y_from_junction.get("bottom", 30)),
            x_half_width_px=int(band_x_window.get("half_width_px", 30)),
            hue_low=int(band_hue_range.get("low", 0)),
            hue_high=int(band_hue_range.get("high", 179)),
            saturation_smooth_half_window_cols=smooth_half_window_cols,
            saturation_threshold=int(band_saturation.get("threshold", 30)),
            width_min_px=int(band_width_px.get("min", 10)),
            width_max_px=int(band_width_px.get("max", 0)),
            kernel_size=self._kernel_size_from_half_window(smooth_half_window_cols),
        )

    def _parse_offset_cfg(self, p: dict) -> _OffsetStageCfg:
        offset = p.get("offset") or {}
        return _OffsetStageCfg(
            reference_edge_shift_px=float(offset.get("reference_edge_shift_px", 3)),
            tolerance_px=float(offset.get("tolerance_px", 11)),
        )

    def _parse_overlay_cfg(self, p: dict) -> _OverlayStyleCfg:
        overlay = p.get("overlay") or {}
        return _OverlayStyleCfg(
            ok_bgr=_as_bgr_triplet(overlay.get("ok_bgr"), (0, 200, 0)),
            ng_bgr=_as_bgr_triplet(overlay.get("ng_bgr"), (0, 0, 220)),
            line_width=max(1, int(overlay.get("line_width", 2))),
        )

    def _parse_dbg_cfg(self, p: dict) -> _DebugOverlayCfg:
        dbg = p.get("dbg") or {}
        return _DebugOverlayCfg(
            show_mask=bool(dbg.get("show_mask", False)),
            show_strips=bool(dbg.get("show_strips", False)),
            show_band=bool(dbg.get("show_band", False)),
            bgr=(220, 0, 0),  # blue in BGR
            mask_bgr=(140, 60, 0),  # darker orange in BGR
        )

    def _validate(self) -> None:
        grid_cfg = self.grid_cfg
        if grid_cfg.rows <= 0:
            raise ValueError("grid.cols_per_row must not be empty")
        if len(grid_cfg.cols_per_row) != grid_cfg.rows:
            raise ValueError("internal grid row count mismatch")
        if any(int(c) <= 0 for c in grid_cfg.cols_per_row):
            raise ValueError("grid.cols_per_row values must be > 0")
        if (
            len(grid_cfg.anchor_left) != grid_cfg.rows
            or len(grid_cfg.anchor_right) != grid_cfg.rows
        ):
            raise ValueError(
                "grid.anchor_left/right length must equal len(grid.cols_per_row)"
            )
        if grid_cfg.roi_w <= 0 or grid_cfg.roi_h <= 0:
            raise ValueError("grid.roi.width and grid.roi.height must be > 0")
        if self.yellow_lower.shape != (3,) or self.yellow_upper.shape != (3,):
            raise ValueError("yellow_hsv.lower/upper must be 3-element lists")
        if len(self.white_cfg.by_row) != grid_cfg.rows:
            raise ValueError("junction.white length must equal len(grid.cols_per_row)")
        if not (0.0 <= self.junction_cfg.bottom_anchor_percentile <= 100.0):
            raise ValueError("junction.bottom_anchor_percentile must be in [0,100]")
        if self.junction_cfg.strip_width_px <= 0:
            raise ValueError("junction.search_region.strips.width_px must be > 0")
        if self.band_cfg.width_min_px < 0 or self.band_cfg.width_max_px < 0:
            raise ValueError("band.width_px.min/max must be >= 0")

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

    def _generate_rois(
        self, img_w: int, img_h: int
    ) -> List[Tuple[int, int, int, int, int, int]]:
        grid_cfg = self.grid_cfg
        rois: List[Tuple[int, int, int, int, int, int]] = []
        w = max(1, min(int(grid_cfg.roi_w), int(img_w)))
        h = max(1, min(int(grid_cfg.roi_h), int(img_h)))
        for r in range(grid_cfg.rows):
            cols = int(grid_cfg.cols_per_row[r])
            xL, yT = grid_cfg.anchor_left[r]
            xR, _yT2 = grid_cfg.anchor_right[r]
            xL = _clamp_int(xL, 0, max(0, img_w - w))
            xR = _clamp_int(xR, 0, max(0, img_w - w))
            yT = _clamp_int(yT, 0, max(0, img_h - h))
            if xR < xL:
                xR = xL
            dx = 0.0 if cols <= 1 else (xR - xL) / float(cols - 1)
            for c in range(cols):
                x = _clamp_int(xL + c * dx, 0, max(0, img_w - w))
                rois.append((r, c, x, yT, w, h))
        return rois

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
                int(self.overlay_cfg.line_width if thickness is None else thickness),
            ),
        )

    def _draw_center(
        self, img: np.ndarray, cx: int, cy: int, color: Tuple[int, int, int]
    ) -> None:
        r = max(1, int(self.overlay_cfg.line_width))
        cv2.circle(img, (int(cx), int(cy)), r, color, thickness=-1)

    def _draw_junction_line(
        self, img: np.ndarray, x: int, w: int, y: int, color: Tuple[int, int, int]
    ) -> None:
        cv2.line(
            img,
            (int(x), int(y)),
            (int(x + w), int(y)),
            color,
            thickness=self.overlay_cfg.line_width,
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

    def _junction_strip_spans(
        self, cx_in_roi: float, roi_w: int
    ) -> Tuple[int, int, int, int]:
        junction_cfg = self.junction_cfg
        cx_i = _clamp_int(cx_in_roi, 0, roi_w - 1)
        d = int(junction_cfg.strip_center_offset_x_px)
        ws = max(1, int(round(junction_cfg.strip_width_px / 2.0)))
        lx0, lx1 = _clamp_span_exclusive(cx_i - d - ws, cx_i - d + ws, 0, roi_w)
        rx0, rx1 = _clamp_span_exclusive(cx_i + d - ws, cx_i + d + ws, 0, roi_w)
        return lx0, lx1, rx0, rx1

    def _collect_slots_until_junction(
        self,
        img: np.ndarray,
        rois: List[Tuple[int, int, int, int, int, int]],
        overlay: _OverlayRecorder,
        fail_tracker: _FailTracker,
    ) -> Tuple[List[_SlotD], List[List[int]]]:
        rod_cfg = self.rod_cfg
        junction_cfg = self.junction_cfg
        split_cfg = self.split_cfg
        white_cfg = self.white_cfg
        dbg_cfg = self.dbg_cfg
        overlay_cfg = self.overlay_cfg
        slots_d_ok: List[_SlotD] = []
        junction_by_row: List[List[int]] = [[] for _ in range(self.grid_cfg.rows)]

        for r, c, x, y, w, h in rois:
            white_row_cfg = white_cfg.by_row[r]
            roi_bgr = img[y : y + h, x : x + w]
            roi_hsv = cv2.cvtColor(roi_bgr, cv2.COLOR_BGR2HSV)
            core = cv2.inRange(roi_hsv, self.yellow_lower, self.yellow_upper)
            self._keep_largest_component_inplace(core)

            if dbg_cfg.show_mask:
                overlay.mask_overlay(x, y, core, dbg_cfg.mask_bgr, alpha=0.55)

            M = cv2.moments(core, binaryImage=True)
            m00 = float(M.get("m00", 0.0))
            if m00 < float(rod_cfg.min_mask_area):
                overlay.rect((x, y, w, h), overlay_cfg.ng_bgr)
                fail_tracker.consider("ROD_NOT_FOUND", r, c)
                continue

            cx = float(M["m10"] / m00)
            cy = float(M["m01"] / m00)
            Cx = int(round(x + cx))
            Cy = int(round(y + cy))
            overlay.center(Cx, Cy, overlay_cfg.ok_bgr)

            row_cnt = (core > 0).sum(axis=1).astype(np.int32)
            total = int(row_cnt.sum())

            cum = np.cumsum(row_cnt, dtype=np.int64)
            pct = (
                max(0.0, min(100.0, float(junction_cfg.bottom_anchor_percentile)))
                / 100.0
            )
            rank = int(math.floor(pct * (total - 1))) if total > 1 else 0
            y_rod_bottom = int(np.searchsorted(cum, rank + 1, side="left"))

            y0, y1 = _clamp_span_inclusive(
                y_rod_bottom - junction_cfg.y_window_up_px,
                y_rod_bottom + junction_cfg.y_window_down_px,
                0,
                h - 1,
            )

            lx0, lx1, rx0, rx1 = self._junction_strip_spans(cx, w)

            if dbg_cfg.show_strips:
                overlay.rect(
                    _rect_from_inclusive(x + lx0, y + y0, x + lx1 - 1, y + y1),
                    dbg_cfg.bgr,
                    thickness=1,
                )
                overlay.rect(
                    _rect_from_inclusive(x + rx0, y + y0, x + rx1 - 1, y + y1),
                    dbg_cfg.bgr,
                    thickness=1,
                )

            win_hsv = roi_hsv[y0 : y1 + 1, :]
            win = cv2.inRange(win_hsv, white_row_cfg.lower, white_row_cfg.upper)
            n_rows = int(y1 - y0 + 1)
            l_mean = _row_mean(win[:, lx0:lx1], n_rows)
            r_mean = _row_mean(win[:, rx0:rx1], n_rows)
            score = 0.5 * (l_mean + r_mean)

            if split_cfg.kernel_size > 0 and score.size > 1:
                score_s = cv2.blur(
                    score.reshape(-1, 1),
                    (1, split_cfg.kernel_size),
                    borderType=cv2.BORDER_REPLICATE,
                ).reshape(-1)
            else:
                score_s = score

            n = int(score_s.size)
            k0 = max(1, int(split_cfg.min_rows_above))
            k1 = n - max(1, int(split_cfg.min_rows_below))
            strip_rects = [
                _rect_from_inclusive(x + lx0, y + y0, x + lx1 - 1, y + y1),
                _rect_from_inclusive(x + rx0, y + y0, x + rx1 - 1, y + y1),
            ]
            if n < 2 or k0 > k1:
                for rect in strip_rects:
                    overlay.rect(rect, overlay_cfg.ng_bgr)
                fail_tracker.consider("JUNCTION_NOT_FOUND", r, c)
                continue

            ps = np.concatenate(([0.0], np.cumsum(score_s, dtype=np.float32)))
            ks = np.arange(k0, k1 + 1, dtype=np.int32)
            sum_above = ps[ks]
            sum_below = ps[n] - sum_above
            mu_above = sum_above / ks
            mu_below = sum_below / (n - ks)
            contrast = mu_below - mu_above
            idx = int(np.argmax(contrast))
            best_k = int(ks[idx])

            y_junction = int(y + y0 + best_k)
            slots_d_ok.append(
                _SlotD(
                    row=r,
                    col=c,
                    roi=(x, y, w, h),
                    x_center=float(x + cx),
                    y_junction=y_junction,
                    hsv=roi_hsv,
                )
            )
            junction_by_row[r].append(y_junction)

        return slots_d_ok, junction_by_row

    def _apply_height_stage(
        self,
        slots_d_ok: List[_SlotD],
        junction_by_row: List[List[int]],
        overlay: _OverlayRecorder,
        fail_tracker: _FailTracker,
    ) -> List[_SlotD]:
        height_cfg = self.height_cfg
        overlay_cfg = self.overlay_cfg
        baseline_by_row: List[Optional[float]] = [
            float(np.median(ys)) if ys else None for ys in junction_by_row
        ]

        slots_height_ok: List[_SlotD] = []
        for slot in slots_d_ok:
            base = baseline_by_row[slot.row]
            x, y, w, h = slot.roi
            if base is None:
                overlay.rect((x, y, w, h), overlay_cfg.ng_bgr)
                fail_tracker.consider("JUNCTION_NOT_FOUND", slot.row, slot.col)
                continue

            lx0, lx1, rx0, rx1 = self._junction_strip_spans(slot.x_center - x, w)
            # Draw only on left/right strips to avoid covering the center feature.
            line_segments = [
                (x + lx0, max(0, lx1 - lx0 - 1)),
                (x + rx0, max(0, rx1 - rx0 - 1)),
            ]

            threshold = float(base) - float(height_cfg.max_upward_deviation_px)
            height_ok = float(slot.y_junction) >= threshold
            line_color = overlay_cfg.ok_bgr if height_ok else overlay_cfg.ng_bgr
            for seg_x, seg_w in line_segments:
                overlay.junction_line(seg_x, seg_w, slot.y_junction, line_color)
            if not height_ok:
                for seg_x, seg_w in line_segments:
                    overlay.junction_line(
                        seg_x, seg_w, slot.y_junction, overlay_cfg.ng_bgr
                    )
                fail_tracker.consider("HEIGHT_NG", slot.row, slot.col)
                continue
            slots_height_ok.append(slot)

        return slots_height_ok

    def _apply_band_offset_stage(
        self,
        slots_height_ok: List[_SlotD],
        overlay: _OverlayRecorder,
        fail_tracker: _FailTracker,
    ) -> None:
        band_cfg = self.band_cfg
        offset_cfg = self.offset_cfg
        dbg_cfg = self.dbg_cfg
        overlay_cfg = self.overlay_cfg
        for slot in slots_height_ok:
            x, y, w, h = slot.roi
            cols = int(self.grid_cfg.cols_per_row[slot.row])
            x1_max = x + w - 1
            y1_max = y + h - 1

            y0f, y1f = _clamp_span_inclusive(
                slot.y_junction + band_cfg.y_from_junction_top_px,
                slot.y_junction + band_cfg.y_from_junction_bottom_px,
                y,
                y1_max,
            )
            xc = _clamp_int(slot.x_center, x, x1_max)
            x0f, x1f = _clamp_span_inclusive(
                xc - band_cfg.x_half_width_px, xc + band_cfg.x_half_width_px, x, x1_max
            )
            band_rect = _rect_from_inclusive(x0f, y0f, x1f, y1f)

            if dbg_cfg.show_band:
                overlay.rect(band_rect, dbg_cfg.bgr, thickness=1)

            if y1f <= y0f or x1f <= x0f:
                overlay.rect(band_rect, overlay_cfg.ng_bgr)
                fail_tracker.consider("BAND_NOT_FOUND", slot.row, slot.col)
                continue

            lx0, lx1 = _clamp_span_inclusive(x0f - x, x1f - x, 0, w - 1)
            ly0, ly1 = _clamp_span_inclusive(y0f - y, y1f - y, 0, h - 1)
            roi_hsv = slot.hsv[ly0 : ly1 + 1, lx0 : lx1 + 1]
            Hc = roi_hsv[:, :, 0]
            Sc = roi_hsv[:, :, 1].astype(np.float32)
            m = _h_in_range(Hc, band_cfg.hue_low, band_cfg.hue_high)
            s_masked = np.where(m, Sc, np.nan)
            s_col = np.nanmedian(s_masked, axis=0).astype(np.float32)
            s_col = np.nan_to_num(s_col, nan=0.0)
            if band_cfg.kernel_size > 0 and s_col.size > 1:
                s_col = cv2.blur(
                    s_col.reshape(1, -1),
                    (band_cfg.kernel_size, 1),
                    borderType=cv2.BORDER_REPLICATE,
                ).reshape(-1)

            run = _longest_true_run(s_col >= float(band_cfg.saturation_threshold))
            if run is None:
                overlay.rect(band_rect, overlay_cfg.ng_bgr)
                fail_tracker.consider("BAND_NOT_FOUND", slot.row, slot.col)
                continue

            L, R = run
            width = int(R - L + 1)
            if width < int(band_cfg.width_min_px) or (
                int(band_cfg.width_max_px) > 0 and width > int(band_cfg.width_max_px)
            ):
                overlay.rect(band_rect, overlay_cfg.ng_bgr)
                fail_tracker.consider("BAND_NOT_FOUND", slot.row, slot.col)
                continue

            xL = int(x0f + L)
            xR = int(x0f + R)
            x_band = float((xL + xR) / 2.0)
            y_mid = int(round((y0f + y1f) / 2.0))

            if dbg_cfg.show_band:
                overlay.rect(
                    _rect_from_inclusive(xL, y0f, xL, y1f), dbg_cfg.bgr, thickness=1
                )
                overlay.rect(
                    _rect_from_inclusive(xR, y0f, xR, y1f), dbg_cfg.bgr, thickness=1
                )

            t = 0.0 if cols <= 1 else float(slot.col) / float(cols - 1)
            edge_shift_px = float(offset_cfg.reference_edge_shift_px)
            base_shift = (-edge_shift_px) + (2.0 * edge_shift_px) * t
            x_ref = float(slot.x_center) + base_shift
            offset_ok = (
                (x_ref - float(offset_cfg.tolerance_px))
                <= x_band
                <= (x_ref + float(offset_cfg.tolerance_px))
            )

            overlay.center(
                int(round(x_band)),
                int(y_mid),
                overlay_cfg.ok_bgr if offset_ok else overlay_cfg.ng_bgr,
            )
            if not offset_ok:
                overlay.center(int(round(x_band)), int(y_mid), overlay_cfg.ng_bgr)
                fail_tracker.consider("OFFSET_NG", slot.row, slot.col)

    def _finalize_result(
        self, overlay_img: Optional[np.ndarray], fail_tracker: _FailTracker
    ):
        if fail_tracker.first_fail is None:
            return True, "OK", overlay_img, "OK"
        code, rr, cc = fail_tracker.first_fail
        return False, f"NG: {code} r={rr} c={cc}", overlay_img, code

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
        rois = self._generate_rois(img_w, img_h)

        overlay_img = img.copy() if self.generate_overlay else None
        overlay = _OverlayRecorder(detector=self, enabled=overlay_img is not None)
        fail_tracker = _FailTracker(detector=self)
        slots_d_ok, junction_by_row = self._collect_slots_until_junction(
            img, rois, overlay, fail_tracker
        )
        slots_height_ok = self._apply_height_stage(
            slots_d_ok, junction_by_row, overlay, fail_tracker
        )
        self._apply_band_offset_stage(slots_height_ok, overlay, fail_tracker)
        overlay.render(overlay_img)
        return self._finalize_result(overlay_img, fail_tracker)


__all__ = ["WeigaoTrayDetector"]
