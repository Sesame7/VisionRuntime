
import math
from dataclasses import dataclass
from typing import Any, Dict, List, Optional, Tuple

import cv2
import numpy as np

from .base import register_detector


def _odd_k(k: Any) -> int:
	try:
		v = int(float(k))
	except Exception:
		v = 1
	return max(1, v) | 1


def _clamp_int(v: Any, lo: int, hi: int) -> int:
	try:
		x = int(round(float(v)))
	except Exception:
		x = lo
	return lo if x < lo else hi if x > hi else x


def _theta_from_central_moments_deg(M: Dict[str, float]) -> float:
	mu11 = float(M.get("mu11", 0.0))
	mu20 = float(M.get("mu20", 0.0))
	mu02 = float(M.get("mu02", 0.0))
	theta = 0.5 * math.atan2(2.0 * mu11, mu20 - mu02)
	return float((math.degrees(theta) + 180.0) % 180.0)


def _h_in_range(h: np.ndarray, lo: int, hi: int) -> np.ndarray:
	lo = int(lo) % 180
	hi = int(hi) % 180
	if lo <= hi:
		return (h >= lo) & (h <= hi)
	return (h >= lo) | (h <= hi)


def _longest_true_run(b: np.ndarray) -> Optional[Tuple[int, int]]:
	if b.size == 0 or not bool(np.any(b)):
		return None
	d = np.diff(np.concatenate((np.array([False]), b.astype(bool, copy=False), np.array([False]))).view(np.int8))
	starts = np.flatnonzero(d == 1)
	ends = np.flatnonzero(d == -1) - 1
	if starts.size == 0:
		return None
	lengths = ends - starts + 1
	i = int(np.argmax(lengths))
	return int(starts[i]), int(ends[i])


@dataclass
class _SlotD:
	row: int
	col: int
	roi: Tuple[int, int, int, int]  # x,y,w,h
	x_center: float  # full-image x (float)
	y_junction: int  # full-image y


def _as_bgr_triplet(value: Any, default: Tuple[int, int, int]) -> Tuple[int, int, int]:
	if isinstance(value, (list, tuple)) and len(value) == 3:
		try:
			b, g, r = value
			return (int(b), int(g), int(r))
		except Exception:
			return default
	return default


@register_detector("weigao_tray")
class WeigaoTrayDetector:
	"""
	Weigao tray inspection (3 rows with configurable columns), adapted for Smart_Camera.

	- No UI/debug overlays.
	- Overlay uses denoised image as the base.
	- Cumulative drawing: green marks keep even if later stages fail.
	"""

	def __init__(self, params: dict, generate_overlay: bool = True, input_pixel_format: str | None = None):
		self.params = params or {}
		self.generate_overlay = bool(generate_overlay)
		if input_pixel_format and input_pixel_format.lower() != "bgr8":
			raise ValueError("weigao_tray requires camera.output_pixel_format=bgr8")

		self._load_params()
		self._cached_rois: Optional[List[Tuple[int, int, int, int, int, int]]] = None
		self._cached_shape: Optional[Tuple[int, int]] = None  # (w, h)

		# Reusable buffers for keep-largest contour filling (allocated lazily).
		self._tmp_u8: Optional[np.ndarray] = None
		self._keep_mask: Optional[np.ndarray] = None

	def _load_params(self) -> None:
		p = self.params

		denoise = p.get("denoise") or {}
		self.denoise_k = _odd_k(denoise.get("k", 7))
		self.denoise_sigma = float(denoise.get("sigma", 2.0))

		grid = p.get("grid") or {}
		self.rows = int(grid.get("rows", 3))
		self.cols_per_row = list(grid.get("cols_per_row") or [17, 17, 17])
		self.anchor_left = list(grid.get("anchor_left") or [])
		self.anchor_right = list(grid.get("anchor_right") or [])
		roi = grid.get("roi") or {}
		self.roi_w = int(roi.get("w", 100))
		self.roi_h = int(roi.get("h", 200))

		yh = p.get("yellow_hsv") or {}
		self.yellow_lower = np.array(yh.get("lower") or [10, 160, 40], dtype=np.uint8)
		self.yellow_upper = np.array(yh.get("upper") or [30, 255, 255], dtype=np.uint8)

		morph = p.get("morph") or {}
		self.open_k = _odd_k(morph.get("open_k", 7))
		self.close_k = _odd_k(morph.get("close_k", 7))
		self.erode_k = _odd_k(morph.get("erode_k", 3))
		self.erode_iter = int(morph.get("erode_iter", 1))
		self.keep_largest = bool(morph.get("keep_largest", True))

		self.ker_open = None if self.open_k <= 1 else np.ones((self.open_k, self.open_k), np.uint8)
		self.ker_close = None if self.close_k <= 1 else np.ones((self.close_k, self.close_k), np.uint8)
		self.ker_erode = None if self.erode_k <= 1 else np.ones((self.erode_k, self.erode_k), np.uint8)

		rod = p.get("rod") or {}
		self.theta_tol_deg = float(rod.get("vertical_tol_deg", 5.0))
		self.min_bbox_area = int(rod.get("min_bbox_area", 3500))

		j = p.get("junction") or {}
		self.rod_bottom_percentile = float(j.get("rod_bottom_percentile", 98.0))
		self.win_up = int(j.get("win_up", 60))
		self.win_down = int(j.get("win_down", 30))
		self.strip_offset_x = int(j.get("strip_offset_x", 24))
		self.strip_half_width = int(j.get("strip_half_width", 12))

		white = p.get("white") or {}
		self.white_s_max = int(white.get("S_max", 50))
		self.white_v_min = int(white.get("V_min", 50))

		split = p.get("split") or {}
		self.split_smooth_half_win = int(split.get("smooth_half_win", 3))
		self.min_rows_above = int(split.get("min_rows_above", 2))
		self.min_rows_below = int(split.get("min_rows_below", 2))
		self.min_delta_white = float(split.get("min_delta_white", 0.5))
		self.split_kernel = None
		if self.split_smooth_half_win > 0:
			k = 2 * self.split_smooth_half_win + 1
			self.split_kernel = (np.ones(k, dtype=np.float32) / float(k)).astype(np.float32, copy=False)

		height = p.get("height") or {}
		self.height_delta_up = int(height.get("delta_up", 12))

		band = p.get("band") or {}
		self.band_y0 = int(band.get("y0", 3))
		self.band_y1 = int(band.get("y1", 30))
		self.band_half_width = int(band.get("half_width", 30))
		self.band_h_low = int(band.get("H_low", 0))
		self.band_h_high = int(band.get("H_high", 179))
		self.band_smooth_half_win = int(band.get("smooth_half_win", 2))
		self.band_s_thr = int(band.get("S_thr", 30))
		self.band_min_width = int(band.get("min_width", 10))
		self.band_max_width = int(band.get("max_width", 0))
		self.band_kernel = None
		if self.band_smooth_half_win > 0:
			k = 2 * self.band_smooth_half_win + 1
			self.band_kernel = (np.ones(k, dtype=np.float32) / float(k)).astype(np.float32, copy=False)

		offset = p.get("offset") or {}
		self.offset_base = float(offset.get("offset_base", 3))
		self.offset_tol = float(offset.get("offset_tol", 11))

		overlay = p.get("overlay") or {}
		self.ok_bgr = _as_bgr_triplet(overlay.get("ok_bgr"), (0, 200, 0))
		self.ng_bgr = _as_bgr_triplet(overlay.get("ng_bgr"), (0, 0, 220))
		self.line_width = max(1, int(overlay.get("line_width", 2)))

		self._validate()

	def _validate(self) -> None:
		if self.rows != 3:
			raise ValueError("grid.rows must be 3")
		if len(self.cols_per_row) != 3:
			raise ValueError("grid.cols_per_row must have length 3")
		if any(int(c) <= 0 for c in self.cols_per_row):
			raise ValueError("grid.cols_per_row values must be > 0")
		if len(self.anchor_left) != 3 or len(self.anchor_right) != 3:
			raise ValueError("grid.anchor_left/right must have length 3")
		if self.roi_w <= 0 or self.roi_h <= 0:
			raise ValueError("grid.roi.w and grid.roi.h must be > 0")
		if self.yellow_lower.shape != (3,) or self.yellow_upper.shape != (3,):
			raise ValueError("yellow_hsv.lower/upper must be 3-element lists")
		if not (0.0 <= self.rod_bottom_percentile <= 100.0):
			raise ValueError("junction.rod_bottom_percentile must be in [0,100]")
		if self.min_delta_white < 0:
			raise ValueError("split.min_delta_white must be >= 0")
		if self.band_min_width < 0 or self.band_max_width < 0:
			raise ValueError("band.min_width/max_width must be >= 0")

	@staticmethod
	def _find_contours_external_u8(img_u8: np.ndarray):
		res = cv2.findContours(img_u8, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
		return res[0] if len(res) == 2 else res[1]

	def _keep_largest_component_inplace(self, mask_u8: np.ndarray) -> None:
		if self._tmp_u8 is None or self._keep_mask is None or self._tmp_u8.shape != mask_u8.shape:
			self._tmp_u8 = np.empty_like(mask_u8)
			self._keep_mask = np.empty_like(mask_u8)
		np.copyto(self._tmp_u8, mask_u8)
		contours = self._find_contours_external_u8(self._tmp_u8)
		if not contours:
			mask_u8.fill(0)
			return
		if len(contours) == 1:
			return
		largest = max(contours, key=cv2.contourArea)
		self._keep_mask.fill(0)
		cv2.drawContours(self._keep_mask, [largest], -1, 255, thickness=-1)
		cv2.bitwise_and(mask_u8, self._keep_mask, dst=mask_u8)

	def _generate_rois(self, img_w: int, img_h: int) -> List[Tuple[int, int, int, int, int, int]]:
		rois: List[Tuple[int, int, int, int, int, int]] = []
		w = max(1, min(int(self.roi_w), int(img_w)))
		h = max(1, min(int(self.roi_h), int(img_h)))
		for r in range(self.rows):
			cols = int(self.cols_per_row[r])
			xL, yT = self.anchor_left[r]
			xR, _yT2 = self.anchor_right[r]
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

	def _draw_rect(self, img: np.ndarray, rect: Tuple[int, int, int, int], color: Tuple[int, int, int]) -> None:
		x, y, w, h = rect
		cv2.rectangle(img, (int(x), int(y)), (int(x + w - 1), int(y + h - 1)), color, thickness=self.line_width)

	def _draw_center(self, img: np.ndarray, cx: int, cy: int, color: Tuple[int, int, int]) -> None:
		r = max(1, int(self.line_width))
		cv2.circle(img, (int(cx), int(cy)), r, color, thickness=-1)

	def _draw_junction_line(self, img: np.ndarray, x: int, w: int, y: int, color: Tuple[int, int, int]) -> None:
		cv2.line(img, (int(x), int(y)), (int(x + w), int(y)), color, thickness=self.line_width)

	def detect(self, img: np.ndarray):
		try:
			return self._detect_impl(img)
		except Exception as e:
			return False, f"Error: {e}", None, "ERROR"

	def _detect_impl(self, img: np.ndarray):
		if img is None or getattr(img, "size", 0) == 0:
			return False, "Error: empty image", None, "ERROR"
		if img.ndim != 3 or img.shape[2] != 3:
			return False, f"Error: expected BGR image HxWx3, got shape={getattr(img, 'shape', None)}", None, "ERROR"

		img_h, img_w = img.shape[:2]
		if self._cached_rois is None or self._cached_shape != (img_w, img_h):
			self._cached_rois = self._generate_rois(img_w, img_h)
			self._cached_shape = (img_w, img_h)

		# Denoise (base for overlay) and HSV once per image.
		img_dn = cv2.GaussianBlur(img, (self.denoise_k, self.denoise_k), self.denoise_sigma)
		hsv = cv2.cvtColor(img_dn, cv2.COLOR_BGR2HSV)
		yellow_full = cv2.inRange(hsv, self.yellow_lower, self.yellow_upper)
		white_full = (hsv[:, :, 1] < self.white_s_max) & (hsv[:, :, 2] > self.white_v_min)

		overlay_img = img_dn.copy() if self.generate_overlay else None

		_fail_stage = {
			"ROD_NOT_FOUND": 0,
			"ROD_ANGLE_NG": 1,
			"ROD_BBOX_AREA_NG": 2,
			"JUNCTION_NOT_FOUND": 3,
			"HEIGHT_NG": 4,
			"BAND_NOT_FOUND": 5,
			"OFFSET_NG": 6,
		}
		first_fail: Optional[Tuple[str, int, int]] = None  # (code, row, col)
		first_fail_key: Optional[Tuple[int, int, int]] = None  # (row, col, stage)

		def consider_fail(code: str, rr: int, cc: int):
			nonlocal first_fail, first_fail_key
			key = (int(rr), int(cc), int(_fail_stage.get(code, 99)))
			if first_fail_key is None or key < first_fail_key:
				first_fail_key = key
				first_fail = (str(code), int(rr), int(cc))

		# Collect stage-D success slots for stage E.
		slots_d_ok: List[_SlotD] = []
		junction_by_row: List[List[int]] = [[], [], []]

		for r, c, x, y, w, h in self._cached_rois:
			roi_mask = yellow_full[y : y + h, x : x + w]
			core = roi_mask.copy()

			if self.ker_open is not None:
				core = cv2.morphologyEx(core, cv2.MORPH_OPEN, self.ker_open)
			if self.ker_close is not None:
				core = cv2.morphologyEx(core, cv2.MORPH_CLOSE, self.ker_close)
			if self.ker_erode is not None and self.erode_iter > 0:
				core = cv2.erode(core, self.ker_erode, iterations=self.erode_iter)

			if self.keep_largest:
				self._keep_largest_component_inplace(core)

			if cv2.countNonZero(core) == 0:
				if overlay_img is not None:
					self._draw_rect(overlay_img, (x, y, w, h), self.ng_bgr)
				consider_fail("ROD_NOT_FOUND", int(r), int(c))
				continue

			M = cv2.moments(core, binaryImage=True)
			if float(M.get("m00", 0.0)) <= 0:
				if overlay_img is not None:
					self._draw_rect(overlay_img, (x, y, w, h), self.ng_bgr)
				consider_fail("ROD_NOT_FOUND", int(r), int(c))
				continue

			cx = float(M["m10"] / M["m00"])
			cy = float(M["m01"] / M["m00"])
			Cx = int(round(x + cx))
			Cy = int(round(y + cy))

			theta = _theta_from_central_moments_deg(M)  # [0,180)
			angle_ok = abs(theta - 90.0) <= self.theta_tol_deg
			if overlay_img is not None:
				self._draw_center(overlay_img, Cx, Cy, self.ok_bgr if angle_ok else self.ng_bgr)
			if not angle_ok:
				consider_fail("ROD_ANGLE_NG", int(r), int(c))
				continue

			bx, by, bw, bh = cv2.boundingRect(core)
			if int(bw * bh) < self.min_bbox_area:
				if overlay_img is not None:
					self._draw_rect(overlay_img, (x + int(bx), y + int(by), int(bw), int(bh)), self.ng_bgr)
				consider_fail("ROD_BBOX_AREA_NG", int(r), int(c))
				continue

			row_cnt = np.count_nonzero(core, axis=1).astype(np.int32, copy=False)
			total = int(row_cnt.sum())
			if total <= 0:
				if overlay_img is not None:
					self._draw_rect(overlay_img, (x, y, w, h), self.ng_bgr)
				consider_fail("ROD_NOT_FOUND", int(r), int(c))
				continue
			cum = np.cumsum(row_cnt, dtype=np.int64)
			pct = max(0.0, min(100.0, float(self.rod_bottom_percentile))) / 100.0
			rank = int(math.floor(pct * (total - 1))) if total > 1 else 0
			y_rod_bottom = int(np.searchsorted(cum, rank + 1, side="left"))

			y0 = _clamp_int(y_rod_bottom - self.win_up, 0, h - 1)
			y1 = _clamp_int(y_rod_bottom + self.win_down, 0, h - 1)
			if y1 <= y0:
				y1 = min(h - 1, y0 + 1)

			cx_i = _clamp_int(cx, 0, w - 1)
			d = int(self.strip_offset_x)
			ws = int(self.strip_half_width)

			lx0 = _clamp_int(cx_i - d - ws, 0, w - 1)
			lx1 = _clamp_int(cx_i - d + ws, 0, w)
			if lx1 <= lx0:
				lx1 = min(w, lx0 + 1)
			rx0 = _clamp_int(cx_i + d - ws, 0, w - 1)
			rx1 = _clamp_int(cx_i + d + ws, 0, w)
			if rx1 <= rx0:
				rx1 = min(w, rx0 + 1)

			white_roi = white_full[y : y + h, x : x + w]
			win = white_roi[y0 : y1 + 1, :]
			n_rows = int(y1 - y0 + 1)
			l_strip = win[:, lx0:lx1]
			r_strip = win[:, rx0:rx1]
			l_mean = l_strip.mean(axis=1, dtype=np.float32) if l_strip.size else np.zeros(n_rows, dtype=np.float32)
			r_mean = r_strip.mean(axis=1, dtype=np.float32) if r_strip.size else np.zeros(n_rows, dtype=np.float32)
			score = (0.5 * (l_mean + r_mean)).astype(np.float32, copy=False)

			if self.split_kernel is not None and score.size > 1:
				score_s = np.convolve(score, self.split_kernel, mode="same").astype(np.float32, copy=False)
			else:
				score_s = score

			n = int(score_s.size)
			k0 = max(1, int(self.min_rows_above))
			k1 = n - max(1, int(self.min_rows_below))
			if n < 2 or k0 > k1:
				if overlay_img is not None:
					self._draw_rect(overlay_img, (x + lx0, y + y0, max(1, lx1 - lx0), max(1, y1 - y0 + 1)), self.ng_bgr)
					self._draw_rect(overlay_img, (x + rx0, y + y0, max(1, rx1 - rx0), max(1, y1 - y0 + 1)), self.ng_bgr)
				consider_fail("JUNCTION_NOT_FOUND", int(r), int(c))
				continue

			ps = np.empty(n + 1, dtype=np.float32)
			ps[0] = 0.0
			np.cumsum(score_s, out=ps[1:])
			ks = np.arange(k0, k1 + 1, dtype=np.int32)
			sum_above = ps[ks]
			sum_below = ps[n] - sum_above
			mu_above = sum_above / ks
			mu_below = sum_below / (n - ks)
			contrast = mu_below - mu_above
			idx = int(np.argmax(contrast))
			best_k = int(ks[idx])
			best_contrast = float(contrast[idx])
			if best_contrast < float(self.min_delta_white):
				if overlay_img is not None:
					self._draw_rect(overlay_img, (x + lx0, y + y0, max(1, lx1 - lx0), max(1, y1 - y0 + 1)), self.ng_bgr)
					self._draw_rect(overlay_img, (x + rx0, y + y0, max(1, rx1 - rx0), max(1, y1 - y0 + 1)), self.ng_bgr)
				consider_fail("JUNCTION_NOT_FOUND", int(r), int(c))
				continue

			y_junction = int(y + y0 + best_k)
			slots_d_ok.append(_SlotD(row=int(r), col=int(c), roi=(int(x), int(y), int(w), int(h)), x_center=float(x + cx), y_junction=y_junction))
			junction_by_row[int(r)].append(y_junction)

		# Stage E: row-wise height decision; draw junction line (green or red).
		baseline_by_row: List[Optional[float]] = [None, None, None]
		for r in range(3):
			ys = junction_by_row[r]
			baseline_by_row[r] = float(np.median(ys)) if ys else None

		slots_height_ok: List[_SlotD] = []
		for slot in slots_d_ok:
			base = baseline_by_row[slot.row]
			x, y, w, h = slot.roi
			if base is None:
				# No baseline: treat as junction-not-found.
				if overlay_img is not None:
					# Preserve center (already drawn) and mark D/E failure minimally by strips is not available here.
					# Use a red ROI rectangle as the minimal fallback marker.
					self._draw_rect(overlay_img, (x, y, w, h), self.ng_bgr)
				consider_fail("JUNCTION_NOT_FOUND", int(slot.row), int(slot.col))
				continue
			threshold = float(base) - float(self.height_delta_up)
			height_ok = float(slot.y_junction) >= threshold
			if overlay_img is not None:
				self._draw_junction_line(overlay_img, x, w, slot.y_junction, self.ok_bgr if height_ok else self.ng_bgr)
			if not height_ok:
				consider_fail("HEIGHT_NG", int(slot.row), int(slot.col))
				continue
			slots_height_ok.append(slot)

		# Stages F/G: band + offset; draw band point on success (green) or offset NG (red).
		for slot in slots_height_ok:
			x, y, w, h = slot.roi
			cols = int(self.cols_per_row[slot.row])

			y0f = _clamp_int(slot.y_junction + self.band_y0, y, y + h - 1)
			y1f = _clamp_int(slot.y_junction + self.band_y1, y, y + h - 1)
			if y1f <= y0f:
				y1f = min(y + h - 1, y0f + 1)
			xc = _clamp_int(slot.x_center, x, x + w - 1)
			x0f = _clamp_int(xc - self.band_half_width, x, x + w - 1)
			x1f = _clamp_int(xc + self.band_half_width, x, x + w - 1)
			if x1f <= x0f:
				x1f = min(x + w - 1, x0f + 1)

			if y1f <= y0f or x1f <= x0f:
				if overlay_img is not None:
					self._draw_rect(overlay_img, (x0f, y0f, max(1, x1f - x0f + 1), max(1, y1f - y0f + 1)), self.ng_bgr)
				consider_fail("BAND_NOT_FOUND", int(slot.row), int(slot.col))
				continue

			roi_hsv = hsv[y0f : y1f + 1, x0f : x1f + 1]
			Hc = roi_hsv[:, :, 0]
			Sc = roi_hsv[:, :, 1].astype(np.float32, copy=False)
			m = _h_in_range(Hc, self.band_h_low, self.band_h_high)
			s_masked = np.where(m, Sc, np.nan)
			s_col = np.nanmedian(s_masked, axis=0).astype(np.float32, copy=False)
			s_col = np.nan_to_num(s_col, nan=0.0)
			if self.band_kernel is not None and s_col.size > 1:
				s_col = np.convolve(s_col, self.band_kernel, mode="same").astype(np.float32, copy=False)
			b = s_col >= float(self.band_s_thr)
			run = _longest_true_run(b)
			if run is None:
				if overlay_img is not None:
					self._draw_rect(overlay_img, (x0f, y0f, max(1, x1f - x0f + 1), max(1, y1f - y0f + 1)), self.ng_bgr)
				consider_fail("BAND_NOT_FOUND", int(slot.row), int(slot.col))
				continue
			L, R = run
			width = int(R - L + 1)
			if width < int(self.band_min_width) or (int(self.band_max_width) > 0 and width > int(self.band_max_width)):
				if overlay_img is not None:
					self._draw_rect(overlay_img, (x0f, y0f, max(1, x1f - x0f + 1), max(1, y1f - y0f + 1)), self.ng_bgr)
				consider_fail("BAND_NOT_FOUND", int(slot.row), int(slot.col))
				continue

			xL = int(x0f + L)
			xR = int(x0f + R)
			x_band = float((xL + xR) / 2.0)
			y_mid = int(round((y0f + y1f) / 2.0))

			t = 0.0 if cols <= 1 else float(slot.col) / float(cols - 1)
			base_shift = (-float(self.offset_base)) + (2.0 * float(self.offset_base)) * t
			x_ref = float(slot.x_center) + base_shift
			offset_ok = (x_ref - float(self.offset_tol)) <= x_band <= (x_ref + float(self.offset_tol))

			if overlay_img is not None:
				self._draw_center(overlay_img, int(round(x_band)), int(y_mid), self.ok_bgr if offset_ok else self.ng_bgr)
			if not offset_ok:
				consider_fail("OFFSET_NG", int(slot.row), int(slot.col))
				continue

		ok = first_fail is None
		if ok:
			return True, "OK", overlay_img, "OK"
		if first_fail is not None:
			code, rr, cc = first_fail
			return False, f"NG: {code} r={rr} c={cc}", overlay_img, code
		return False, "NG: Unknown error", overlay_img, "UNKNOWN"


__all__ = ["WeigaoTrayDetector"]
