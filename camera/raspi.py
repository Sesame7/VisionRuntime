# -- coding: utf-8 --

import time
from contextlib import contextmanager
from datetime import datetime, timezone

import cv2
import numpy as np
from picamera2 import Picamera2, Preview

from camera.base import BaseCamera, CameraConfig, CaptureResult, register_camera


def _to_bgr(arr: np.ndarray) -> np.ndarray:
	if arr.ndim == 3 and arr.shape[2] == 4:
		return arr[:, :, :3]
	return arr


def _to_mono(arr: np.ndarray) -> np.ndarray:
	if arr.ndim == 2:
		return arr
	if arr.ndim == 3 and arr.shape[2] == 1:
		return arr[:, :, 0]
	return cv2.cvtColor(arr, cv2.COLOR_BGR2GRAY)


@register_camera("raspi")
class RaspiCamera(BaseCamera):
	def __init__(self, cfg: CameraConfig):
		super().__init__(cfg)
		self._cam: Picamera2 | None = None

	def _build_controls(self) -> dict:
		ctrls: dict = {}
		if self.cfg.ae_enable is not None:
			ctrls["AeEnable"] = bool(self.cfg.ae_enable)
		if self.cfg.awb_enable is not None:
			ctrls["AwbEnable"] = bool(self.cfg.awb_enable)
		if self.cfg.exposure_us and int(self.cfg.exposure_us) > 0:
			ctrls["ExposureTime"] = int(self.cfg.exposure_us)
		if self.cfg.analogue_gain and float(self.cfg.analogue_gain) > 0:
			ctrls["AnalogueGain"] = float(self.cfg.analogue_gain)
		if self.cfg.frame_duration_us and int(self.cfg.frame_duration_us) > 0:
			val = int(self.cfg.frame_duration_us)
			ctrls["FrameDurationLimits"] = (val, val)
		return ctrls

	def _apply_controls(self):
		ctrls = self._build_controls()
		if not ctrls:
			return
		cam = self._cam
		if cam is None:
			return
		cam.set_controls(ctrls)
		settle_ms = max(int(self.cfg.settle_ms or 0), 0)
		if settle_ms > 0:
			time.sleep(settle_ms / 1000.0)

	def capture_once(self, idx):
		if not self._cam:
			return CaptureResult(
				success=False,
				trigger_seq=idx,
				device_id="raspi",
				error="camera_not_started",
			)
		start = time.perf_counter()
		arr = self._cam.capture_array()
		grab_ms = (time.perf_counter() - start) * 1000
		bgr = _to_bgr(arr).astype(np.uint8, copy=False)
		if str(self.cfg.output_pixel_format).lower() == "mono8":
			bgr = _to_mono(bgr).astype(np.uint8, copy=False)
		return CaptureResult(
			success=True,
			trigger_seq=idx,
			source="",
			device_id="raspi",
			path=None,
			image=bgr,
			timings={"grab_ms": grab_ms},
			captured_at=datetime.now(timezone.utc),
		)

	@contextmanager
	def session(self):
		self._cam = Picamera2()
		main_cfg: dict[str, object] = {"format": "XBGR8888"}
		if self.cfg.width and self.cfg.height:
			main_cfg["size"] = (int(self.cfg.width), int(self.cfg.height))
		if self.cfg.use_still:
			cfg = self._cam.create_still_configuration(main=main_cfg)
		else:
			cfg = self._cam.create_preview_configuration(main=main_cfg)
		self._cam.configure(cfg)
		self._cam.start_preview(Preview.NULL)
		self._cam.start()
		self._apply_controls()
		try:
			yield
		finally:
			try:
				self._cam.stop()
			finally:
				self._cam.close()
				self._cam = None


__all__ = ["RaspiCamera"]
