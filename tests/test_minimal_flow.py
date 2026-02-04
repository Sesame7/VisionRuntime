import os
import time
import unittest

from camera import CameraConfig, create_camera
from core.config import load_config
from core.runtime import build_runtime
from detect import create_detector
from trigger import TriggerConfig

REPO_ROOT = os.path.abspath(os.path.join(os.path.dirname(__file__), os.pardir))
DETECT_CONFIG_PATH = os.path.join(REPO_ROOT, "config", "detect_overexposure.yaml")
TEST_CONFIG_DIR = os.path.join(REPO_ROOT, "config", "tests")
TEST_IMAGE_DIR = os.path.join(REPO_ROOT, "data", "test")


def _wait_for_records(output_mgr, count: int, timeout_s: float = 2.0):
    start = time.perf_counter()
    while (time.perf_counter() - start) < timeout_s:
        records = output_mgr.latest_records
        if len(records) >= count:
            return records
        time.sleep(0.05)
    raise AssertionError(f"timeout waiting for {count} records")


class TestMinimalFlow(unittest.TestCase):
    def test_trigger_mock_camera_overexposure(self):
        if not os.path.exists(DETECT_CONFIG_PATH):
            raise AssertionError(f"missing detect config: {DETECT_CONFIG_PATH}")
        if not os.path.isdir(TEST_CONFIG_DIR):
            raise AssertionError(f"missing test config dir: {TEST_CONFIG_DIR}")
        if not os.path.isfile(os.path.join(TEST_IMAGE_DIR, "ok.png")):
            raise AssertionError(f"missing ok image: {TEST_IMAGE_DIR}")
        if not os.path.isfile(os.path.join(TEST_IMAGE_DIR, "ng.png")):
            raise AssertionError(f"missing ng image: {TEST_IMAGE_DIR}")

        cfg = load_config(TEST_CONFIG_DIR)
        cfg.runtime.save_dir = TEST_IMAGE_DIR
        cfg.camera.image_dir = TEST_IMAGE_DIR
        camera_cfg = CameraConfig(
            save_dir=cfg.runtime.save_dir,
            ext=cfg.camera.ext,
            device_index=cfg.camera.device_index,
            timeout_ms=cfg.camera.grab_timeout_ms,
            max_retry_per_frame=cfg.camera.max_retry_per_frame,
            save_images=cfg.camera.save_images,
            output_pixel_format=cfg.camera.output_pixel_format,
            width=cfg.camera.width,
            height=cfg.camera.height,
            ae_enable=cfg.camera.ae_enable,
            awb_enable=cfg.camera.awb_enable,
            exposure_us=cfg.camera.exposure_us,
            analogue_gain=cfg.camera.analogue_gain,
            frame_duration_us=cfg.camera.frame_duration_us,
            settle_ms=cfg.camera.settle_ms,
            use_still=cfg.camera.use_still,
            image_dir=cfg.camera.image_dir,
            order=cfg.camera.order,
            end_mode=cfg.camera.end_mode,
        )
        camera = create_camera(cfg.camera.type, camera_cfg)
        detector = create_detector(
            cfg.detect.impl,
            cfg.detect_params or {},
            generate_overlay=False,
            input_pixel_format=cfg.camera.output_pixel_format,
        )
        runtime = build_runtime(
            camera,
            save_dir=cfg.runtime.save_dir,
            history_size=cfg.runtime.history_size,
            debounce_ms=cfg.runtime.debounce_ms,
            enable_http=cfg.output.enable_http,
            max_pending_triggers=5,
            enable_modbus=cfg.output.enable_modbus,
            write_csv=cfg.output.write_csv,
            detector=detector,
            detect_timeout_ms=cfg.detect.timeout_ms,
            enable_preview=cfg.detect.enable_preview,
            trigger_cfg=TriggerConfig(),
        )

        with camera.session():
            runtime.start()
            try:
                ok1 = runtime.app_context.trigger_gateway.report_raw_trigger("TEST")
                ok2 = runtime.app_context.trigger_gateway.report_raw_trigger("TEST")
                self.assertTrue(ok1 and ok2)
                records = _wait_for_records(runtime.output_mgr, 2)
            finally:
                runtime.stop()

        records = sorted(records, key=lambda r: r.trigger_seq)
        self.assertEqual(records[0].result, "OK")
        self.assertEqual(records[1].result, "NG")
