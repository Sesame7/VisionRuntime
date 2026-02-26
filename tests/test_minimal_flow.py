import os
import time
import unittest
from datetime import datetime, timezone

import numpy as np

from camera import build_camera_config, create_camera
from core.config import load_config
from core.runtime import RuntimeBuildConfig, build_runtime
from core.worker import AcqTask, DetectQueueManager
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


def _assert_test_assets_present():
    if not os.path.exists(DETECT_CONFIG_PATH):
        raise AssertionError(f"missing detect config: {DETECT_CONFIG_PATH}")
    if not os.path.isdir(TEST_CONFIG_DIR):
        raise AssertionError(f"missing test config dir: {TEST_CONFIG_DIR}")
    if not os.path.isfile(os.path.join(TEST_IMAGE_DIR, "ok.png")):
        raise AssertionError(f"missing ok image: {TEST_IMAGE_DIR}")
    if not os.path.isfile(os.path.join(TEST_IMAGE_DIR, "ng.png")):
        raise AssertionError(f"missing ng image: {TEST_IMAGE_DIR}")


def _make_acq_task(frame_id: int, now: datetime) -> AcqTask:
    return AcqTask(
        frame_id=frame_id,
        triggered_at=now,
        source="TEST",
        device_id="mock",
        t0=time.perf_counter(),
        captured_at=now,
        image=np.zeros((2, 2, 3), dtype=np.uint8),
    )


def _build_test_runtime():
    cfg = load_config(TEST_CONFIG_DIR)
    cfg.runtime.save_dir = TEST_IMAGE_DIR
    cfg.trigger.debounce_ms = 0.0
    cfg.camera.image_dir = TEST_IMAGE_DIR
    camera_cfg = build_camera_config(cfg.camera, save_dir=cfg.runtime.save_dir)
    camera = create_camera(cfg.camera.type, camera_cfg)
    detector = create_detector(
        cfg.detect.impl,
        cfg.detect_params or {},
        generate_overlay=False,
        input_pixel_format=cfg.camera.capture_output_format,
    )
    runtime = build_runtime(
        camera,
        config=RuntimeBuildConfig(
            save_dir=cfg.runtime.save_dir,
            history_size=cfg.output.hmi.history_size,
            debounce_ms=cfg.trigger.debounce_ms,
            enable_http=cfg.output.hmi.enabled,
            detect_queue_capacity=5,
            enable_modbus_trigger=cfg.trigger.modbus.enabled,
            enable_modbus_output=cfg.output.modbus.enabled,
            write_csv=cfg.output.write_csv,
            detect_timeout_ms=cfg.detect.timeout_ms,
            preview_enabled=cfg.detect.preview_enabled,
        ),
        detector=detector,
        trigger_cfg=TriggerConfig(),
    )
    return runtime


class TestMinimalFlow(unittest.TestCase):
    def test_detect_queue_overflow_keeps_newest_task(self):
        dropped_ids = []
        now = datetime.now(timezone.utc)

        def _sink(rec, _overlay):
            dropped_ids.append(int(rec.trigger_seq))

        queue_mgr = DetectQueueManager(maxsize=1, result_sink=_sink)

        queue_mgr.enqueue(_make_acq_task(1, now))
        queue_mgr.enqueue(_make_acq_task(2, now))

        self.assertEqual(queue_mgr.queue.qsize(), 1)
        remaining = queue_mgr.queue.get_nowait()
        self.assertEqual(remaining.frame_id, 2)
        self.assertEqual(dropped_ids, [1])

    def test_trigger_mock_camera_overexposure(self):
        _assert_test_assets_present()
        runtime = _build_test_runtime()

        runtime.start()
        try:
            time.sleep(0.05)
            ok1 = runtime.app_context.trigger_gateway.report_raw_trigger("TEST")
            ok2 = runtime.app_context.trigger_gateway.report_raw_trigger("TEST")
            self.assertTrue(ok1 and ok2)
            records = _wait_for_records(runtime.output_mgr, 2)
        finally:
            runtime.stop()

        records = sorted(records, key=lambda r: r.trigger_seq)
        self.assertEqual(records[0].result, "OK")
        self.assertEqual(records[1].result, "NG")
