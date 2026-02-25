import unittest
from types import SimpleNamespace

from core.config import ConfigError
from main import _validate_config


def _make_cfg():
    return SimpleNamespace(
        runtime=SimpleNamespace(
            max_pending_triggers=10,
            history_size=10,
            debounce_ms=10.0,
            max_runtime_s=0.0,
            opencv_num_threads=0,
        ),
        camera=SimpleNamespace(
            device_index=0,
            grab_timeout_ms=2000,
            max_retry_per_frame=3,
            width=0,
            height=0,
            exposure_us=0,
            analogue_gain=0.0,
            frame_duration_us=0,
            settle_ms=200,
        ),
        trigger=SimpleNamespace(
            global_min_interval_ms=0.0,
            high_priority_cooldown_ms=0.0,
            modbus=SimpleNamespace(poll_ms=20),
        ),
        comm=SimpleNamespace(
            http=SimpleNamespace(port=8000),
            tcp=SimpleNamespace(port=9000),
            modbus=SimpleNamespace(
                port=1502,
                coil_offset=800,
                di_offset=800,
                ir_offset=50,
                heartbeat_ms=1000,
            ),
        ),
        detect=SimpleNamespace(timeout_ms=2000),
    )


class TestMainConfigValidation(unittest.TestCase):
    def test_valid_config_passes(self):
        _validate_config(_make_cfg())

    def test_invalid_values_raise_config_error(self):
        cases = [
            ("detect.timeout_ms", "detect", {"timeout_ms": -1}),
            ("camera.max_retry_per_frame", "camera", {"max_retry_per_frame": 0}),
            ("camera.grab_timeout_ms", "camera", {"grab_timeout_ms": 0}),
            ("comm.http.port", "comm.http", {"port": 70000}),
            ("comm.modbus.coil_offset", "comm.modbus", {"coil_offset": -1}),
        ]
        for expected_name, target, patch in cases:
            with self.subTest(field=expected_name):
                cfg = _make_cfg()
                obj = cfg
                for part in target.split("."):
                    obj = getattr(obj, part)
                for k, v in patch.items():
                    setattr(obj, k, v)
                with self.assertRaises(ConfigError) as cm:
                    _validate_config(cfg)
                self.assertIn(expected_name, str(cm.exception))


if __name__ == "__main__":
    unittest.main()
