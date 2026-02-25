# -- coding: utf-8 --

import argparse
import logging
import os
import time

import cv2

from camera import build_camera_config, create_camera, ensure_dir
from core.config import ConfigError, load_config
from detect import create_detector
from core.runtime import build_runtime
from trigger import TriggerConfig, create_trigger


def parse_args():
    p = argparse.ArgumentParser(
        description="VisionRuntime capture service (config-driven)",
    )
    p.add_argument(
        "--config-dir", default="config", help="Directory containing main_*.yaml"
    )
    p.add_argument("--verbose", action="store_true", help="Debug log")
    p.add_argument(
        "--log-level", default="", help="Override log level (debug/info/warning/error)"
    )
    return p.parse_args()


def setup_logging(verbose: bool, log_level: str = ""):
    if verbose:
        level = logging.DEBUG
    else:
        level_map = {
            "debug": logging.DEBUG,
            "info": logging.INFO,
            "warning": logging.WARNING,
            "warn": logging.WARNING,
            "error": logging.ERROR,
            "critical": logging.CRITICAL,
        }
        level = level_map.get(str(log_level or "").strip().lower(), logging.INFO)
    # Use UTC for all %(asctime)s timestamps in logs.
    logging.Formatter.converter = time.gmtime
    logging.basicConfig(
        level=level, format="%(asctime)sZ [%(levelname)s] %(message)s", force=True
    )
    if not verbose:
        # Silence noisy third-party info logs (e.g., pymodbus internal "Server listening" messages).
        for name in ("pymodbus", "pymodbus.server", "pymodbus.server.async_io"):
            logging.getLogger(name).setLevel(logging.WARNING)


def main():
    args = parse_args()
    setup_logging(args.verbose, args.log_level)
    try:
        cfg = load_config(args.config_dir)
    except ConfigError as e:
        logging.error("Config load failed: %s", e)
        raise SystemExit(1)
    if int(getattr(cfg.runtime, "opencv_num_threads", 0)) > 0:
        cv2.setNumThreads(int(cfg.runtime.opencv_num_threads))
    # Config-driven log level (unless overridden by CLI).
    if not args.verbose and not args.log_level:
        setup_logging(args.verbose, getattr(cfg.runtime, "log_level", "info"))

    try:
        _validate_config(cfg)
        enable_preview = bool(getattr(cfg.detect, "enable_preview", True))
        output_pixel_format = _normalize_pixel_format(
            getattr(cfg.camera, "output_pixel_format", "bgr8")
        )
        detector = create_detector(
            cfg.detect.impl,
            cfg.detect_params or {},
            generate_overlay=bool(cfg.detect.generate_overlay) and enable_preview,
            input_pixel_format=output_pixel_format,
        )
    except (ConfigError, ValueError) as e:
        logging.error("Config invalid: %s", e)
        raise SystemExit(1) from e

    tcp_info = (
        f"{cfg.comm.tcp.host}:{cfg.comm.tcp.port}" if cfg.trigger.tcp.enabled else "off"
    )
    modbus_info = (
        f"{cfg.comm.modbus.host}:{cfg.comm.modbus.port}"
        if (cfg.output.enable_modbus or cfg.trigger.modbus.enabled)
        else "off"
    )
    logging.info(
        "Starting: camera=%s http=%s tcp=%s modbus=%s runtime=%s",
        cfg.camera.type,
        f"{cfg.comm.http.host}:{cfg.comm.http.port}"
        if cfg.output.enable_http
        else "off",
        tcp_info,
        modbus_info,
        f"{cfg.runtime.max_runtime_s}s" if cfg.runtime.max_runtime_s else "unlimited",
    )
    logging.info(
        "Config files: main=%s detect=%s",
        cfg.paths.get("main"),
        cfg.paths.get("detect"),
    )

    image_root = os.path.join(cfg.runtime.save_dir, "images")
    if cfg.camera.save_images:
        ensure_dir(image_root)

    cam_cfg = build_camera_config(
        cfg.camera,
        save_dir=image_root,
        ext=_normalize_ext(cfg.camera.ext),
        output_pixel_format=output_pixel_format,
    )
    trigger_cfg = TriggerConfig(
        host=cfg.comm.tcp.host,
        port=cfg.comm.tcp.port,
        word=_ensure_bytes(cfg.trigger.tcp.word),
        global_min_interval_ms=cfg.trigger.global_min_interval_ms,
        high_priority_cooldown_ms=cfg.trigger.high_priority_cooldown_ms,
        high_priority_sources=cfg.trigger.high_priority_sources,
        low_priority_sources=cfg.trigger.low_priority_sources,
        ip_whitelist=cfg.trigger.ip_whitelist,
    )

    camera = create_camera(cfg.camera.type, cam_cfg)
    try:
        with camera.session():
            modbus_trigger_enabled = bool(cfg.trigger.modbus.enabled)
            modbus_io_needed = bool(cfg.output.enable_modbus or modbus_trigger_enabled)
            runtime = build_runtime(
                camera,
                save_dir=cfg.runtime.save_dir,
                history_size=cfg.runtime.history_size,
                debounce_ms=cfg.runtime.debounce_ms,
                http_host=cfg.comm.http.host,
                http_port=cfg.comm.http.port,
                enable_http=cfg.output.enable_http,
                max_pending_triggers=cfg.runtime.max_pending_triggers,
                enable_modbus=cfg.output.enable_modbus,
                enable_modbus_io=modbus_io_needed,
                modbus_host=cfg.comm.modbus.host,
                modbus_port=cfg.comm.modbus.port,
                coil_offset=cfg.comm.modbus.coil_offset,
                di_offset=cfg.comm.modbus.di_offset,
                ir_offset=cfg.comm.modbus.ir_offset,
                modbus_heartbeat_ms=cfg.comm.modbus.heartbeat_ms,
                write_csv=cfg.output.write_csv,
                detector=detector,
                detect_timeout_ms=cfg.detect.timeout_ms,
                enable_preview=enable_preview,
                trigger_cfg=trigger_cfg,
            )

            triggers = []
            if modbus_trigger_enabled:
                modbus_io = getattr(runtime.app_context, "modbus_io", None)
                if modbus_io is None:
                    logging.error(
                        "Modbus trigger enabled but ModbusIO is not available"
                    )
                else:

                    def on_modbus_trigger(_src):
                        return runtime.app_context.trigger_gateway.report_raw_trigger(
                            "MODBUS"
                        )

                    triggers.append(
                        create_trigger(
                            "modbus",
                            trigger_cfg,
                            on_modbus_trigger,
                            modbus_io=modbus_io,
                            poll_ms=cfg.trigger.modbus.poll_ms,
                            on_reset=runtime.reset_system,
                        )
                    )
                    logging.info("Modbus trigger enabled")

            if cfg.trigger.tcp.enabled:

                def on_trigger(addr):
                    runtime.app_context.trigger_gateway.report_raw_trigger(
                        "TCP", remote_ip=addr
                    )

                triggers.append(create_trigger("tcp", trigger_cfg, on_trigger))

            if not triggers:
                logging.info("All triggers disabled by config!")

            runtime.start(triggers=triggers)
            runtime.run(
                runtime_limit_s=cfg.runtime.max_runtime_s
                if cfg.runtime.max_runtime_s > 0
                else None
            )
        logging.info("Done")
    except KeyboardInterrupt:
        logging.info("Service STOPPED by user (Ctrl+C)")
    except Exception:
        logging.exception("Error")
        raise


def _normalize_ext(ext: str) -> str:
    if not ext:
        return ".bmp"
    return ext if ext.startswith(".") else f".{ext}"


def _normalize_pixel_format(value: str) -> str:
    fmt = str(value or "").strip().lower()
    return fmt or "bgr8"


def _ensure_bytes(word) -> bytes:
    if isinstance(word, bytes):
        return word
    if isinstance(word, str):
        return word.encode("utf-8")
    return str(word).encode("utf-8")


def _validate_config(cfg):
    # runtime
    _require_int(
        "runtime.max_pending_triggers", cfg.runtime.max_pending_triggers, min_v=1
    )
    _require_int("runtime.history_size", cfg.runtime.history_size, min_v=1)
    _require_float("runtime.debounce_ms", cfg.runtime.debounce_ms, min_v=0.0)
    _require_float("runtime.max_runtime_s", cfg.runtime.max_runtime_s, min_v=0.0)
    _require_int("runtime.opencv_num_threads", cfg.runtime.opencv_num_threads, min_v=0)

    # camera
    _require_int("camera.device_index", cfg.camera.device_index, min_v=0)
    _require_int("camera.grab_timeout_ms", cfg.camera.grab_timeout_ms, min_v=1)
    _require_int("camera.max_retry_per_frame", cfg.camera.max_retry_per_frame, min_v=1)
    _require_int("camera.width", cfg.camera.width, min_v=0)
    _require_int("camera.height", cfg.camera.height, min_v=0)
    _require_int("camera.exposure_us", cfg.camera.exposure_us, min_v=0)
    _require_float("camera.analogue_gain", cfg.camera.analogue_gain, min_v=0.0)
    _require_int("camera.frame_duration_us", cfg.camera.frame_duration_us, min_v=0)
    _require_int("camera.settle_ms", cfg.camera.settle_ms, min_v=0)

    # trigger
    _require_float(
        "trigger.global_min_interval_ms", cfg.trigger.global_min_interval_ms, min_v=0.0
    )
    _require_float(
        "trigger.high_priority_cooldown_ms",
        cfg.trigger.high_priority_cooldown_ms,
        min_v=0.0,
    )
    _require_int("trigger.modbus.poll_ms", cfg.trigger.modbus.poll_ms, min_v=1)

    # comm
    _require_port("comm.http.port", cfg.comm.http.port)
    _require_port("comm.tcp.port", cfg.comm.tcp.port)
    _require_port("comm.modbus.port", cfg.comm.modbus.port)
    _require_int("comm.modbus.coil_offset", cfg.comm.modbus.coil_offset, min_v=0)
    _require_int("comm.modbus.di_offset", cfg.comm.modbus.di_offset, min_v=0)
    _require_int("comm.modbus.ir_offset", cfg.comm.modbus.ir_offset, min_v=0)
    _require_int("comm.modbus.heartbeat_ms", cfg.comm.modbus.heartbeat_ms, min_v=1)

    # detect
    _require_int("detect.timeout_ms", cfg.detect.timeout_ms, min_v=0)


def _require_int(
    name: str, value, *, min_v: int | None = None, max_v: int | None = None
):
    try:
        iv = int(value)
    except Exception as e:
        raise ConfigError(f"{name} must be an integer") from e
    if min_v is not None and iv < min_v:
        op = ">=" if min_v != 1 else ">"
        threshold = min_v if min_v != 1 else 0
        raise ConfigError(f"{name} must be {op} {threshold}")
    if max_v is not None and iv > max_v:
        raise ConfigError(f"{name} must be <= {max_v}")
    return iv


def _require_float(
    name: str, value, *, min_v: float | None = None, max_v: float | None = None
):
    try:
        fv = float(value)
    except Exception as e:
        raise ConfigError(f"{name} must be a number") from e
    if min_v is not None and fv < min_v:
        raise ConfigError(f"{name} must be >= {min_v:g}")
    if max_v is not None and fv > max_v:
        raise ConfigError(f"{name} must be <= {max_v:g}")
    return fv


def _require_port(name: str, value):
    _require_int(name, value, min_v=1, max_v=65535)


if __name__ == "__main__":
    main()
