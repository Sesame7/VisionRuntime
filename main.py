# -- coding: utf-8 --

import argparse
import logging
import time

from camera import create_camera_from_loaded_config
from core.config import load_config, validate_config
from core.runtime import build_runtime_from_loaded_config
from detect import create_detector_from_loaded_config
from trigger import (
    build_trigger_config_from_loaded_config,
    create_trigger,
)


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


def build_app(cfg):
    validate_config(cfg)
    _log_startup(cfg)
    camera = create_camera_from_loaded_config(cfg)
    detector = create_detector_from_loaded_config(
        cfg,
        input_pixel_format=camera.cfg.output_pixel_format,
    )
    trigger_cfg = build_trigger_config_from_loaded_config(cfg)
    runtime = build_runtime_from_loaded_config(
        camera,
        cfg,
        detector=detector,
        trigger_cfg=trigger_cfg,
    )
    triggers = _build_runtime_triggers(cfg, runtime, trigger_cfg)
    runtime_limit_s = _runtime_limit_s(cfg)
    return runtime, triggers, runtime_limit_s


def main():
    args = parse_args()
    setup_logging(args.verbose, args.log_level)
    cfg = load_config(args.config_dir)
    # Config-driven log level (unless overridden by CLI).
    if not args.verbose and not args.log_level:
        setup_logging(args.verbose, cfg.runtime.log_level)

    runtime = None

    try:
        runtime, triggers, runtime_limit_s = build_app(cfg)
        runtime.start(triggers=triggers)
        runtime.run(runtime_limit_s=runtime_limit_s)
        logging.info("Done")
    except KeyboardInterrupt:
        if runtime is not None:
            try:
                runtime.stop()
            except Exception:
                logging.exception("Runtime stop failed during Ctrl+C handling")
        logging.info("Service STOPPED by user (Ctrl+C)")


def _log_startup(cfg) -> None:
    tcp_info = (
        f"{cfg.comm.tcp.host}:{cfg.comm.tcp.port}" if cfg.trigger.tcp.enabled else "off"
    )
    modbus_info = (
        f"{cfg.comm.modbus.host}:{cfg.comm.modbus.port}"
        if (cfg.output.modbus.enabled or cfg.trigger.modbus.enabled)
        else "off"
    )
    logging.info(
        "Starting: camera=%s http=%s tcp=%s modbus=%s runtime=%s",
        cfg.camera.type,
        f"{cfg.comm.http.host}:{cfg.comm.http.port}"
        if cfg.output.hmi.enabled
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


def _runtime_limit_s(cfg) -> float | None:
    limit = float(cfg.runtime.max_runtime_s or 0)
    return limit if limit > 0 else None


def _build_runtime_triggers(cfg, runtime, trigger_cfg):
    triggers = []
    if cfg.trigger.modbus.enabled:
        modbus_io = runtime.app_context.modbus_io
        if modbus_io is None:
            raise RuntimeError("Modbus trigger enabled but ModbusIO is not available")
        else:

            def on_modbus_trigger(_src):
                return runtime.app_context.trigger_gateway.report_raw_trigger("MODBUS")

            triggers.append(
                create_trigger(
                    "modbus",
                    trigger_cfg,
                    on_modbus_trigger,
                    modbus_io=modbus_io,
                    poll_ms=cfg.trigger.modbus.poll_ms,
                    on_reset=runtime.reset_system,
                    loop_runner=runtime.loop_runner,
                )
            )
            logging.info("Modbus trigger enabled")

    if cfg.trigger.tcp.enabled:

        def on_tcp_trigger(addr):
            runtime.app_context.trigger_gateway.report_raw_trigger(
                "TCP", remote_ip=addr
            )

        triggers.append(
            create_trigger(
                "tcp",
                trigger_cfg,
                on_tcp_trigger,
                loop_runner=runtime.loop_runner,
            )
        )

    if not triggers:
        logging.info("All triggers disabled by config!")
    return triggers


if __name__ == "__main__":
    main()
