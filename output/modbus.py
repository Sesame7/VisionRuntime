# -- coding: utf-8 --

from collections.abc import Callable, Mapping
from datetime import datetime, timezone
from typing import Any

from core.contracts import OutputRecord
from utils.modbus.modbus_server_io import ModbusIO

_StatsProvider = Callable[[], Mapping[str, Any] | None]


class ModbusIoLifecycleChannel:
    """Own ModbusIO service lifecycle independently of result publishing."""

    def __init__(self, modbus_io: ModbusIO):
        self.io = modbus_io

    def start(self):
        self.io.start()

    def stop(self):
        self.io.stop()

    def publish(self, rec: OutputRecord, overlay: tuple[bytes, str] | None):
        _ = rec, overlay
        return None

    def publish_heartbeat(self, ts: float | None = None):
        _ = ts
        return None

    def raise_if_failed(self):
        return None


class ModbusOutput:
    def __init__(
        self, modbus_io: ModbusIO, stats_provider: _StatsProvider | None = None
    ):
        self.io = modbus_io
        self._stats_provider = stats_provider

    def start(self):
        # ModbusIO lifecycle is owned by ModbusIoLifecycleChannel.
        return None

    def stop(self):
        # ModbusIO lifecycle is owned by ModbusIoLifecycleChannel.
        return None

    def publish(self, rec: OutputRecord, overlay: tuple[bytes, str] | None):
        _ = overlay
        trig_time = rec.triggered_at or datetime.now(timezone.utc)
        seq = int(rec.trigger_seq or 0) & 0xFFFF
        result_code, error_code = _map_result_codes(rec)
        cycle_ms = _saturate_u16(rec.duration_ms)
        ok, ng, err = _map_result_bits(rec)
        stats = self._read_stats()
        self.io.write_result(
            trig_time=trig_time,
            seq=seq,
            result_code=result_code,
            error_code=error_code,
            cycle_ms=cycle_ms,
            ok=ok,
            ng=ng,
            err=err,
            total_count=_saturate_u16(stats.get("total")),
            ok_count=_saturate_u16(stats.get("ok")),
            ng_count=_saturate_u16(stats.get("ng")),
            err_count=_saturate_u16(stats.get("error")),
        )
        return None

    def publish_heartbeat(self, ts: float | None = None):
        # ModbusIO handles its own heartbeat loop.
        _ = ts
        return None

    def raise_if_failed(self):
        return None

    def _read_stats(self) -> dict[str, int]:
        provider = self._stats_provider
        if not callable(provider):
            return {"total": 0, "ok": 0, "ng": 0, "error": 0}
        try:
            raw_payload = provider()
        except Exception:
            return {"total": 0, "ok": 0, "ng": 0, "error": 0}
        if not isinstance(raw_payload, Mapping):
            return {"total": 0, "ok": 0, "ng": 0, "error": 0}
        payload = raw_payload
        return {
            "total": int(payload.get("total", 0) or 0),
            "ok": int(payload.get("ok", 0) or 0),
            "ng": int(payload.get("ng", 0) or 0),
            "error": int(payload.get("error", 0) or 0),
        }


def _saturate_u16(value: float | int | None) -> int:
    if value is None:
        return 0
    v = int(round(float(value)))
    if v < 0:
        return 0
    return 0xFFFF if v > 0xFFFF else v


def _map_result_codes(rec: OutputRecord) -> tuple[int, int]:
    result = (rec.result or "").strip().upper()
    result_code = 3  # ERROR
    error_code = 2  # Generic processing error default
    if result == "OK":
        return 1, 0
    if result == "NG":
        return 2, 0
    if result == "TIMEOUT":
        return 3, 1
    # ERROR or others
    rc = (rec.result_code or "").strip().upper()
    if rc == "TIMEOUT":
        error_code = 1
    elif rc == "CAMERA_ERROR":
        error_code = 3
    elif rc == "QUEUE_OVERFLOW":
        error_code = 4
    elif rc == "ERROR":
        error_code = 2
    elif rc == "":
        error_code = 2
    else:
        # Unknown error code from impl; map to generic detect exception.
        error_code = 2
    return result_code, error_code


def _map_result_bits(rec: OutputRecord) -> tuple[int, int, int]:
    result = (rec.result or "").strip().upper()
    if result == "OK":
        return 1, 0, 0
    if result == "NG":
        return 0, 1, 0
    return 0, 1, 1


__all__ = ["ModbusIoLifecycleChannel", "ModbusOutput"]
