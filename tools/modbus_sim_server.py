# -- coding: utf-8 --
"""Modbus TCP simulator for the v2.2 point table (coils + discrete + input regs)."""

import argparse
import asyncio
import logging
from collections.abc import Iterable
from datetime import datetime, timezone
from threading import Lock
from typing import List, Tuple

from core.pymodbus_compat import (
    ModbusDeviceContext,
    ModbusDeviceContextType,
    ModbusSequentialDataBlock,
    ModbusTcpServer,
    build_server_context,
    is_modbus_exception,
)

LOG = logging.getLogger("modbus.sim")

RESULT_CODE_MAP = {
    "OK": 1,
    "NG": 2,
    "ERROR": 3,
}

ERROR_CODE_MAP = {
    "OK": 0,
    "NG": 0,
    "TIMEOUT": 1,
    "ERROR": 2,
    "CAMERA_ERROR": 3,
    "QUEUE_OVERFLOW": 4,
}

RESULT_CHOICES = ("OK", "NG", "TIMEOUT", "ERROR", "CAMERA_ERROR", "QUEUE_OVERFLOW")


class SimState:
    def __init__(
        self,
        device_ctx: ModbusDeviceContextType,
        coil_addr: int,
        di_addr: int,
        ir_addr: int,
    ):
        self._lock = Lock()
        self._ctx = device_ctx
        self._coil_addr = int(coil_addr)
        self._di_addr = int(di_addr)
        self._ir_addr = int(ir_addr)
        self._last_cmd_trig = 0
        self._last_cmd_reset = 0
        self._seq = 0

    def next_seq(self) -> int:
        with self._lock:
            self._seq += 1
            if self._seq > 0xFFFF:
                self._seq = 1
            return self._seq

    def toggle_di(self, idx: int) -> None:
        addr = self._di_addr + int(idx)
        cur = self._ctx.getValues(2, addr, 1)
        cur_vals = _require_values(cur, 1, "discrete_inputs")
        value = 0 if (cur_vals and cur_vals[0]) else 1
        self._ctx.setValues(2, addr, [value])

    def set_result_bits(self, ok: bool, ng: bool, err: bool) -> None:
        values = [1 if ok else 0, 1 if ng else 0, 1 if err else 0]
        self._ctx.setValues(2, self._di_addr + 3, values)

    def set_result_regs(
        self,
        trig_time: datetime,
        seq: int,
        result_code: int,
        error_code: int,
        cycle_ms: int,
    ) -> None:
        values = [
            int(trig_time.year),
            int(trig_time.month),
            int(trig_time.day),
            int(trig_time.hour),
            int(trig_time.minute),
            int(trig_time.second),
            int(seq) & 0xFFFF,
            int(result_code) & 0xFFFF,
            int(error_code) & 0xFFFF,
            int(cycle_ms) & 0xFFFF,
        ]
        self._ctx.setValues(4, self._ir_addr, values)

    def read_cmd_snapshot(self) -> Tuple[int, int]:
        values = self._ctx.getValues(1, self._coil_addr, 2)
        bits = [1 if v else 0 for v in _require_values(values, 2, "coils")]
        return bits[0], bits[1]

    def update_cmd_latches(self, trig_val: int, reset_val: int) -> None:
        with self._lock:
            self._last_cmd_trig = trig_val
            self._last_cmd_reset = reset_val

    def read_cmd_latches(self) -> Tuple[int, int]:
        with self._lock:
            return self._last_cmd_trig, self._last_cmd_reset


def _require_values(
    values: List[int] | List[bool] | object, count: int, label: str
) -> List[int]:
    if is_modbus_exception(values):
        raise RuntimeError(f"Sim read failed for {label}: {values!r}")
    if not isinstance(values, Iterable):
        raise RuntimeError(f"Sim read failed for {label}: non-iterable {values!r}")
    vals = list(values)
    if len(vals) < count:
        return vals + [0] * (count - len(vals))
    return vals


def _map_codes(result: str) -> Tuple[int, int]:
    result_key = (result or "").strip().upper()
    result_code = RESULT_CODE_MAP.get(result_key, 3)
    error_code = ERROR_CODE_MAP.get(result_key, 2)
    return result_code, error_code


def _as_cycle_ms(value: int | None, default_ms: int) -> int:
    if value is None:
        value = default_ms
    if value < 0:
        value = 0
    return min(int(value), 0xFFFF)


async def _heartbeat_loop(state: SimState, interval_ms: int):
    interval_s = max(interval_ms, 50) / 1000.0
    while True:
        await asyncio.sleep(interval_s)
        state.toggle_di(0)


async def _trigger_loop(state: SimState, result: str, process_ms: int, poll_ms: int):
    poll_s = max(poll_ms, 10) / 1000.0
    while True:
        await asyncio.sleep(poll_s)
        cmd_trig, cmd_reset = state.read_cmd_snapshot()
        last_trig, last_reset = state.read_cmd_latches()
        if cmd_trig != last_trig:
            state.update_cmd_latches(cmd_trig, last_reset)
            seq = state.next_seq()
            trig_time = datetime.now(timezone.utc)
            state.toggle_di(1)  # ST_ACCEPT_TOGGLE
            asyncio.create_task(_emit_result(state, result, process_ms, trig_time, seq))
        if cmd_reset != last_reset:
            state.update_cmd_latches(last_trig, cmd_reset)
            LOG.info("CMD_RESET toggled (reserved; no action in simulator)")


async def _emit_result(
    state: SimState, result: str, process_ms: int, trig_time: datetime, seq: int
):
    await asyncio.sleep(max(process_ms, 0) / 1000.0)
    result_code, error_code = _map_codes(result)
    cycle_ms = _as_cycle_ms(process_ms, process_ms)
    state.set_result_regs(trig_time, seq, result_code, error_code, cycle_ms)
    if result == "OK":
        state.set_result_bits(ok=True, ng=False, err=False)
    elif result == "NG":
        state.set_result_bits(ok=False, ng=True, err=False)
    else:
        state.set_result_bits(ok=False, ng=True, err=True)
    state.toggle_di(2)  # ST_RESULT_TOGGLE
    LOG.info(
        "result=%s seq=%s code=%s err=%s cycle_ms=%s",
        result,
        seq,
        result_code,
        error_code,
        cycle_ms,
    )


async def _run_server(args):
    # pymodbus server adds +1 to the PDU address internally, so use base=offset+1.
    coil_addr = max(int(args.coil_offset), 0)
    di_addr = max(int(args.di_offset), 0)
    ir_addr = max(int(args.ir_offset), 0)
    coil_base = coil_addr + 1
    di_base = di_addr + 1
    ir_base = ir_addr + 1
    coil_block = ModbusSequentialDataBlock(coil_base, [0, 0])
    di_block = ModbusSequentialDataBlock(di_base, [0] * 6)
    ir_block = ModbusSequentialDataBlock(ir_base, [0] * 10)
    device_ctx = ModbusDeviceContext(di=di_block, co=coil_block, ir=ir_block, hr=None)
    state = SimState(device_ctx, coil_addr, di_addr, ir_addr)
    context = build_server_context(device_ctx)
    server = ModbusTcpServer(context, address=(args.host, args.port))

    tasks = [
        asyncio.create_task(_heartbeat_loop(state, args.heartbeat_ms)),
        asyncio.create_task(
            _trigger_loop(state, args.result, args.process_ms, args.poll_ms)
        ),
    ]
    LOG.info(
        "Modbus sim listening on %s:%d (coil=%s di=%s ir=%s)",
        args.host,
        args.port,
        args.coil_offset,
        args.di_offset,
        args.ir_offset,
    )
    try:
        await server.serve_forever()
    except asyncio.CancelledError:
        return
    finally:
        for task in tasks:
            task.cancel()
        await server.shutdown()


def _parse_args():
    p = argparse.ArgumentParser(description="Modbus TCP simulator (v2.2 point table)")
    p.add_argument("--host", default="0.0.0.0", help="Modbus TCP host")
    p.add_argument("--port", type=int, default=1502, help="Modbus TCP port")
    p.add_argument(
        "--coil-offset", type=int, default=800, help="Coil start offset (PDU 0-based)"
    )
    p.add_argument(
        "--di-offset",
        type=int,
        default=800,
        help="Discrete input start offset (PDU 0-based)",
    )
    p.add_argument(
        "--ir-offset",
        type=int,
        default=50,
        help="Input register start offset (PDU 0-based)",
    )
    p.add_argument(
        "--heartbeat-ms", type=int, default=1000, help="Heartbeat toggle interval (ms)"
    )
    p.add_argument(
        "--poll-ms", type=int, default=20, help="CMD_TRIG_TOGGLE polling interval (ms)"
    )
    p.add_argument(
        "--process-ms", type=int, default=80, help="Simulated processing time (ms)"
    )
    p.add_argument(
        "--result", default="OK", choices=RESULT_CHOICES, help="Fixed result mode value"
    )
    p.add_argument(
        "--log-level", default="info", help="Log level (debug/info/warning/error)"
    )
    return p.parse_args()


def main():
    args = _parse_args()
    level_map = {
        "debug": logging.DEBUG,
        "info": logging.INFO,
        "warning": logging.WARNING,
        "error": logging.ERROR,
        "critical": logging.CRITICAL,
    }
    logging.basicConfig(
        level=level_map.get(args.log_level, logging.INFO),
        format="%(asctime)s [%(levelname)s] %(message)s",
    )
    try:
        asyncio.run(_run_server(args))
    except KeyboardInterrupt:
        LOG.info("Simulator stopped by user")


if __name__ == "__main__":
    main()
