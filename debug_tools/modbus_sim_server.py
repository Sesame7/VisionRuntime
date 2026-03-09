# -- coding: utf-8 --
"""Modbus TCP simulator for the v3 point table (coil + DI + IR + HR)."""

import argparse
import asyncio
import logging
import time
from collections.abc import Iterable
from datetime import datetime, timezone
from threading import Lock

from pymodbus.datastore import (
    ModbusDeviceContext,
    ModbusSequentialDataBlock,
    ModbusServerContext,
)
from pymodbus.pdu import ExceptionResponse
from pymodbus.server import ModbusTcpServer

LOG = logging.getLogger("modbus.sim")

IR_RESULT_REG_COUNT = 10
IR_COUNTER_REG_BASE = 10
IR_RECIPE_ACK_SEQ_REG = 14
IR_RECIPE_ACK_STATUS_REG = 15
IR_REG_COUNT = 16
HR_REG_COUNT = 8

RECIPE_ACK_IDLE = 0
RECIPE_ACK_RUNNING = 1
RECIPE_ACK_OK = 2
RECIPE_ACK_ERR = 3

RESULT_CODE_MAP = {
    "OK": 1,
    "NG": 2,
    "TIMEOUT": 3,
    "ERROR": 3,
    "DETECT_EXCEPTION": 3,
    "CAMERA_ERROR": 3,
    "QUEUE_OVERFLOW": 3,
}

ERROR_CODE_MAP = {
    "OK": 0,
    "NG": 0,
    "TIMEOUT": 1,
    "ERROR": 2,
    "DETECT_EXCEPTION": 2,
    "CAMERA_ERROR": 3,
    "QUEUE_OVERFLOW": 4,
}

RESULT_CHOICES = (
    "OK",
    "NG",
    "TIMEOUT",
    "ERROR",
    "DETECT_EXCEPTION",
    "CAMERA_ERROR",
    "QUEUE_OVERFLOW",
)


def _to_u16(value: int | float | None) -> int:
    if value is None:
        return 0
    iv = int(round(float(value)))
    if iv < 0:
        return 0
    return 0xFFFF if iv > 0xFFFF else iv


def _require_values(
    values: list[int] | list[bool] | object, count: int, label: str
) -> list[int]:
    if _is_modbus_error(values):
        raise RuntimeError(f"Sim read failed for {label}: {values!r}")
    if not isinstance(values, Iterable):
        raise RuntimeError(f"Sim read failed for {label}: non-iterable {values!r}")
    vals = list(values)
    if len(vals) < count:
        vals += [0] * (count - len(vals))
    return vals


def _is_modbus_error(value: object) -> bool:
    if isinstance(value, ExceptionResponse):
        return True
    checker = getattr(value, "isError", None)
    if callable(checker):
        try:
            return bool(checker())
        except Exception:
            return True
    return False


class SimState:
    def __init__(
        self,
        device_ctx: ModbusDeviceContext,
        *,
        coil_addr: int,
        di_addr: int,
        ir_addr: int,
        hr_addr: int,
    ):
        self._lock = Lock()
        self._ctx = device_ctx
        self._coil_addr = int(coil_addr)
        self._di_addr = int(di_addr)
        self._ir_addr = int(ir_addr)
        self._hr_addr = int(hr_addr)

        self._last_cmd_trig = 0
        self._last_recipe_seq = 0
        self._result_seq = 0

        self._total_count = 0
        self._ok_count = 0
        self._ng_count = 0
        self._err_count = 0

    def init_recipe_ack(self) -> None:
        self.write_recipe_ack(seq=0, status=RECIPE_ACK_IDLE)

    def next_result_seq(self) -> int:
        with self._lock:
            self._result_seq += 1
            if self._result_seq > 0xFFFF:
                self._result_seq = 1
            return self._result_seq

    def read_trigger_toggle(self) -> int:
        with self._lock:
            values = self._ctx.getValues(1, self._coil_addr, 1)
            bits = [1 if v else 0 for v in _require_values(values, 1, "coils")]
            return int(bits[0])

    def read_recipe_command(self) -> tuple[int, int]:
        with self._lock:
            values = self._ctx.getValues(3, self._hr_addr, 2)
            regs = _require_values(values, 2, "holding_registers")
            return int(regs[0]) & 0xFFFF, int(regs[1]) & 0xFFFF

    def read_latches(self) -> tuple[int, int]:
        with self._lock:
            return int(self._last_cmd_trig), int(self._last_recipe_seq)

    def update_latches(
        self, *, trig_val: int | None = None, recipe_seq: int | None = None
    ):
        with self._lock:
            if trig_val is not None:
                self._last_cmd_trig = int(trig_val) & 0x1
            if recipe_seq is not None:
                self._last_recipe_seq = int(recipe_seq) & 0xFFFF

    def toggle_di(self, idx: int) -> None:
        with self._lock:
            self._toggle_di_locked(int(idx))

    def write_recipe_ack_status(self, status: int) -> None:
        with self._lock:
            self._set_values_locked(
                4,
                self._ir_addr + IR_RECIPE_ACK_STATUS_REG,
                [_to_u16(status)],
                "ir_recipe_ack_status",
            )

    def write_recipe_ack(self, *, seq: int, status: int) -> None:
        with self._lock:
            self._set_values_locked(
                4,
                self._ir_addr + IR_RECIPE_ACK_SEQ_REG,
                [_to_u16(seq), _to_u16(status)],
                "ir_recipe_ack",
            )

    def commit_result(
        self,
        *,
        trig_time: datetime,
        seq: int,
        result_code: int,
        error_code: int,
        cycle_ms: int,
    ) -> None:
        with self._lock:
            ok = 1 if int(result_code) == 1 else 0
            ng = 1 if int(result_code) != 1 else 0
            err = 1 if int(result_code) == 3 else 0

            self._total_count = _to_u16(self._total_count + 1)
            if ok:
                self._ok_count = _to_u16(self._ok_count + 1)
            elif err:
                self._err_count = _to_u16(self._err_count + 1)
            else:
                self._ng_count = _to_u16(self._ng_count + 1)

            result_regs = [
                _to_u16(trig_time.year),
                _to_u16(trig_time.month),
                _to_u16(trig_time.day),
                _to_u16(trig_time.hour),
                _to_u16(trig_time.minute),
                _to_u16(trig_time.second),
                _to_u16(seq),
                _to_u16(result_code),
                _to_u16(error_code),
                _to_u16(cycle_ms),
            ]
            self._set_values_locked(4, self._ir_addr, result_regs, "ir_result")

            counter_regs = [
                _to_u16(self._total_count),
                _to_u16(self._ok_count),
                _to_u16(self._ng_count),
                _to_u16(self._err_count),
            ]
            self._set_values_locked(
                4,
                self._ir_addr + IR_COUNTER_REG_BASE,
                counter_regs,
                "ir_counters",
            )

            # Emit order: IR -> DI bits -> result toggle.
            self._set_values_locked(
                2,
                self._di_addr + 3,
                [ok, ng, err],
                "di_result",
            )
            self._toggle_di_locked(2)  # ST_RESULT_TOGGLE

    def _toggle_di_locked(self, idx: int) -> None:
        addr = self._di_addr + int(idx)
        cur = self._ctx.getValues(2, addr, 1)
        cur_vals = _require_values(cur, 1, "discrete_inputs")
        val = 0 if (cur_vals and cur_vals[0]) else 1
        self._set_values_locked(2, addr, [val], "discrete_inputs")

    def _set_values_locked(
        self,
        func_code: int,
        address: int,
        values: list[int],
        label: str,
    ) -> None:
        res = self._ctx.setValues(func_code, int(address), list(values))
        if _is_modbus_error(res):
            raise RuntimeError(f"Sim write failed for {label}: {res!r}")


def _map_codes(result: str) -> tuple[int, int]:
    key = str(result or "").strip().upper()
    result_code = int(RESULT_CODE_MAP.get(key, 3))
    error_code = int(ERROR_CODE_MAP.get(key, 2))
    return result_code, error_code


async def _heartbeat_loop(state: SimState, interval_ms: int):
    interval_s = max(int(interval_ms), 50) / 1000.0
    while True:
        await asyncio.sleep(interval_s)
        state.toggle_di(0)  # ST_HEARTBEAT_TOGGLE


async def _emit_result(
    state: SimState,
    *,
    result: str,
    process_ms: int,
    trig_time: datetime,
    seq: int,
):
    await asyncio.sleep(max(int(process_ms), 0) / 1000.0)
    result_code, error_code = _map_codes(result)
    cycle_ms = _to_u16(process_ms)
    state.commit_result(
        trig_time=trig_time,
        seq=seq,
        result_code=result_code,
        error_code=error_code,
        cycle_ms=cycle_ms,
    )
    LOG.info(
        "result=%s seq=%s code=%s err=%s cycle_ms=%s",
        result,
        seq,
        result_code,
        error_code,
        cycle_ms,
    )


async def _emit_recipe_ack(
    state: SimState,
    *,
    recipe_slot: int,
    recipe_seq: int,
    recipe_ms: int,
):
    state.write_recipe_ack_status(RECIPE_ACK_RUNNING)
    await asyncio.sleep(max(int(recipe_ms), 0) / 1000.0)
    state.write_recipe_ack(seq=recipe_seq, status=RECIPE_ACK_OK)
    LOG.info(
        "recipe_switch seq=%s slot=%s status=OK",
        recipe_seq,
        recipe_slot,
    )


async def _trigger_loop(
    state: SimState,
    *,
    result: str,
    process_ms: int,
    recipe_ms: int,
    poll_ms: int,
):
    poll_s = max(int(poll_ms), 10) / 1000.0
    while True:
        await asyncio.sleep(poll_s)
        trig_val = state.read_trigger_toggle()
        recipe_slot, recipe_seq = state.read_recipe_command()
        last_trig, last_recipe_seq = state.read_latches()

        if trig_val != last_trig:
            state.update_latches(trig_val=trig_val)
            seq = state.next_result_seq()
            trig_time = datetime.now(timezone.utc)
            state.toggle_di(1)  # ST_ACCEPT_TOGGLE
            asyncio.create_task(
                _emit_result(
                    state,
                    result=result,
                    process_ms=process_ms,
                    trig_time=trig_time,
                    seq=seq,
                )
            )

        if recipe_seq != last_recipe_seq:
            state.update_latches(recipe_seq=recipe_seq)
            asyncio.create_task(
                _emit_recipe_ack(
                    state,
                    recipe_slot=recipe_slot,
                    recipe_seq=recipe_seq,
                    recipe_ms=recipe_ms,
                )
            )


async def _run_server(args):
    # pymodbus server adds +1 internally; block base must be offset + 1.
    coil_addr = max(int(args.coil_offset), 0)
    di_addr = max(int(args.di_offset), 0)
    ir_addr = max(int(args.ir_offset), 0)
    hr_addr = max(int(args.hr_offset), 0)
    coil_base = coil_addr + 1
    di_base = di_addr + 1
    ir_base = ir_addr + 1
    hr_base = hr_addr + 1

    coil_block = ModbusSequentialDataBlock(coil_base, [0] * 8)
    di_block = ModbusSequentialDataBlock(di_base, [0] * 8)
    ir_block = ModbusSequentialDataBlock(ir_base, [0] * IR_REG_COUNT)
    hr_block = ModbusSequentialDataBlock(hr_base, [0] * HR_REG_COUNT)
    device_ctx = ModbusDeviceContext(
        di=di_block, co=coil_block, ir=ir_block, hr=hr_block
    )
    state = SimState(
        device_ctx,
        coil_addr=coil_addr,
        di_addr=di_addr,
        ir_addr=ir_addr,
        hr_addr=hr_addr,
    )
    state.init_recipe_ack()
    context = ModbusServerContext(devices=device_ctx, single=True)
    server = ModbusTcpServer(context, address=(args.host, args.port))

    tasks = [
        asyncio.create_task(_heartbeat_loop(state, args.heartbeat_ms)),
        asyncio.create_task(
            _trigger_loop(
                state,
                result=args.result,
                process_ms=args.process_ms,
                recipe_ms=args.recipe_ms,
                poll_ms=args.poll_ms,
            )
        ),
    ]

    LOG.info(
        "Modbus sim listening on %s:%d (coil=%s di=%s ir=%s hr=%s)",
        args.host,
        args.port,
        args.coil_offset,
        args.di_offset,
        args.ir_offset,
        args.hr_offset,
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
    p = argparse.ArgumentParser(description="Modbus TCP simulator (v3 point table)")
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
        "--hr-offset",
        type=int,
        default=50,
        help="Holding register start offset (PDU 0-based)",
    )
    p.add_argument(
        "--heartbeat-ms", type=int, default=1000, help="Heartbeat toggle interval (ms)"
    )
    p.add_argument(
        "--poll-ms", type=int, default=20, help="Trigger/recipe polling interval (ms)"
    )
    p.add_argument(
        "--process-ms",
        type=int,
        default=80,
        help="Simulated result processing time (ms)",
    )
    p.add_argument(
        "--recipe-ms", type=int, default=80, help="Simulated recipe switch time (ms)"
    )
    p.add_argument(
        "--result",
        default="OK",
        choices=RESULT_CHOICES,
        help="Fixed result mode value",
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
    logging.Formatter.converter = time.gmtime
    logging.basicConfig(
        level=level_map.get(args.log_level, logging.INFO),
        format="%(asctime)sZ [%(levelname)s] %(message)s",
    )
    try:
        asyncio.run(_run_server(args))
    except KeyboardInterrupt:
        LOG.info("Simulator stopped by user")


if __name__ == "__main__":
    main()
