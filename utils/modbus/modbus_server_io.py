# -- coding: utf-8 --

import asyncio
import logging
import threading
from collections.abc import Iterable
from datetime import datetime
from typing import Sequence

from utils.lifecycle import (
    AsyncTaskOwner,
    LoopRunner,
    wait_task_done,
)
from utils.modbus.pymodbus_compat import (
    ModbusDeviceContext,
    ModbusSequentialDataBlock,
    ModbusTcpServer,
    ModbusTcpServerType,
    build_modbus_server_context,
    is_modbus_exception,
)

L = logging.getLogger("vision_runtime.modbus.io")

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


def _require_values(
    values: Sequence[int] | Sequence[bool] | object, count: int, label: str
) -> list[int]:
    if is_modbus_exception(values):
        raise RuntimeError(f"Modbus read failed for {label}: {values!r}")
    if not isinstance(values, Iterable):
        raise RuntimeError(f"Modbus read returned non-iterable for {label}: {values!r}")
    vals = list(values)
    if len(vals) < count:
        vals += [0] * (count - len(vals))
    return vals


def _to_u16(value: int) -> int:
    iv = int(value)
    if iv < 0:
        return 0
    return 0xFFFF if iv > 0xFFFF else iv


class ModbusIO:
    def __init__(
        self,
        host: str,
        port: int,
        coil_offset: int,
        di_offset: int,
        ir_offset: int,
        hr_offset: int,
        heartbeat_ms: int,
        *,
        loop_runner: LoopRunner,
    ):
        self.host = host
        self.port = port
        self.coil_offset = max(int(coil_offset), 0)
        self.di_offset = max(int(di_offset), 0)
        self.ir_offset = max(int(ir_offset), 0)
        self.hr_offset = max(int(hr_offset), 0)
        self.heartbeat_ms = max(int(heartbeat_ms), 100)
        self._lifecycle_lock = threading.Lock()
        self._data_lock = threading.Lock()
        self._started = False
        self._stopping = False
        self._server: ModbusTcpServerType | None = None
        self._serve_task = None
        self._heartbeat_task = None
        self._heartbeat_stop = threading.Event()
        self._tasks = AsyncTaskOwner(
            owner_name="modbus_io",
            loop_runner=loop_runner,
        )

        # pymodbus server adds +1 internally; base must be offset + 1.
        coil_base = self.coil_offset + 1
        di_base = self.di_offset + 1
        ir_base = self.ir_offset + 1
        hr_base = self.hr_offset + 1

        self._coil_block = ModbusSequentialDataBlock(coil_base, [0] * 8)
        self._di_block = ModbusSequentialDataBlock(di_base, [0] * 8)
        self._ir_block = ModbusSequentialDataBlock(ir_base, [0] * IR_REG_COUNT)
        self._hr_block = ModbusSequentialDataBlock(hr_base, [0] * HR_REG_COUNT)
        self._device_ctx = ModbusDeviceContext(
            di=self._di_block, co=self._coil_block, ir=self._ir_block, hr=self._hr_block
        )
        self._context = build_modbus_server_context(self._device_ctx)

    def start(self):
        with self._lifecycle_lock:
            if self._started:
                return
            self._started = True
            self._heartbeat_stop.clear()
            try:
                self._tasks.cancel_and_clear_local_tasks()
                self._serve_task = self._tasks.spawn(self._serve())
                self._heartbeat_task = self._tasks.spawn(self._heartbeat_loop())
            except Exception:
                self._started = False
                self._serve_task = None
                self._heartbeat_task = None
                self._heartbeat_stop.set()
                raise

    def stop(self):
        with self._lifecycle_lock:
            if self._stopping:
                return
            if not self._started and self._server is None:
                return
            self._stopping = True
            self._started = False
            self._heartbeat_stop.set()
            serve_task = self._serve_task
            heartbeat_task = self._heartbeat_task
            server = self._server
            self._serve_task = None
            self._heartbeat_task = None
            self._server = None
        try:
            # Heartbeat is an internal periodic task; cancel immediately.
            if heartbeat_task is not None:
                heartbeat_task.cancel()

            # Gracefully stop listener first so accept coroutine can exit cleanly.
            if server is not None:

                async def _cleanup():
                    await server.shutdown()

                try:
                    self._tasks.loop_runner.run_async(_cleanup(), timeout=0.5)
                except Exception:
                    L.exception("Modbus server graceful shutdown failed")

            wait_task_done(serve_task, timeout=0.5, label="modbus_io.serve", logger=L)
            wait_task_done(
                heartbeat_task, timeout=0.5, label="modbus_io.heartbeat", logger=L
            )
            self._tasks.cancel_and_clear_local_tasks()
            L.info("Modbus TCP server stopped")
        finally:
            with self._lifecycle_lock:
                self._stopping = False

    def read_coils(self, offset: int, count: int) -> list[int]:
        with self._data_lock:
            values = self._device_ctx.getValues(
                1, self.coil_offset + int(offset), int(count)
            )
            return _require_values(values, count, "coils")

    def read_holding_registers(self, offset: int, count: int) -> list[int]:
        with self._data_lock:
            values = self._device_ctx.getValues(
                3, self.hr_offset + int(offset), int(count)
            )
            return _require_values(values, count, "holding_registers")

    def read_input_registers(self, offset: int, count: int) -> list[int]:
        with self._data_lock:
            values = self._device_ctx.getValues(
                4, self.ir_offset + int(offset), int(count)
            )
            return _require_values(values, count, "input_registers")

    def toggle_di(self, idx: int):
        with self._data_lock:
            self._toggle_di_locked(idx)

    def write_result(
        self,
        trig_time: datetime,
        seq: int,
        result_code: int,
        error_code: int,
        cycle_ms: int,
        ok: int,
        ng: int,
        err: int,
        total_count: int = 0,
        ok_count: int = 0,
        ng_count: int = 0,
        err_count: int = 0,
    ):
        with self._data_lock:
            self._write_result_regs_locked(
                trig_time, seq, result_code, error_code, cycle_ms
            )
            self._write_counter_regs_locked(total_count, ok_count, ng_count, err_count)
            self._write_result_bits_locked(ok, ng, err)
            self._toggle_di_locked(2)  # ST_RESULT_TOGGLE

    def write_recipe_ack_status(self, status: int):
        with self._data_lock:
            self._write_recipe_ack_locked(seq=None, status=status)

    def write_recipe_ack(self, seq: int, status: int):
        with self._data_lock:
            self._write_recipe_ack_locked(seq=seq, status=status)

    def reset_outputs(self):
        with self._data_lock:
            self._set_values_locked(2, self.di_offset + 0, [0] * 6, "di_reset")
            self._set_values_locked(
                4, self.ir_offset + 0, [0] * IR_REG_COUNT, "ir_reset"
            )

    async def _serve(self):
        server = ModbusTcpServer(self._context, address=(self.host, self.port))
        self._server = server
        L.info("Modbus TCP server listening on %s:%d", self.host, self.port)
        try:
            await server.serve_forever()
        except asyncio.CancelledError:
            return
        finally:
            if self._server is server:
                self._server = None

    async def _heartbeat_loop(self):
        interval_s = max(self.heartbeat_ms, 100) / 1000.0
        while not self._heartbeat_stop.is_set():
            await asyncio.sleep(interval_s)
            if self._heartbeat_stop.is_set():
                break
            with self._data_lock:
                self._toggle_di_locked(0)  # ST_HEARTBEAT_TOGGLE

    def _set_values_locked(
        self, func_code: int, address: int, values: Sequence[int], label: str
    ):
        res = self._device_ctx.setValues(func_code, address, list(values))
        if is_modbus_exception(res):
            raise RuntimeError(f"Modbus write failed for {label}: {res!r}")

    def _toggle_di_locked(self, idx: int):
        addr = self.di_offset + int(idx)
        cur = self._device_ctx.getValues(2, addr, 1)
        cur_vals = _require_values(cur, 1, "di")
        value = 0 if (cur_vals and cur_vals[0]) else 1
        self._set_values_locked(2, addr, [value], "di")

    def _write_result_bits_locked(self, ok: int, ng: int, err: int):
        self._set_values_locked(
            2, self.di_offset + 3, [int(ok), int(ng), int(err)], "di_result"
        )

    def _write_result_regs_locked(
        self,
        trig_time: datetime,
        seq: int,
        result_code: int,
        error_code: int,
        cycle_ms: int,
    ):
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
        self._set_values_locked(4, self.ir_offset, values, "ir_result")

    def _write_counter_regs_locked(
        self, total_count: int, ok_count: int, ng_count: int, err_count: int
    ):
        values = [
            _to_u16(total_count),
            _to_u16(ok_count),
            _to_u16(ng_count),
            _to_u16(err_count),
        ]
        self._set_values_locked(
            4,
            self.ir_offset + IR_COUNTER_REG_BASE,
            values,
            "ir_counters",
        )

    def _write_recipe_ack_locked(self, seq: int | None, status: int):
        values = []
        start_addr = self.ir_offset + IR_RECIPE_ACK_SEQ_REG
        if seq is None:
            start_addr = self.ir_offset + IR_RECIPE_ACK_STATUS_REG
        else:
            values.append(_to_u16(seq))
        values.append(_to_u16(status))
        self._set_values_locked(4, start_addr, values, "ir_recipe_ack")


__all__ = [
    "ModbusIO",
    "RECIPE_ACK_IDLE",
    "RECIPE_ACK_RUNNING",
    "RECIPE_ACK_OK",
    "RECIPE_ACK_ERR",
]
