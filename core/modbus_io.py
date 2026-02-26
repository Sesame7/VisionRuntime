# -- coding: utf-8 --

import asyncio
import logging
import threading
from collections.abc import Iterable
from datetime import datetime
from typing import Sequence

from core.lifecycle import AsyncTaskOwner, LoopRunner, run_async_cleanup
from core.pymodbus_compat import (
    ModbusDeviceContext,
    ModbusSequentialDataBlock,
    ModbusTcpServer,
    ModbusTcpServerType,
    build_server_context,
    is_modbus_exception,
)

L = logging.getLogger("vision_runtime.modbus.io")


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


class ModbusIO:
    def __init__(
        self,
        host: str,
        port: int,
        coil_offset: int,
        di_offset: int,
        ir_offset: int,
        heartbeat_ms: int,
        task_reg=None,
        *,
        loop_runner: LoopRunner,
    ):
        self.host = host
        self.port = port
        self.coil_offset = max(int(coil_offset), 0)
        self.di_offset = max(int(di_offset), 0)
        self.ir_offset = max(int(ir_offset), 0)
        self.heartbeat_ms = max(int(heartbeat_ms), 100)
        self._state_lock = threading.Lock()
        self._lock = threading.Lock()
        self._started = False
        self._server: ModbusTcpServerType | None = None
        self._serve_task = None
        self._heartbeat_task = None
        self._heartbeat_stop = threading.Event()
        self._tasks = AsyncTaskOwner(
            task_reg=task_reg,
            logger=L,
            owner_name="modbus_io",
            loop_runner=loop_runner,
        )

        # pymodbus server adds +1 internally; base must be offset + 1.
        coil_base = self.coil_offset + 1
        di_base = self.di_offset + 1
        ir_base = self.ir_offset + 1

        self._coil_block = ModbusSequentialDataBlock(coil_base, [0] * 8)
        self._di_block = ModbusSequentialDataBlock(di_base, [0] * 8)
        self._ir_block = ModbusSequentialDataBlock(ir_base, [0] * 10)
        self._device_ctx = ModbusDeviceContext(
            di=self._di_block, co=self._coil_block, ir=self._ir_block, hr=None
        )
        self._context = build_server_context(self._device_ctx)

    def start(self):
        with self._state_lock:
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
        with self._state_lock:
            if not self._started and self._server is None:
                return
            self._started = False
            self._heartbeat_stop.set()
            serve_task = self._serve_task
            heartbeat_task = self._heartbeat_task
            self._serve_task = None
            self._heartbeat_task = None
        for task in (serve_task, heartbeat_task):
            if task is not None:
                task.cancel()

        async def _cleanup():
            # Tasks successfully registered via task_reg are owned/cancelled by the
            # external manager (OutputManager). Only local fallback tasks are
            # cancelled here.
            self._tasks.cancel_local_tasks()
            if self._server:
                server = self._server
                server.close()
                await server.shutdown()

        run_async_cleanup(
            _cleanup(),
            timeout=0.5,
            loop_runner=self._tasks.loop_runner,
        )
        self._server = None
        self._tasks.cancel_and_clear_local_tasks()
        L.info("Modbus TCP server stopped")

    def read_coils(self, offset: int, count: int) -> list[int]:
        with self._lock:
            values = self._device_ctx.getValues(
                1, self.coil_offset + int(offset), int(count)
            )
            return _require_values(values, count, "coils")

    def toggle_di(self, idx: int):
        with self._lock:
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
    ):
        with self._lock:
            self._write_result_regs_locked(
                trig_time, seq, result_code, error_code, cycle_ms
            )
            self._write_result_bits_locked(ok, ng, err)
            self._toggle_di_locked(2)  # ST_RESULT_TOGGLE

    def reset_outputs(self):
        with self._lock:
            self._set_values_locked(2, self.di_offset + 0, [0] * 6, "di_reset")
            self._set_values_locked(4, self.ir_offset + 0, [0] * 10, "ir_reset")

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
            with self._lock:
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


__all__ = ["ModbusIO"]
