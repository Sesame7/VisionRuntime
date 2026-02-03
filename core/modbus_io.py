# -- coding: utf-8 --

import asyncio
import logging
import threading
from collections.abc import Iterable
from datetime import datetime
from typing import Sequence

from core.pymodbus_compat import (
	ModbusDeviceContext,
	ModbusSequentialDataBlock,
	ModbusTcpServer,
	ModbusTcpServerType,
	build_server_context,
	is_modbus_exception,
)

from core import runtime

L = logging.getLogger("sci_cam.modbus.io")


def _require_values(values: Sequence[int] | Sequence[bool] | object, count: int, label: str) -> list[int]:
	if is_modbus_exception(values):
		L.warning("Modbus read failed for %s: %r", label, values)
		return [0] * max(0, int(count))
	if not isinstance(values, Iterable):
		L.warning("Modbus read returned non-iterable for %s: %r", label, values)
		return [0] * max(0, int(count))
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
	):
		self.host = host
		self.port = port
		self.coil_offset = max(int(coil_offset), 0)
		self.di_offset = max(int(di_offset), 0)
		self.ir_offset = max(int(ir_offset), 0)
		self.heartbeat_ms = max(int(heartbeat_ms), 100)
		self._task_reg = task_reg
		self._lock = threading.Lock()
		self._server: ModbusTcpServerType | None = None
		self._tasks: list[object] = []

		# pymodbus server adds +1 internally; base must be offset + 1.
		coil_base = self.coil_offset + 1
		di_base = self.di_offset + 1
		ir_base = self.ir_offset + 1

		self._coil_block = ModbusSequentialDataBlock(coil_base, [0] * 8)
		self._di_block = ModbusSequentialDataBlock(di_base, [0] * 8)
		self._ir_block = ModbusSequentialDataBlock(ir_base, [0] * 10)
		self._device_ctx = ModbusDeviceContext(di=self._di_block, co=self._coil_block, ir=self._ir_block, hr=None)
		self._context = build_server_context(self._device_ctx)

	def start(self):
		if self._server:
			return
		L.info("Modbus TCP server listening on %s:%d", self.host, self.port)
		self._tasks = [
			self._register_task(runtime.spawn_background_task(self._serve())),
			self._register_task(runtime.spawn_background_task(self._heartbeat_loop())),
		]

	def stop(self):
		async def _cleanup():
			for task in list(self._tasks):
				cancel = getattr(task, "cancel", None)
				if callable(cancel):
					cancel()
			if self._server:
				asyncio_server = getattr(self._server, "server", None)
				if asyncio_server:
					asyncio_server.close()
					if hasattr(asyncio_server, "wait_closed"):
						await asyncio_server.wait_closed()
				await self._server.shutdown()

		runtime.run_async(_cleanup(), timeout=0.5)
		self._server = None
		self._tasks.clear()
		L.info("Modbus TCP server stopped")

	@property
	def is_running(self) -> bool:
		return bool(self._server)

	def read_coils(self, offset: int, count: int) -> list[int]:
		with self._lock:
			values = self._device_ctx.getValues(1, self.coil_offset + int(offset), int(count))
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
			self._write_result_regs_locked(trig_time, seq, result_code, error_code, cycle_ms)
			self._write_result_bits_locked(ok, ng, err)
			self._toggle_di_locked(2)  # ST_RESULT_TOGGLE

	def reset_outputs(self):
		with self._lock:
			self._set_values_locked(2, self.di_offset + 0, [0] * 6, "di_reset")
			self._set_values_locked(4, self.ir_offset + 0, [0] * 10, "ir_reset")

	def _register_task(self, task):
		if self._task_reg:
			try:
				return self._task_reg(task)
			except Exception:
				L.warning("Failed to register modbus background task", exc_info=True)
		return task

	async def _serve(self):
		try:
			server = ModbusTcpServer(self._context, address=(self.host, self.port))
			self._server = server
			await server.serve_forever()
		except asyncio.CancelledError:
			return
		except Exception:
			L.exception("Modbus server stopped unexpectedly")

	async def _heartbeat_loop(self):
		interval_s = max(self.heartbeat_ms, 100) / 1000.0
		while True:
			await asyncio.sleep(interval_s)
			with self._lock:
				self._toggle_di_locked(0)  # ST_HEARTBEAT_TOGGLE

	def _set_values_locked(self, func_code: int, address: int, values: Sequence[int], label: str):
		res = self._device_ctx.setValues(func_code, address, list(values))
		if is_modbus_exception(res):
			L.warning("Modbus write failed for %s: %r", label, res)

	def _toggle_di_locked(self, idx: int):
		addr = self.di_offset + int(idx)
		cur = self._device_ctx.getValues(2, addr, 1)
		cur_vals = _require_values(cur, 1, "di")
		value = 0 if (cur_vals and cur_vals[0]) else 1
		self._set_values_locked(2, addr, [value], "di")

	def _write_result_bits_locked(self, ok: int, ng: int, err: int):
		self._set_values_locked(2, self.di_offset + 3, [int(ok), int(ng), int(err)], "di_result")

	def _write_result_regs_locked(self, trig_time: datetime, seq: int, result_code: int, error_code: int, cycle_ms: int):
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
