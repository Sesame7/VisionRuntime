import asyncio
import unittest
from datetime import datetime, timezone

from core.contracts import OutputRecord
from output.modbus import ModbusOutput
from trigger.base import TriggerConfig
from trigger.modbus import ModbusTrigger
from utils.lifecycle import LoopRunner
from utils.modbus.modbus_server_io import (
    ModbusIO,
    RECIPE_ACK_OK,
    RECIPE_ACK_RUNNING,
)


class _FakeResultIo:
    def __init__(self):
        self.calls = []

    def write_result(self, **kwargs):
        self.calls.append(dict(kwargs))


class _FakeTriggerIo:
    def __init__(self):
        self._coils = [0, 1, 1, 1]
        self._holding = [(1, 0), (1, 0), (2, 1), (2, 1)]
        self._coil_idx = 0
        self._holding_idx = 0
        self.toggled_di: list[int] = []
        self.ack_statuses: list[int] = []
        self.acks: list[tuple[int, int]] = []

    def read_coils(self, offset: int, count: int):
        _ = offset, count
        if self._coil_idx < len(self._coils):
            v = self._coils[self._coil_idx]
            self._coil_idx += 1
        else:
            v = self._coils[-1]
        return [int(v)]

    def read_holding_registers(self, offset: int, count: int):
        _ = offset, count
        if self._holding_idx < len(self._holding):
            slot, seq = self._holding[self._holding_idx]
            self._holding_idx += 1
        else:
            slot, seq = self._holding[-1]
        return [int(slot), int(seq)]

    def write_recipe_ack_status(self, status: int):
        self.ack_statuses.append(int(status))

    def write_recipe_ack(self, seq: int, status: int):
        self.acks.append((int(seq), int(status)))

    def toggle_di(self, idx: int):
        self.toggled_di.append(int(idx))


class TestModbusProtocol(unittest.TestCase):
    def test_modbus_output_writes_result_with_counters(self):
        io = _FakeResultIo()
        stats_payload = {"total": 12, "ok": 7, "ng": 3, "error": 2}
        output = ModbusOutput(io, stats_provider=lambda: dict(stats_payload))
        rec = OutputRecord(
            trigger_seq=10,
            triggered_at=datetime.now(timezone.utc),
            result="NG",
            result_code="NG",
            duration_ms=18.4,
        )

        output.publish(rec, None)

        self.assertEqual(len(io.calls), 1)
        payload = io.calls[0]
        self.assertEqual(payload["total_count"], 12)
        self.assertEqual(payload["ok_count"], 7)
        self.assertEqual(payload["ng_count"], 3)
        self.assertEqual(payload["err_count"], 2)

    def test_modbus_trigger_handles_recipe_seq_change_and_ack(self):
        io = _FakeTriggerIo()
        switched_slots: list[int] = []

        def _on_recipe_switch(slot: int):
            switched_slots.append(int(slot))
            return True, "ok"

        trig = ModbusTrigger(
            TriggerConfig(),
            lambda _src: True,
            io,
            poll_ms=5,
            on_recipe_switch=_on_recipe_switch,
            loop_runner=LoopRunner(),
        )

        async def _run_briefly():
            task = asyncio.create_task(trig._poll_loop())
            await asyncio.sleep(0.08)
            task.cancel()
            try:
                await task
            except asyncio.CancelledError:
                pass

        asyncio.run(_run_briefly())

        self.assertEqual(switched_slots, [2])
        self.assertIn(1, io.toggled_di)  # ST_ACCEPT_TOGGLE on trigger accepted
        self.assertIn(RECIPE_ACK_RUNNING, io.ack_statuses)
        self.assertIn((1, RECIPE_ACK_OK), io.acks)

    def test_modbus_io_writes_recipe_ack_and_counters_to_ir(self):
        io = ModbusIO(
            host="127.0.0.1",
            port=1502,
            coil_offset=0,
            di_offset=0,
            ir_offset=0,
            hr_offset=0,
            heartbeat_ms=1000,
            loop_runner=LoopRunner(),
        )

        io.write_result(
            trig_time=datetime(2026, 3, 9, 1, 2, 3, tzinfo=timezone.utc),
            seq=7,
            result_code=1,
            error_code=0,
            cycle_ms=99,
            ok=1,
            ng=0,
            err=0,
            total_count=21,
            ok_count=12,
            ng_count=5,
            err_count=4,
        )
        self.assertEqual(io.read_input_registers(10, 4), [21, 12, 5, 4])

        io.write_recipe_ack_status(RECIPE_ACK_RUNNING)
        self.assertEqual(io.read_input_registers(15, 1), [RECIPE_ACK_RUNNING])

        io.write_recipe_ack(seq=11, status=RECIPE_ACK_OK)
        self.assertEqual(io.read_input_registers(14, 2), [11, RECIPE_ACK_OK])


if __name__ == "__main__":
    unittest.main()
