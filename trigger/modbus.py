# -- coding: utf-8 --

import asyncio
from concurrent.futures import CancelledError as FutureCancelledError
import logging
import threading

from utils.lifecycle import AsyncTaskOwner, LoopRunner, wait_task_done
from utils.modbus.modbus_server_io import (
    RECIPE_ACK_ERR,
    RECIPE_ACK_IDLE,
    RECIPE_ACK_OK,
    RECIPE_ACK_RUNNING,
)
from trigger.base import BaseTrigger, TriggerConfig, register_trigger

L = logging.getLogger("vision_runtime.trigger.modbus")


@register_trigger("modbus")
class ModbusTrigger(BaseTrigger):
    def __init__(
        self,
        cfg: TriggerConfig,
        on_trigger,
        modbus_io,
        poll_ms: int = 20,
        on_recipe_switch=None,
        *,
        loop_runner: LoopRunner,
    ):
        super().__init__(cfg, on_trigger)
        self._io = modbus_io
        self._poll_ms = max(int(poll_ms), 5)
        self._on_recipe_switch = on_recipe_switch
        self._tasks = AsyncTaskOwner(
            owner_name="modbus_trigger",
            loop_runner=loop_runner,
        )
        self._task = None
        self._stopping = False
        self._state_lock = threading.Lock()
        self._last_cmd_trig = None
        self._last_recipe_seq = None

    def start(self):
        with self._state_lock:
            if self._task is not None:
                return
            self._tasks.clear_local_tasks()
            self._task = self._tasks.spawn(self._poll_loop())

    def stop(self):
        with self._state_lock:
            if self._stopping:
                return
            task = self._task
            self._task = None
            if task is None:
                return
            self._stopping = True
        try:
            task.cancel()
            wait_task_done(task, timeout=0.5, label="modbus_trigger.poll", logger=L)
            self._tasks.cancel_and_clear_local_tasks()
            L.info("Modbus trigger stopped")
        finally:
            with self._state_lock:
                self._stopping = False

    def raise_if_failed(self):
        task = self._task
        if task is None or not hasattr(task, "done") or not task.done():
            return
        try:
            err = task.exception()
        except FutureCancelledError:
            return
        if err is None:
            return
        raise RuntimeError(
            f"ModbusTrigger stopped unexpectedly ({type(err).__name__})"
        ) from err

    async def _poll_loop(self):
        interval_s = max(self._poll_ms, 5) / 1000.0
        try:
            while True:
                await asyncio.sleep(interval_s)
                try:
                    cmds = self._io.read_coils(0, 1)
                    trig_val = int(cmds[0])
                    recipe_regs = self._io.read_holding_registers(0, 2)
                    recipe_slot = int(recipe_regs[0])
                    recipe_seq = int(recipe_regs[1]) & 0xFFFF
                except Exception as exc:
                    raise RuntimeError(
                        "ModbusTrigger poll failed stage=read_inputs"
                    ) from exc
                last_trig = self._last_cmd_trig
                if last_trig is None:
                    self._last_cmd_trig = trig_val
                    self._last_recipe_seq = recipe_seq
                    try:
                        self._io.write_recipe_ack(
                            seq=recipe_seq, status=RECIPE_ACK_IDLE
                        )
                    except Exception as exc:
                        raise RuntimeError(
                            "ModbusTrigger poll failed stage=write_recipe_ack_init"
                        ) from exc
                    continue

                if trig_val != last_trig:
                    self._last_cmd_trig = trig_val
                    try:
                        accepted = bool(self.on_trigger("MODBUS"))
                    except Exception as exc:
                        raise RuntimeError(
                            "ModbusTrigger poll failed stage=on_trigger"
                        ) from exc
                    if accepted:
                        try:
                            self._io.toggle_di(1)  # ST_ACCEPT_TOGGLE
                        except Exception as exc:
                            raise RuntimeError(
                                "ModbusTrigger poll failed stage=toggle_di"
                            ) from exc

                if recipe_seq != self._last_recipe_seq:
                    self._last_recipe_seq = recipe_seq
                    try:
                        self._io.write_recipe_ack_status(RECIPE_ACK_RUNNING)
                    except Exception as exc:
                        raise RuntimeError(
                            "ModbusTrigger poll failed stage=recipe_ack_running"
                        ) from exc
                    ok = False
                    msg = ""
                    if self._on_recipe_switch:
                        try:
                            outcome = self._on_recipe_switch(recipe_slot)
                            if isinstance(outcome, tuple):
                                ok = bool(outcome[0]) if len(outcome) > 0 else False
                                msg = str(outcome[1]) if len(outcome) > 1 else ""
                            else:
                                ok = bool(outcome)
                                msg = ""
                        except Exception as exc:
                            msg = str(exc) or type(exc).__name__
                            ok = False
                    else:
                        msg = "recipe switch callback is not configured"
                    ack_status = RECIPE_ACK_OK if bool(ok) else RECIPE_ACK_ERR
                    try:
                        self._io.write_recipe_ack(seq=recipe_seq, status=ack_status)
                    except Exception as exc:
                        raise RuntimeError(
                            "ModbusTrigger poll failed stage=recipe_ack_done"
                        ) from exc
                    if ok:
                        L.info(
                            "Modbus recipe switch success seq=%d slot=%d msg=%s",
                            recipe_seq,
                            recipe_slot,
                            msg,
                        )
                    else:
                        L.warning(
                            "Modbus recipe switch failed seq=%d slot=%d msg=%s",
                            recipe_seq,
                            recipe_slot,
                            msg,
                        )
        except asyncio.CancelledError:
            raise
        except Exception:
            L.exception("Modbus trigger poll loop failed")
            raise
