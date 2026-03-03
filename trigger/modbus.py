# -- coding: utf-8 --

import asyncio
from concurrent.futures import CancelledError as FutureCancelledError
import logging
import threading

from utils.lifecycle import AsyncTaskOwner, LoopRunner, wait_task_done
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
        on_reset=None,
        *,
        loop_runner: LoopRunner,
    ):
        super().__init__(cfg, on_trigger)
        self._io = modbus_io
        self._poll_ms = max(int(poll_ms), 5)
        self._on_reset = on_reset
        self._tasks = AsyncTaskOwner(
            owner_name="modbus_trigger",
            loop_runner=loop_runner,
        )
        self._task = None
        self._stopping = False
        self._state_lock = threading.Lock()
        self._last_cmd_trig = None
        self._last_cmd_reset = None

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
                    cmds = self._io.read_coils(0, 2)
                    trig_val, reset_val = int(cmds[0]), int(cmds[1])
                except Exception as exc:
                    raise RuntimeError(
                        "ModbusTrigger poll failed stage=read_coils"
                    ) from exc
                last_trig = self._last_cmd_trig
                last_reset = self._last_cmd_reset
                if last_trig is None:
                    self._last_cmd_trig = trig_val
                    self._last_cmd_reset = reset_val
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

                if reset_val != last_reset:
                    self._last_cmd_reset = reset_val
                    if self._on_reset:
                        try:
                            self._on_reset()
                        except Exception as exc:
                            raise RuntimeError(
                                "ModbusTrigger poll failed stage=on_reset"
                            ) from exc
        except asyncio.CancelledError:
            raise
        except Exception:
            L.exception("Modbus trigger poll loop failed")
            raise
