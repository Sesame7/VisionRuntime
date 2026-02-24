# -- coding: utf-8 --

import asyncio
import logging

from core import runtime
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
    ):
        super().__init__(cfg, on_trigger)
        self._io = modbus_io
        self._poll_ms = max(int(poll_ms), 5)
        self._on_reset = on_reset
        self._task = None
        self._last_cmd_trig = None
        self._last_cmd_reset = None

    def start(self):
        if self._task:
            return
        self._task = runtime.spawn_background_task(self._poll_loop())

    def stop(self):
        task = self._task
        self._task = None
        if task is None:
            return
        cancel = getattr(task, "cancel", None)
        if callable(cancel):
            cancel()
        L.info("Modbus trigger stopped")

    async def _poll_loop(self):
        interval_s = max(self._poll_ms, 5) / 1000.0
        while True:
            await asyncio.sleep(interval_s)
            cmds = self._io.read_coils(0, 2)
            trig_val, reset_val = int(cmds[0]), int(cmds[1])
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
                except Exception:
                    L.warning("Modbus trigger handler failed", exc_info=True)
                    accepted = False
                if accepted:
                    self._io.toggle_di(1)  # ST_ACCEPT_TOGGLE

            if reset_val != last_reset:
                self._last_cmd_reset = reset_val
                if self._on_reset:
                    try:
                        self._on_reset()
                    except Exception:
                        L.warning("Modbus reset handler failed", exc_info=True)

    @property
    def is_running(self) -> bool:
        return bool(self._task)
