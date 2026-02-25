# -- coding: utf-8 --

import asyncio
import logging

from core.lifecycle import AsyncTaskOwner, LoopRunner, run_async_cleanup
from trigger.base import BaseTrigger, TriggerConfig, register_trigger

L = logging.getLogger("vision_runtime.trigger.tcp")


@register_trigger("tcp")
class TcpTrigger(BaseTrigger):
    def __init__(
        self,
        cfg: TriggerConfig,
        on_trigger,
        *,
        loop_runner: LoopRunner,
    ):
        super().__init__(cfg, on_trigger)
        self._server = None
        self._tasks = AsyncTaskOwner(
            logger=L,
            owner_name="tcp_trigger",
            loop_runner=loop_runner,
        )

    def start(self):
        if self._server:
            return
        self._tasks.spawn(self._serve())

    def stop(self):
        self._tasks.cancel_and_clear_local_tasks()

        async def _cleanup():
            if self._server:
                self._server.close()
                await self._server.wait_closed()
            self._server = None

        run_async_cleanup(
            _cleanup(),
            timeout=0.5,
            loop_runner=self._tasks.loop_runner,
        )
        L.info("TCP trigger socket stopped")

    async def _serve(self):
        self._server = await asyncio.start_server(
            self._handle_client, self.cfg.host, self.cfg.port, reuse_address=True
        )
        L.info(
            "TCP trigger socket listening on %s:%d word=%r",
            self.cfg.host,
            self.cfg.port,
            self.cfg.word,
        )
        async with self._server:
            await self._server.serve_forever()

    async def _handle_client(
        self, reader: asyncio.StreamReader, writer: asyncio.StreamWriter
    ):
        try:
            data = await reader.read(32)
            if data and data.strip() == self.cfg.word:
                self.on_trigger(writer.get_extra_info("peername"))
        finally:
            writer.close()
            await writer.wait_closed()
