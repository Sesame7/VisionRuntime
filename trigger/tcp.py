# -- coding: utf-8 --

import asyncio
from concurrent.futures import CancelledError as FutureCancelledError
import logging
import threading

from utils.lifecycle import (
    AsyncTaskOwner,
    LoopRunner,
    wait_task_done,
)
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
        self._serve_task = None
        self._stopping = False
        self._state_lock = threading.Lock()
        self._tasks = AsyncTaskOwner(
            owner_name="tcp_trigger",
            loop_runner=loop_runner,
        )

    def start(self):
        with self._state_lock:
            if self._stopping:
                return
            if self._server is not None or self._serve_task is not None:
                return
        self._tasks.clear_local_tasks()
        try:
            self._tasks.loop_runner.run_async(self._start_server(), timeout=1.0)
        except Exception:
            self.stop()
            raise

    def stop(self):
        with self._state_lock:
            if self._stopping:
                return
            serve_task = self._serve_task
            self._serve_task = None
            self._stopping = True
            server = self._server
            self._server = None
        try:
            if serve_task is not None:
                serve_task.cancel()
            self._tasks.cancel_local_tasks()

            async def _cleanup():
                if server is not None:
                    server.close()
                    await server.wait_closed()

            self._tasks.loop_runner.run_async(_cleanup(), timeout=0.5)
            wait_task_done(
                serve_task,
                timeout=0.5,
                label="tcp_trigger.serve",
                logger=L,
            )
            self._tasks.clear_local_tasks()
            L.info("TCP trigger socket stopped")
        finally:
            with self._state_lock:
                self._stopping = False

    def raise_if_failed(self):
        task = self._serve_task
        if task is None or not hasattr(task, "done") or not task.done():
            return
        try:
            err = task.exception()
        except (asyncio.CancelledError, FutureCancelledError):
            return
        if err is None:
            return
        raise RuntimeError(
            f"TcpTrigger stopped unexpectedly ({type(err).__name__})"
        ) from err

    async def _start_server(self):
        self._server = await asyncio.start_server(
            self._handle_client, self.cfg.host, self.cfg.port, reuse_address=True
        )
        L.info(
            "TCP trigger socket listening on %s:%d word=%r",
            self.cfg.host,
            self.cfg.port,
            self.cfg.word,
        )
        self._serve_task = self._tasks.register(
            asyncio.create_task(
                self._serve_forever(),
                name="tcp_trigger.serve_forever",
            )
        )

    async def _serve_forever(self):
        if self._server is None:
            return
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
