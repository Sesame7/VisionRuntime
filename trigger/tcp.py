# -- coding: utf-8 --

import asyncio
import contextlib
import logging

from core import runtime
from trigger.base import BaseTrigger, TriggerConfig, register_trigger

L = logging.getLogger("sci_cam.trigger.tcp")


@register_trigger("tcp")
class TcpTrigger(BaseTrigger):
	def __init__(self, cfg: TriggerConfig, on_trigger):
		super().__init__(cfg, on_trigger)
		self._server = None

	def start(self):
		if self._server:
			return
		runtime.spawn_background_task(self._serve())

	def stop(self):
		async def _cleanup():
			if self._server:
				self._server.close()
				await self._server.wait_closed()
			self._server = None

		runtime.run_async(_cleanup(), timeout=0.5)
		L.info("TCP trigger socket stopped")

	async def _serve(self):
		self._server = await asyncio.start_server(self._handle_client, self.cfg.host, self.cfg.port, reuse_address=True)
		L.info("TCP trigger socket listening on %s:%d word=%r", self.cfg.host, self.cfg.port, self.cfg.word)
		async with self._server:
			await self._server.serve_forever()

	async def _handle_client(self, reader: asyncio.StreamReader, writer: asyncio.StreamWriter):
		try:
			data = await reader.read(32)
			if data and data.strip() == self.cfg.word:
				self.on_trigger(writer.get_extra_info("peername"))
		except Exception:
			L.warning("Trigger client error", exc_info=True)
		finally:
			with contextlib.suppress(Exception):
				writer.close()
				await writer.wait_closed()

	@property
	def is_running(self) -> bool:
		return bool(self._server)
