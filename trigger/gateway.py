import contextlib
import logging
import queue
import threading
import time
from datetime import datetime, timezone
from typing import Callable, Optional, Set

from core.contracts import TriggerEvent

L = logging.getLogger("sci_cam.gateway")


class TriggerGateway:
	def __init__(
		self,
		trigger_queue: queue.Queue,
		debounce_ms: float = 10.0,
		min_interval_ms: float = 0.0,
		high_priority_cooldown_ms: float = 0.0,
		high_priority_sources: Optional[Set[str]] = None,
		low_priority_sources: Optional[Set[str]] = None,
		ip_whitelist: Optional[Set[str]] = None,
		on_overflow: Optional[Callable[[TriggerEvent], None]] = None,
	):
		self.trigger_queue = trigger_queue
		self.debounce_ms = max(debounce_ms, 0.0)
		self.min_interval_ms = max(min_interval_ms, 0.0)
		self.high_priority_cooldown_ms = max(high_priority_cooldown_ms, 0.0)
		self.high_priority_sources = high_priority_sources or set()
		self.low_priority_sources = low_priority_sources or set()
		# Empty whitelist means disabled; non-empty set enforces allowlist.
		self.ip_whitelist = set(ip_whitelist) if ip_whitelist else None
		self.last_accept_ts = 0.0
		self._last_high_pri_ts = 0.0
		self._lock = threading.Lock()
		self.on_overflow = on_overflow
		self._seq = 0

	def report_raw_trigger(self, source: str, payload: object | None = None, remote_ip: str | None = None) -> bool:
		now = time.perf_counter()
		with self._lock:
			if self.ip_whitelist is not None:
				remote_ip_str = self._normalize_ip(remote_ip)
				if remote_ip_str not in self.ip_whitelist:
					L.debug("Reject trigger from disallowed IP %s", remote_ip)
					return False

			if self.min_interval_ms and (now - self.last_accept_ts) * 1000 < self.min_interval_ms:
				L.debug("Global min interval drop from %s", source)
				return False

			if self.debounce_ms and (now - self.last_accept_ts) * 1000 < self.debounce_ms:
				L.debug("Debounce drop from %s", source)
				return False

			if self._is_low_priority_blocked(now, source):
				L.debug("Low priority blocked during high priority cooldown: %s", source)
				return False

			self.last_accept_ts = now
			if source in self.high_priority_sources:
				self._last_high_pri_ts = now

			self._seq += 1
			event = TriggerEvent(
				trigger_seq=self._seq,
				source=source,
				triggered_at=datetime.now(timezone.utc),
				monotonic_ms=int(now * 1000),
				payload=payload,
			)
		try:
			self.trigger_queue.put_nowait(event)
			return True
		except queue.Full:
			dropped = None
			with contextlib.suppress(queue.Empty):
				dropped = self.trigger_queue.get_nowait()
				with contextlib.suppress(Exception):
					self.trigger_queue.task_done()
			L.warning("Trigger queue full, dropping oldest and accepting %s", source)
			if dropped and self.on_overflow:
				with contextlib.suppress(Exception):
					self.on_overflow(dropped)
			try:
				self.trigger_queue.put_nowait(event)
				return True
			except queue.Full:
				L.warning("Trigger queue still full, drop %s", source)
			return False


	def _is_low_priority_blocked(self, now: float, source: str) -> bool:
		if not self.high_priority_sources or self.high_priority_cooldown_ms <= 0:
			return False
		if source in self.high_priority_sources:
			return False
		if self.low_priority_sources and source not in self.low_priority_sources:
			return False
		if self._last_high_pri_ts <= 0:
			return False
		return (now - self._last_high_pri_ts) * 1000 < self.high_priority_cooldown_ms

	def reset(self):
		with self._lock:
			self.last_accept_ts = 0.0
			self._last_high_pri_ts = 0.0
			self._seq = 0

	@staticmethod
	def _normalize_ip(remote_ip: object) -> str:
		if isinstance(remote_ip, tuple) and remote_ip:
			return str(remote_ip[0])
		return str(remote_ip) if remote_ip is not None else ""


__all__ = ["TriggerGateway"]
