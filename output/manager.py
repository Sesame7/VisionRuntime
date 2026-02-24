# -- coding: utf-8 --
"""OutputManager: persist results and fan out to output channels."""

import asyncio
import contextlib
import logging
import os
import queue
import threading
import time
from collections import deque
from datetime import datetime, timezone
from typing import Any, Deque, Iterable, List, Optional, Tuple, Protocol

from core.contracts import OutputRecord
from core import runtime


class OutputChannel(Protocol):
    def start(self): ...
    def stop(self): ...
    def publish(self, rec: OutputRecord, overlay: Optional[Tuple[bytes, str]]): ...
    def publish_heartbeat(self, ts: float | None = None): ...


L = logging.getLogger("vision_runtime.output.manager")


class ResultStore:
    L = logging.getLogger("vision_runtime.result_store")

    def __init__(self, base_dir: str, max_records: int = 10, write_csv: bool = True):
        self.base_dir = base_dir
        self.csv_root_dir = os.path.join(base_dir, "images")
        self._max_records = max_records
        self._records: Deque[OutputRecord] = deque(maxlen=max_records)
        self._latest_overlay: Optional[Tuple[bytes, str]] = None
        self._lock = threading.Lock()
        self.total_count = 0
        self.ok_count = 0
        self.ng_count = 0
        self.timeout_count = 0
        self.error_count = 0
        self._write_enabled = write_csv
        self._write_queue: Optional[queue.Queue[OutputRecord]] = (
            queue.Queue() if write_csv else None
        )
        self._writer_stop = threading.Event()
        self._writer_thread = (
            threading.Thread(target=self._writer_loop, daemon=True)
            if write_csv
            else None
        )
        if write_csv:
            os.makedirs(self.csv_root_dir, exist_ok=True)
        # Fresh start on each run; do not preload historical CSV into registers.
        if self._writer_thread:
            self._writer_thread.start()

    def stop(self, join_timeout: float = 1.0):
        """Stop background writer and flush pending CSV writes."""
        if not self._write_enabled or not self._writer_thread:
            return
        self._writer_stop.set()
        if self._write_queue:
            deadline = time.perf_counter() + max(join_timeout, 0.0)
            while (
                self._write_queue.unfinished_tasks > 0
                and time.perf_counter() < deadline
            ):
                time.sleep(0.05)
        self._writer_thread.join(timeout=join_timeout)
        if self._writer_thread.is_alive():
            self.L.warning("ResultStore writer thread did not exit cleanly")

    def submit(self, rec: OutputRecord, overlay: Optional[Tuple[bytes, str]]):
        with self._lock:
            self._records.appendleft(rec)
            if overlay is not None:
                self._latest_overlay = overlay
            self.total_count += 1
            if rec.result == "OK":
                self.ok_count += 1
            elif rec.result == "NG":
                self.ng_count += 1
            elif rec.result == "TIMEOUT":
                self.timeout_count += 1
            else:
                self.error_count += 1
        if self._write_enabled and self._write_queue:
            self._write_queue.put(rec)

    def reset(self):
        with self._lock:
            self._records.clear()
            self._latest_overlay = None
            self.total_count = 0
            self.ok_count = 0
            self.ng_count = 0
            self.timeout_count = 0
            self.error_count = 0
        # Drop pending writes to keep reset semantics clean.
        if self._write_queue:
            with contextlib.suppress(queue.Empty):
                while True:
                    self._write_queue.get_nowait()
                    self._write_queue.task_done()

    @property
    def latest_records(self) -> List[OutputRecord]:
        with self._lock:
            return list(self._records)

    @property
    def max_records(self) -> int:
        return self._max_records

    def latest_overlay(self) -> Optional[Tuple[bytes, str]]:
        with self._lock:
            return self._latest_overlay

    def stats(self):
        with self._lock:
            total = self.total_count
            ok = self.ok_count
            ng = self.ng_count
            timeout = self.timeout_count
            err = self.error_count
        pass_rate = (ok / total) if total else 0.0
        return {
            "total": total,
            "ok": ok,
            "ng": ng,
            "timeout": timeout,
            "error": err + timeout,  # TIMEOUT is counted as ERROR for HMI/PLC semantics
            "pass_rate": pass_rate,
        }

    def _writer_loop(self):
        """Background CSV writer to avoid blocking critical paths."""
        queue_ref = self._write_queue
        if not queue_ref:
            return
        while True:
            try:
                rec = queue_ref.get(timeout=0.5)
            except queue.Empty:
                if self._writer_stop.is_set():
                    break
                continue
            try:
                self._append_csv(rec)
            except Exception:
                self.L.warning("CSV write failed", exc_info=True)
            finally:
                with contextlib.suppress(Exception):
                    queue_ref.task_done()
            if self._writer_stop.is_set() and queue_ref.empty():
                break

    def _append_csv(self, rec: OutputRecord):
        csv_path = self._csv_path_for_record(rec)
        write_header = not os.path.exists(csv_path)
        t_date, t_time = _fmt_date_time(rec.triggered_at)
        save_time = _fmt_time(rec.detected_at)
        with open(csv_path, "a", encoding="utf-8") as f:
            if write_header:
                f.write(
                    "id,trigger_date,trigger_time,save_finish_time,result,result_code,duration_ms,remark\n"
                )
            f.write(
                f"{rec.trigger_seq},{t_date},{t_time},{save_time},{rec.result},{rec.result_code or ''},{(rec.duration_ms or 0.0):.3f},{rec.remark}\n"
            )

    def _csv_path_for_record(self, rec: OutputRecord) -> str:
        date_key = _record_date_key(rec)
        day_dir = os.path.join(self.csv_root_dir, date_key)
        os.makedirs(day_dir, exist_ok=True)
        return os.path.join(day_dir, "records.csv")


def _fmt_date_time(dt: Optional[datetime]) -> Tuple[str, str]:
    ref = dt or datetime.now(timezone.utc)
    return ref.date().isoformat(), ref.strftime("%H:%M:%S.%f")[:-3]


def _fmt_time(dt: Optional[datetime]) -> str:
    ref = dt or datetime.now(timezone.utc)
    return ref.strftime("%H:%M:%S.%f")[:-3]


def _record_date_key(rec: OutputRecord) -> str:
    ref = (
        rec.captured_at
        or rec.detected_at
        or rec.triggered_at
        or datetime.now(timezone.utc)
    )
    if ref.tzinfo is None:
        ref = ref.replace(tzinfo=timezone.utc)
    local_ref = ref.astimezone()
    return local_ref.date().isoformat()


class OutputManager:
    def __init__(self, store: ResultStore):
        self._store = store
        self._channels: List[OutputChannel] = []
        self._tasks: List[Any] = []
        self._tick_lock = threading.Lock()
        self._last_tick_ts: float | None = None

    def publish(self, rec: OutputRecord, overlay: Optional[Tuple[bytes, str]]):
        # Store as the single source of truth.
        self._store.submit(rec, overlay)
        # Fan out to channels, isolating failures.
        for ch in list(self._channels):
            publish = getattr(ch, "publish", None)
            if publish:
                try:
                    publish(rec, overlay)
                except Exception:
                    L.warning("Output channel publish failed", exc_info=True)

    def add_channel(self, channel: OutputChannel):
        self._channels.append(channel)

    def start(self):
        for ch in list(self._channels):
            with contextlib.suppress(Exception):
                ch.start()

    def stop(self):
        for ch in list(self._channels):
            with contextlib.suppress(Exception):
                ch.stop()
        self._drain_tasks()
        with contextlib.suppress(Exception):
            self._store.stop()
        self._tasks.clear()

    def reset(self):
        with contextlib.suppress(Exception):
            self._store.reset()

    def tick(self):
        ts = time.time()
        with self._tick_lock:
            self._last_tick_ts = ts
        for ch in list(self._channels):
            publish = getattr(ch, "publish_heartbeat", None)
            if publish:
                try:
                    publish(ts)
                except Exception:
                    L.warning("Output channel heartbeat failed", exc_info=True)

    def register_task(self, task: Any):
        """Register a background task created by a channel for cleanup on stop."""
        if task is None:
            return None
        self._tasks.append(task)
        return task

    # ---- Read API for HMI / Modbus (proxy to internal store) ----
    @property
    def latest_records(self):
        return self._store.latest_records

    @property
    def max_records(self) -> int:
        return self._store.max_records

    def latest_overlay(self):
        return self._store.latest_overlay()

    def stats(self):
        return self._store.stats()

    def last_tick_ts(self) -> float | None:
        with self._tick_lock:
            return self._last_tick_ts

    # ---- Internal helpers ----

    def _drain_tasks(self, timeout: float = 1.0):
        if not self._tasks:
            return
        asyncio_tasks: List[asyncio.Future] = []
        threadsafe_futs: List[Any] = []
        for t in list(self._tasks):
            with contextlib.suppress(Exception):
                cancel = getattr(t, "cancel", None)
                if callable(cancel):
                    cancel()
            if hasattr(t, "get_loop"):
                asyncio_tasks.append(t)  # type: ignore[arg-type]
            else:
                threadsafe_futs.append(t)

        if asyncio_tasks:

            async def _wait_tasks(tasks: Iterable[asyncio.Future]):
                with contextlib.suppress(Exception):
                    await asyncio.wait(tasks, timeout=timeout)

            with contextlib.suppress(Exception):
                runtime.run_async(_wait_tasks(asyncio_tasks), timeout=timeout)

        for fut in threadsafe_futs:
            if hasattr(fut, "done") and hasattr(fut, "result"):
                with contextlib.suppress(Exception):
                    if not fut.done():
                        fut.result(timeout=timeout)


__all__ = ["ResultStore", "OutputManager", "OutputChannel"]
