# -- coding: utf-8 --
"""OutputManager: persist results and fan out to output channels."""

import asyncio
import csv
import logging
import os
import queue
import threading
import time
from concurrent.futures import (
    CancelledError as ConcurrentCancelledError,
    Future as ConcurrentFuture,
    TimeoutError as ConcurrentTimeoutError,
)
from collections import deque
from datetime import datetime, timezone
from typing import Any, Protocol

from core.contracts import OutputRecord
from core.lifecycle import LoopRunner

L = logging.getLogger("vision_runtime.output.manager")


class OutputChannel(Protocol):
    def start(self): ...
    def stop(self): ...
    def publish(self, rec: OutputRecord, overlay: tuple[bytes, str] | None): ...
    def publish_heartbeat(self, ts: float | None = None): ...
    def raise_if_failed(self): ...


class ResultStore:
    _STOP_SENTINEL = None

    def __init__(self, base_dir: str, max_records: int = 10, write_csv: bool = True):
        self.base_dir = base_dir
        self.csv_root_dir = os.path.join(base_dir, "images")
        self._max_records = max_records
        self._records: deque[OutputRecord] = deque(maxlen=max_records)
        self._latest_overlay: tuple[bytes, str] | None = None
        self._lock = threading.Lock()
        self.total_count = 0
        self.ok_count = 0
        self.ng_count = 0
        self.timeout_count = 0
        self.error_count = 0
        self._write_queue: queue.Queue[OutputRecord | None] | None = (
            queue.Queue() if write_csv else None
        )
        self._writer_thread = (
            threading.Thread(target=self._writer_loop, daemon=True)
            if write_csv
            else None
        )
        self._writer_error: Exception | None = None
        if write_csv:
            os.makedirs(self.csv_root_dir, exist_ok=True)
        # Fresh start on each run; do not preload historical CSV into registers.
        if self._writer_thread:
            self._writer_thread.start()

    def stop(self, timeout: float = 1.0):
        thread = self._writer_thread
        q = self._write_queue
        if thread is None or q is None:
            return
        q.put(self._STOP_SENTINEL)
        thread.join(timeout=timeout)
        if thread.is_alive():
            L.warning("ResultStore writer thread did not exit within %.2fs", timeout)
            return
        self._writer_thread = None
        self._write_queue = None

    def submit(self, rec: OutputRecord, overlay: tuple[bytes, str] | None):
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
        if self._write_queue is not None:
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

    @property
    def latest_records(self) -> list[OutputRecord]:
        with self._lock:
            return list(self._records)

    @property
    def max_records(self) -> int:
        return self._max_records

    def latest_overlay(self) -> tuple[bytes, str] | None:
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

    def raise_if_failed(self):
        err = self._writer_error
        if err is not None:
            raise RuntimeError(
                f"ResultStore writer thread failed ({type(err).__name__})"
            ) from err
        thread = self._writer_thread
        if thread is not None and not thread.is_alive():
            raise RuntimeError("ResultStore writer thread stopped unexpectedly")

    def _writer_loop(self):
        try:
            queue_ref = self._write_queue
            if queue_ref is None:
                raise RuntimeError("writer queue missing")
            while True:
                item = queue_ref.get()
                try:
                    if item is None:
                        break
                    assert item is not None
                    self._append_csv(item)
                finally:
                    queue_ref.task_done()
        except Exception as exc:
            self._writer_error = exc
            L.exception("ResultStore CSV writer thread failed")

    def _append_csv(self, rec: OutputRecord):
        csv_path = self._csv_path_for_record(rec)
        write_header = not os.path.exists(csv_path)
        t_date, t_time = _fmt_date_time(rec.triggered_at)
        save_time = _fmt_time(rec.detected_at)
        with open(csv_path, "a", encoding="utf-8", newline="") as f:
            writer = csv.writer(f)
            if write_header:
                writer.writerow(
                    [
                        "id",
                        "trigger_date",
                        "trigger_time",
                        "save_finish_time",
                        "result",
                        "result_code",
                        "duration_ms",
                        "remark",
                    ]
                )
            writer.writerow(
                [
                    rec.trigger_seq,
                    t_date,
                    t_time,
                    save_time,
                    rec.result,
                    rec.result_code or "",
                    f"{(rec.duration_ms or 0.0):.3f}",
                    rec.remark or "",
                ]
            )

    def _csv_path_for_record(self, rec: OutputRecord) -> str:
        date_key = _record_date_key(rec)
        day_dir = os.path.join(self.csv_root_dir, date_key)
        os.makedirs(day_dir, exist_ok=True)
        return os.path.join(day_dir, "records.csv")


def _fmt_date_time(dt: datetime | None) -> tuple[str, str]:
    ref = _to_utc(dt)
    return ref.date().isoformat(), ref.strftime("%H:%M:%S.%f")[:-3] + "Z"


def _fmt_time(dt: datetime | None) -> str:
    ref = _to_utc(dt)
    return ref.strftime("%H:%M:%S.%f")[:-3] + "Z"


def _to_utc(dt: datetime | None) -> datetime:
    ref = dt or datetime.now(timezone.utc)
    if ref.tzinfo is None:
        ref = ref.replace(tzinfo=timezone.utc)
    return ref.astimezone(timezone.utc)


def _record_date_key(rec: OutputRecord) -> str:
    ref = rec.triggered_at or rec.captured_at or rec.detected_at
    return _to_utc(ref).date().isoformat()


class OutputManager:
    def __init__(self, store: ResultStore, loop_runner: LoopRunner):
        self._store = store
        self._loop_runner = loop_runner
        self._channels: list[OutputChannel] = []
        self._tasks: list[asyncio.Task[Any] | ConcurrentFuture[Any]] = []
        self._heartbeat_seq: int = 0

    def publish(self, rec: OutputRecord, overlay: tuple[bytes, str] | None):
        self._store.submit(rec, overlay)
        for ch in self._channels:
            ch.publish(rec, overlay)

    def add_channel(self, channel: OutputChannel):
        self._channels.append(channel)

    def start(self):
        for ch in self._channels:
            ch.start()

    def stop(self):
        def _run_stage(name: str, fn):
            try:
                fn()
            except Exception:
                L.exception("OutputManager stop stage failed: %s", name)

        def _stop_channels():
            for ch in self._channels:
                try:
                    ch.stop()
                except Exception:
                    L.exception("Output channel stop failed: %r", ch)

        _run_stage("channels", _stop_channels)
        _run_stage("tasks", self._drain_tasks)
        _run_stage("store", self._store.stop)

    def reset(self):
        self._store.reset()

    def tick(self):
        ts = time.time()
        self._heartbeat_seq += 1
        for ch in self._channels:
            ch.publish_heartbeat(ts)

    def raise_if_failed(self):
        self._store.raise_if_failed()
        for ch in self._channels:
            checker = getattr(ch, "raise_if_failed", None)
            if checker is None:
                continue
            checker()

    def adopt_task(self, task: Any) -> bool:
        if task is None:
            return False
        self._tasks.append(task)
        return True

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

    def heartbeat_seq(self) -> int | None:
        return self._heartbeat_seq or None

    # ---- Internal helpers ----

    def _drain_tasks(self, timeout: float = 1.0):
        tasks = self._tasks
        self._tasks = []
        if not tasks:
            return
        asyncio_tasks: list[asyncio.Task[Any]] = []
        threadsafe_futs: list[ConcurrentFuture[Any]] = []
        for t in tasks:
            t.cancel()
            if isinstance(t, asyncio.Task):
                asyncio_tasks.append(t)
            else:
                threadsafe_futs.append(t)

        if asyncio_tasks:
            try:
                self._loop_runner.run_async(
                    asyncio.wait(asyncio_tasks, timeout=timeout),
                    timeout=timeout,
                )
            except Exception:
                L.exception("OutputManager async task drain failed")

        for fut in threadsafe_futs:
            if fut.done():
                try:
                    fut.result()
                except ConcurrentCancelledError:
                    continue
                except Exception:
                    L.exception("OutputManager background future failed during drain")
                continue
            try:
                fut.result(timeout=timeout)
            except ConcurrentCancelledError:
                continue
            except ConcurrentTimeoutError:
                L.warning("OutputManager future drain timeout after %.2fs", timeout)
            except Exception:
                L.exception("OutputManager background future failed during drain")


__all__ = ["ResultStore", "OutputManager", "OutputChannel"]
