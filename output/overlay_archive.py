# -- coding: utf-8 --
from __future__ import annotations

import logging
import os
import queue
import threading
from dataclasses import dataclass
from datetime import datetime, timezone

from core.contracts import OutputRecord

L = logging.getLogger("vision_runtime.output.overlay_archive")

_INVALID_BATCH_CHARS = set('<>:"/\\|?*')
_RESERVED_BASENAMES = {
    "CON",
    "PRN",
    "AUX",
    "NUL",
    "COM1",
    "COM2",
    "COM3",
    "COM4",
    "COM5",
    "COM6",
    "COM7",
    "COM8",
    "COM9",
    "LPT1",
    "LPT2",
    "LPT3",
    "LPT4",
    "LPT5",
    "LPT6",
    "LPT7",
    "LPT8",
    "LPT9",
}
_STOP_SENTINEL = object()


@dataclass(slots=True)
class _WriteTask:
    batch_id: str
    filename: str
    data: bytes


def _to_utc(dt: datetime | None) -> datetime:
    ref = dt or datetime.now(timezone.utc)
    if ref.tzinfo is None:
        ref = ref.replace(tzinfo=timezone.utc)
    return ref.astimezone(timezone.utc)


def _ext_from_mime(mime: str | None) -> str:
    m = str(mime or "").strip().lower()
    if m == "image/png":
        return ".png"
    if m in {"image/jpeg", "image/jpg"}:
        return ".jpg"
    return ".jpg"


def validate_batch_name(batch_id: str) -> str:
    candidate = str(batch_id or "").strip()
    if not candidate:
        raise ValueError("batch name is required")
    if len(candidate) > 32:
        raise ValueError("batch name must be <= 32 characters")
    if candidate in {".", ".."}:
        raise ValueError("batch name is invalid")
    if candidate[-1] in {" ", "."}:
        raise ValueError("batch name cannot end with space or '.'")
    # Keep names portable across Windows/Linux deployments.
    base_name = candidate.split(".", 1)[0].upper()
    if base_name in _RESERVED_BASENAMES:
        raise ValueError(f"batch name is reserved: {base_name}")
    for ch in candidate:
        if ch in _INVALID_BATCH_CHARS:
            raise ValueError(f"batch name contains invalid character: {ch!r}")
        if ord(ch) < 32:
            raise ValueError("batch name contains control characters")
    return candidate


# Backward-compatible internal alias for legacy imports.
_validate_batch_name = validate_batch_name


class OverlayArchiveOutput:
    def __init__(
        self,
        *,
        base_dir: str,
        batch_state: "BatchState",
        only_ng: bool = True,
        queue_capacity: int = 256,
    ):
        self.base_dir = os.path.abspath(str(base_dir))
        self.batch_state = batch_state
        self.only_ng = bool(only_ng)
        self.queue_capacity = max(1, int(queue_capacity))
        self._queue: queue.Queue[object] = queue.Queue(maxsize=self.queue_capacity)
        self._writer_thread: threading.Thread | None = None

    def start(self):
        if self._writer_thread is not None:
            return
        os.makedirs(self.base_dir, exist_ok=True)
        os.makedirs(self._batch_dir(self.batch_state.current_batch_id()), exist_ok=True)
        self._writer_thread = threading.Thread(
            target=self._writer_loop,
            name="overlay_archive_writer",
            daemon=True,
        )
        self._writer_thread.start()

    def stop(self, timeout: float = 2.0):
        thread = self._writer_thread
        if thread is None:
            return
        self._enqueue_stop_sentinel()
        thread.join(timeout=timeout)
        if thread.is_alive():
            L.warning(
                "Overlay archive writer thread did not exit within %.2fs", timeout
            )
            return
        self._writer_thread = None

    def publish(self, rec: OutputRecord, overlay: tuple[bytes, str] | None):
        if overlay is None:
            return
        if self.only_ng and str(rec.result or "").upper() != "NG":
            return
        data, mime = overlay
        if not data:
            return
        batch_id = self.batch_state.current_batch_id()
        ts = _to_utc(rec.detected_at)
        seq = int(rec.trigger_seq or 0)
        filename = f"{ts.strftime('%Y%m%dT%H%M%S.%fZ')}_{seq:06d}{_ext_from_mime(mime)}"
        task = _WriteTask(
            batch_id=batch_id,
            filename=filename,
            data=bytes(data),
        )
        self._enqueue_task(task)

    def publish_heartbeat(self, ts: float | None = None):
        _ = ts
        return None

    def raise_if_failed(self):
        # Archive failures should not stop the runtime.
        return None

    def _batch_dir(self, batch_id: str) -> str:
        return os.path.join(self.base_dir, batch_id)

    def _enqueue_task(self, task: _WriteTask):
        try:
            self._queue.put_nowait(task)
            return
        except queue.Full:
            try:
                dropped = self._queue.get_nowait()
            except queue.Empty:
                dropped = None
            if dropped is not None:
                self._queue.task_done()
            try:
                self._queue.put_nowait(task)
            except queue.Full:
                L.warning(
                    "Overlay archive queue full; dropping image batch=%s file=%s",
                    task.batch_id,
                    task.filename,
                )

    def _enqueue_stop_sentinel(self):
        while True:
            try:
                self._queue.put_nowait(_STOP_SENTINEL)
                return
            except queue.Full:
                try:
                    _ = self._queue.get_nowait()
                    self._queue.task_done()
                except queue.Empty:
                    return

    def _writer_loop(self):
        while True:
            item = self._queue.get()
            try:
                if item is _STOP_SENTINEL:
                    return
                task = item
                assert isinstance(task, _WriteTask)
                self._write_task(task)
            except Exception:
                L.exception("Overlay archive writer failed")
            finally:
                self._queue.task_done()

    def _write_task(self, task: _WriteTask):
        batch_dir = self._batch_dir(task.batch_id)
        os.makedirs(batch_dir, exist_ok=True)
        out_path = os.path.join(batch_dir, task.filename)
        with open(out_path, "wb") as f:
            f.write(task.data)


class BatchState:
    def __init__(
        self,
        *,
        default_batch_id: str,
    ):
        self.default_batch_id = validate_batch_name(default_batch_id)
        self._batch_lock = threading.Lock()
        self._current_batch_id = self.default_batch_id

    def batch_status(self) -> dict[str, object]:
        with self._batch_lock:
            current = self._current_batch_id
        return {
            "current": current,
            "default": self.default_batch_id,
        }

    def set_batch(self, batch_id: str) -> tuple[bool, str]:
        try:
            candidate = validate_batch_name(batch_id)
        except ValueError as exc:
            return False, str(exc)
        with self._batch_lock:
            current = self._current_batch_id
            if candidate == current:
                L.info("Batch switch ignored (unchanged) current=%s", current)
                return True, f"batch unchanged: {candidate}"
            self._current_batch_id = candidate
        L.info("Batch switch from=%s to=%s", current, candidate)
        return True, f"batch set to {candidate}"

    def current_batch_id(self) -> str:
        with self._batch_lock:
            return self._current_batch_id


__all__ = ["BatchState", "OverlayArchiveOutput", "validate_batch_name"]
