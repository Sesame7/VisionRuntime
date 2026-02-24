import contextlib
import logging
import queue
import threading
import time
from dataclasses import dataclass
from datetime import datetime, timezone
from typing import Callable, Optional, Tuple

import numpy as np

from core.contracts import CaptureResult, OutputRecord
from core.runtime import BaseWorker
from detect import encode_image_jpeg

L = logging.getLogger("vision_runtime.workers")


@dataclass
class AcqTask:
    frame_id: int
    triggered_at: datetime
    source: str
    device_id: str
    t0: float
    captured_at: datetime
    image: np.ndarray
    grab_ms: Optional[float] = None


class GlobalIdManager:
    def __init__(self):
        self._lock = threading.Lock()
        self._current_day = datetime.now(timezone.utc).date()
        self._counter = 0

    def next_id(self) -> int:
        with self._lock:
            today = datetime.now(timezone.utc).date()
            if today != self._current_day:
                self._current_day = today
                self._counter = 0
            self._counter += 1
            return self._counter

    def reset(self):
        with self._lock:
            self._current_day = datetime.now(timezone.utc).date()
            self._counter = 0


def make_queue_overflow_record(
    frame_id: int,
    source: str,
    device_id: str,
    triggered_at: datetime,
    captured_at: datetime,
    detected_at: datetime,
    t0: Optional[float] = None,
    remark: str = "queue_overflow",
) -> OutputRecord:
    duration_ms = ((time.perf_counter() - t0) * 1000) if t0 is not None else 0.0
    return OutputRecord(
        trigger_seq=frame_id,
        source=source,
        device_id=device_id,
        result="ERROR",
        triggered_at=triggered_at,
        captured_at=captured_at,
        detected_at=detected_at,
        message="Error: queue overflow",
        result_code="QUEUE_OVERFLOW",
        duration_ms=duration_ms,
        remark=remark,
    )


class CameraWorker(BaseWorker):
    def __init__(
        self,
        camera,
        id_manager: GlobalIdManager,
        trigger_queue: queue.Queue,
        detect_queue: "DetectQueueManager",
        result_sink: Callable[[OutputRecord, Optional[Tuple[bytes, str]]], None],
    ):
        super().__init__("CameraWorker")
        self.camera = camera
        self.id_manager = id_manager
        self.trigger_queue = trigger_queue
        self.detect_mgr = detect_queue
        self.result_sink = result_sink

    def run(self):
        while not self._stop_evt.is_set():
            try:
                event = self.trigger_queue.get(timeout=0.1)
            except queue.Empty:
                continue
            source = getattr(event, "source", "")
            t0_val = getattr(event, "monotonic_ms", None)
            t0 = (
                (t0_val / 1000.0)
                if isinstance(t0_val, (int, float))
                else time.perf_counter()
            )
            frame_id = getattr(event, "trigger_seq", None) or self.id_manager.next_id()
            start_dt = datetime.now(timezone.utc)
            triggered_at = getattr(event, "triggered_at", start_dt) or start_dt
            try:
                res: CaptureResult
                with self.camera.lock:
                    res = self.camera.capture_once(frame_id, triggered_at=triggered_at)
                device_id = str(
                    getattr(res, "device_id", "")
                    or getattr(getattr(self.camera, "cfg", None), "device_index", "")
                )
                captured_at = getattr(res, "captured_at", None) or datetime.now(
                    timezone.utc
                )
                if not res.success or res.image is None:
                    error_msg = res.error or "capture_failed"
                    msg = (
                        error_msg
                        if error_msg.startswith("Error:")
                        else f"Error: {error_msg}"
                    )
                    rec = _to_output_record(
                        frame_id=frame_id,
                        source=source,
                        device_id=device_id,
                        ok=False,
                        message=msg,
                        result_code="CAMERA_ERROR",
                        triggered_at=triggered_at,
                        captured_at=captured_at,
                        detected_at=datetime.now(timezone.utc),
                        duration_ms=(time.perf_counter() - t0) * 1000,
                        detect_ms=None,
                        timeout_ms=None,
                        preview=None,
                    )
                    self.result_sink(rec, None)
                    continue

                task = AcqTask(
                    frame_id=frame_id,
                    triggered_at=triggered_at,
                    source=source,
                    device_id=device_id,
                    t0=t0,
                    captured_at=captured_at,
                    image=res.image,
                    grab_ms=(res.timings or {}).get("grab_ms") if res.timings else None,
                )
                self.detect_mgr.enqueue(task, self.result_sink)
            except Exception:
                L.exception("Camera worker error")
            finally:
                with contextlib.suppress(Exception):
                    self.trigger_queue.task_done()


class DetectQueueManager:
    def __init__(self, maxsize: int):
        self.queue = queue.Queue(maxsize=maxsize)
        self.queue_overflow_count = 0

    def enqueue(
        self,
        task: AcqTask,
        result_sink: Callable[[OutputRecord, Optional[Tuple[bytes, str]]], None],
    ):
        def _record_drop(dropped: AcqTask, remark: str = "queue_overflow"):
            self.queue_overflow_count += 1
            now = datetime.now(timezone.utc)
            rec = make_queue_overflow_record(
                frame_id=dropped.frame_id,
                source=dropped.source,
                device_id=dropped.device_id,
                triggered_at=dropped.triggered_at,
                captured_at=dropped.captured_at,
                detected_at=now,
                t0=dropped.t0,
                remark=remark,
            )
            result_sink(rec, None)

        try:
            self.queue.put_nowait(task)
            return
        except queue.Full:
            try:
                dropped = self.queue.get_nowait()
            except queue.Empty:
                dropped = None

            if dropped:
                L.warning("Queue full, dropping frame %s", dropped.frame_id)
                _record_drop(dropped)
                # Mark dropped task as done to keep queue counters consistent.
                with contextlib.suppress(Exception):
                    self.queue.task_done()

    def clear(self):
        while True:
            try:
                self.queue.get_nowait()
            except queue.Empty:
                break
            finally:
                with contextlib.suppress(Exception):
                    self.queue.task_done()
        self.queue_overflow_count = 0


class DetectWorker(BaseWorker):
    def __init__(
        self,
        queue_mgr: DetectQueueManager,
        result_sink: Callable[[OutputRecord, Optional[Tuple[bytes, str]]], None],
        detector,
        timeout_ms: float = 2000.0,
        enable_preview: bool = True,
    ):
        super().__init__("DetectWorker")
        self.queue_mgr = queue_mgr
        self.result_sink = result_sink
        self.detector = detector
        self.timeout_ms = timeout_ms
        self.enable_preview = enable_preview

    def run(self):
        while not self._stop_evt.is_set():
            try:
                task = self.queue_mgr.queue.get(timeout=0.1)
            except queue.Empty:
                continue
            try:
                img = task.image
                det_start = time.perf_counter()
                try:
                    ok, message, overlay_img, result_code = self.detector.detect(img)
                except Exception:
                    L.exception("Detect error")
                    ok, message, overlay_img, result_code = (
                        False,
                        "Error: detection exception",
                        None,
                        "ERROR",
                    )
                detect_ms = (time.perf_counter() - det_start) * 1000
                rec = _to_output_record(
                    frame_id=task.frame_id,
                    source=task.source,
                    device_id=task.device_id,
                    ok=bool(ok),
                    message=str(message),
                    result_code=result_code,
                    triggered_at=task.triggered_at,
                    captured_at=task.captured_at,
                    detected_at=datetime.now(timezone.utc),
                    duration_ms=(time.perf_counter() - task.t0) * 1000,
                    detect_ms=detect_ms,
                    timeout_ms=self.timeout_ms,
                    preview=None,
                )
                preview_bytes = None
                preview_mime = None
                if (
                    self.enable_preview
                    and overlay_img is not None
                    and rec.result in ("OK", "NG")
                ):
                    try:
                        preview_bytes, preview_mime = _encode_image(overlay_img)
                    except Exception:
                        L.warning(
                            "Preview encode failed; disable detect.enable_preview or install dependencies",
                            exc_info=True,
                        )
                        preview_bytes, preview_mime = None, None
                overlay = (
                    (preview_bytes, preview_mime)
                    if (preview_bytes and preview_mime)
                    else None
                )
                # Emit per-frame timing at INFO for both OK/NG; TIMEOUT/ERROR at WARNING.
                log_fn = L.info if rec.result in ("OK", "NG") else L.warning
                log_fn(
                    "[%5s] grab=%.2fms detect=%.2fms total=%.2fms result=%s",
                    task.frame_id,
                    (task.grab_ms or 0.0),
                    detect_ms,
                    rec.duration_ms or 0.0,
                    rec.result,
                )
                self.result_sink(rec, overlay)
            finally:
                with contextlib.suppress(Exception):
                    self.queue_mgr.queue.task_done()


def _to_output_record(
    frame_id: int,
    source: str,
    device_id: str,
    ok: bool,
    message: str,
    result_code: str | None,
    triggered_at: datetime | None,
    captured_at: datetime | None,
    detected_at: datetime | None,
    duration_ms: float,
    detect_ms: float | None,
    timeout_ms: float | None,
    preview: bytes | None,
) -> OutputRecord:
    msg = str(message or "")
    result = "OK" if ok else "NG"
    code = result_code
    if timeout_ms and detect_ms is not None and detect_ms > timeout_ms:
        ok = False
        result = "TIMEOUT"
        msg = f"Timeout: detect {detect_ms:.1f} ms > {timeout_ms:.1f} ms"
        code = "TIMEOUT"
    if not ok and msg.startswith("Error:"):
        result = "ERROR"
    elif not ok and not msg.startswith("NG:") and not msg.startswith("Error:"):
        msg = f"NG: {msg}"

    rec = OutputRecord(
        trigger_seq=frame_id,
        source=source,
        device_id=device_id,
        result=result,
        triggered_at=triggered_at,
        captured_at=captured_at,
        detected_at=detected_at or datetime.now(timezone.utc),
        message=msg,
        data=None,
        result_code=code or ("OK" if ok else result),
        duration_ms=duration_ms,
        remark=msg,
    )
    return rec


def _encode_image(img: np.ndarray) -> Tuple[bytes, str]:
    return encode_image_jpeg(img, quality=85, subsampling=1)


__all__ = [
    "GlobalIdManager",
    "AcqTask",
    "CameraWorker",
    "DetectQueueManager",
    "DetectWorker",
    "make_queue_overflow_record",
]
