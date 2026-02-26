import logging
import queue
import threading
import time
from dataclasses import dataclass
from datetime import datetime, timezone
from typing import Callable, Optional, Tuple

import numpy as np

from core.contracts import CaptureResult, OutputRecord
from core.queue_utils import drain_queue_nowait
from detect import encode_image_jpeg

L = logging.getLogger("vision_runtime.workers")


class BaseWorker:
    def __init__(self, name: str = ""):
        self.name = name or self.__class__.__name__
        self._stop_evt = threading.Event()
        self._thread: threading.Thread | None = None
        self._last_error: Exception | None = None

    def start(self):
        if self._thread is not None:
            raise RuntimeError(
                f"{self.name} is single-use; start() may only be called once"
            )
        self._thread = threading.Thread(target=self._run, daemon=True)
        self._thread.start()

    def stop(self, timeout: float = 2.0):
        self._stop_evt.set()
        if self._thread:
            self._thread.join(timeout=timeout)
            if self._thread.is_alive():
                L.warning("%s worker thread did not exit cleanly", self.name)

    def _run(self):
        try:
            self.run()
        except Exception as e:
            self._last_error = e
            L.exception("%s worker error", self.name)

    @property
    def is_alive(self) -> bool:
        return bool(self._thread and self._thread.is_alive())

    @property
    def has_started(self) -> bool:
        return self._thread is not None

    @property
    def last_error(self) -> Exception | None:
        return self._last_error

    def run(self):
        raise NotImplementedError


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
        trigger_queue: queue.Queue,
        detect_queue: "DetectQueueManager",
        result_sink: Callable[[OutputRecord, Optional[Tuple[bytes, str]]], None],
    ):
        super().__init__("CameraWorker")
        self.camera = camera
        self.trigger_queue = trigger_queue
        self.detect_mgr = detect_queue
        self.result_sink = result_sink

    def run(self):
        while not self._stop_evt.is_set():
            try:
                event = self.trigger_queue.get(timeout=0.1)
            except queue.Empty:
                continue
            try:
                source = event.source
                t0_val = event.monotonic_ms
                t0 = (
                    (t0_val / 1000.0)
                    if isinstance(t0_val, (int, float))
                    else time.perf_counter()
                )
                frame_id = int(event.trigger_seq or 0)
                if frame_id <= 0:
                    raise RuntimeError(
                        _worker_stage_context(
                            worker="CameraWorker",
                            stage="validate_trigger_event",
                            frame_id=frame_id,
                            source=source,
                            device_id=str(self.camera.cfg.device_index),
                        )
                    )
                start_dt = datetime.now(timezone.utc)
                triggered_at = event.triggered_at or start_dt
                res: CaptureResult
                try:
                    res = self.camera.capture_once(frame_id, triggered_at=triggered_at)
                except Exception as e:
                    raise RuntimeError(
                        _worker_stage_context(
                            worker="CameraWorker",
                            stage="capture",
                            frame_id=frame_id,
                            source=source,
                            device_id=str(self.camera.cfg.device_index),
                        )
                    ) from e
                device_id = str(res.device_id or self.camera.cfg.device_index)
                captured_at = res.captured_at or datetime.now(timezone.utc)
                if not res.success or res.image is None:
                    error_msg = res.error or "capture_failed"
                    L.warning(
                        "[%5s] src=%s dev=%s camera_error=%s",
                        frame_id,
                        source,
                        device_id,
                        error_msg,
                    )
                    msg = (
                        error_msg
                        if error_msg.startswith("Error:")
                        else f"Error: {error_msg}"
                    )
                    rec = _make_error_output_record(
                        frame_id=frame_id,
                        source=source,
                        device_id=device_id,
                        message=msg,
                        result_code="CAMERA_ERROR",
                        triggered_at=triggered_at,
                        captured_at=captured_at,
                        detected_at=datetime.now(timezone.utc),
                        duration_ms=(time.perf_counter() - t0) * 1000,
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
                try:
                    self.detect_mgr.enqueue(task)
                except Exception as e:
                    raise RuntimeError(
                        _worker_stage_context(
                            worker="CameraWorker",
                            stage="enqueue_detect",
                            frame_id=frame_id,
                            source=source,
                            device_id=device_id,
                        )
                    ) from e
            finally:
                self.trigger_queue.task_done()


class DetectQueueManager:
    def __init__(
        self,
        maxsize: int,
        result_sink: Callable[[OutputRecord, Optional[Tuple[bytes, str]]], None],
    ):
        self.queue = queue.Queue(maxsize=maxsize)
        self.result_sink = result_sink

    def enqueue(self, task: AcqTask):
        def _record_drop(dropped: AcqTask, remark: str = "queue_overflow"):
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
            self.result_sink(rec, None)

        try:
            self.queue.put_nowait(task)
            return
        except queue.Full:
            try:
                dropped = self.queue.get_nowait()
            except queue.Empty:
                dropped = None

            if dropped:
                _record_drop(dropped)
                L.warning(
                    "Detect queue overflow: drop_oldest frame=%s src=%s dev=%s qsize=%d/%d",
                    dropped.frame_id,
                    dropped.source,
                    dropped.device_id,
                    self.queue.qsize(),
                    self.queue.maxsize,
                )
                # Mark dropped task as done to keep queue counters consistent.
                self.queue.task_done()
            try:
                self.queue.put_nowait(task)
                return
            except queue.Full:
                # A concurrent producer/consumer race can still leave the queue full.
                _record_drop(task, remark="queue_overflow_incoming")
                L.warning(
                    "Detect queue overflow: drop_incoming frame=%s src=%s dev=%s qsize=%d/%d",
                    task.frame_id,
                    task.source,
                    task.device_id,
                    self.queue.qsize(),
                    self.queue.maxsize,
                )

    def clear(self):
        drain_queue_nowait(self.queue)


class DetectWorker(BaseWorker):
    def __init__(
        self,
        queue_mgr: DetectQueueManager,
        result_sink: Callable[[OutputRecord, Optional[Tuple[bytes, str]]], None],
        detector,
        timeout_ms: float = 2000.0,
        preview_enabled: bool = True,
    ):
        super().__init__("DetectWorker")
        self.queue_mgr = queue_mgr
        self.result_sink = result_sink
        self.detector = detector
        self.timeout_ms = timeout_ms
        self.preview_enabled = preview_enabled

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
                except Exception as e:
                    raise RuntimeError(
                        _worker_stage_context(
                            worker="DetectWorker",
                            stage="detect",
                            frame_id=task.frame_id,
                            source=task.source,
                            device_id=task.device_id,
                        )
                    ) from e
                detect_ms = (time.perf_counter() - det_start) * 1000
                rec = _make_detect_output_record(
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
                )
                preview_bytes = None
                preview_mime = None
                if (
                    self.preview_enabled
                    and overlay_img is not None
                    and rec.result in ("OK", "NG")
                ):
                    try:
                        preview_bytes, preview_mime = _encode_image(overlay_img)
                    except Exception as e:
                        raise RuntimeError(
                            _worker_stage_context(
                                worker="DetectWorker",
                                stage="preview_encode",
                                frame_id=task.frame_id,
                                source=task.source,
                                device_id=task.device_id,
                                result=rec.result,
                                result_code=rec.result_code,
                            )
                        ) from e
                overlay = (
                    (preview_bytes, preview_mime)
                    if (preview_bytes and preview_mime)
                    else None
                )
                # Emit per-frame timing at INFO for both OK/NG; TIMEOUT/ERROR at WARNING.
                log_fn = L.info if rec.result in ("OK", "NG") else L.warning
                log_fn(
                    "[%5s] src=%s dev=%s grab=%.2fms detect=%.2fms total=%.2fms result=%s code=%s msg=%s",
                    task.frame_id,
                    task.source,
                    task.device_id,
                    (task.grab_ms or 0.0),
                    detect_ms,
                    rec.duration_ms or 0.0,
                    rec.result,
                    rec.result_code or "",
                    rec.message,
                )
                try:
                    self.result_sink(rec, overlay)
                except Exception as e:
                    raise RuntimeError(
                        _worker_stage_context(
                            worker="DetectWorker",
                            stage="publish",
                            frame_id=task.frame_id,
                            source=task.source,
                            device_id=task.device_id,
                            result=rec.result,
                            result_code=rec.result_code,
                        )
                    ) from e
            finally:
                self.queue_mgr.queue.task_done()


def _make_error_output_record(
    frame_id: int,
    source: str,
    device_id: str,
    message: str,
    result_code: str | None,
    triggered_at: datetime | None,
    captured_at: datetime | None,
    detected_at: datetime | None,
    duration_ms: float,
) -> OutputRecord:
    msg = str(message or "")
    if not msg.startswith("Error:"):
        msg = f"Error: {msg}" if msg else "Error: error"
    code = str(result_code).strip() if result_code is not None else "ERROR"
    return OutputRecord(
        trigger_seq=frame_id,
        source=source,
        device_id=device_id,
        result="ERROR",
        triggered_at=triggered_at,
        captured_at=captured_at,
        detected_at=detected_at or datetime.now(timezone.utc),
        message=msg,
        data=None,
        result_code=code or "ERROR",
        duration_ms=duration_ms,
        remark=msg,
    )


def _make_detect_output_record(
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
) -> OutputRecord:
    result = "OK" if ok else "NG"
    msg = str(message or "")
    code = str(result_code).strip() if result_code is not None else None
    if timeout_ms and detect_ms is not None and detect_ms > timeout_ms:
        result = "TIMEOUT"
        msg = f"Timeout: detect {detect_ms:.1f} ms > {timeout_ms:.1f} ms"
        code = "TIMEOUT"
    elif result == "NG" and not msg.startswith("NG:"):
        msg = f"NG: {msg}" if msg else "NG"
    elif result == "OK" and not msg:
        msg = "OK"

    return OutputRecord(
        trigger_seq=frame_id,
        source=source,
        device_id=device_id,
        result=result,
        triggered_at=triggered_at,
        captured_at=captured_at,
        detected_at=detected_at or datetime.now(timezone.utc),
        message=msg,
        data=None,
        result_code=code or result,
        duration_ms=duration_ms,
        remark=msg,
    )


def _encode_image(img: np.ndarray) -> Tuple[bytes, str]:
    return encode_image_jpeg(img, quality=85, subsampling=1)


def _worker_stage_context(
    *,
    worker: str,
    stage: str,
    frame_id: int,
    source: str,
    device_id: str,
    result: str | None = None,
    result_code: str | None = None,
) -> str:
    parts = [
        f"{worker} stage={stage}",
        f"frame_id={frame_id}",
        f"source={source}",
        f"device_id={device_id}",
    ]
    if result is not None:
        parts.append(f"result={result}")
    if result_code:
        parts.append(f"result_code={result_code}")
    return " ".join(parts)


__all__ = [
    "AcqTask",
    "CameraWorker",
    "DetectQueueManager",
    "DetectWorker",
    "make_queue_overflow_record",
]
