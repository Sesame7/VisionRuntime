"""Data contracts for trigger, camera, detect, and output channels."""

from dataclasses import dataclass
from datetime import datetime
from typing import Any


@dataclass(slots=True)
class TriggerEvent:
    trigger_seq: int = 0
    source: str = ""
    triggered_at: datetime | None = None
    monotonic_ms: int = 0
    payload: Any | None = None


@dataclass(slots=True)
class CaptureResult:
    trigger_seq: int = 0
    source: str = ""
    device_id: str = ""
    success: bool = False
    error: str | None = None
    image: Any | None = None  # runtime np.ndarray
    triggered_at: datetime | None = None
    captured_at: datetime | None = None
    timings: dict[str, float] | None = None


@dataclass(slots=True)
class OutputRecord:
    trigger_seq: int = 0
    source: str = ""
    device_id: str = ""
    result: str = "ERROR"  # "OK"/"NG"/"TIMEOUT"/"ERROR"
    triggered_at: datetime | None = None
    captured_at: datetime | None = None
    detected_at: datetime | None = None
    message: str = ""
    data: dict[str, Any] | None = None
    # Extensions for current implementation
    result_code: str | None = None
    duration_ms: float | None = None
    remark: str = ""


__all__ = [
    "TriggerEvent",
    "CaptureResult",
    "OutputRecord",
]
