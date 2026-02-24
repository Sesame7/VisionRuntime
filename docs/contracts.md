# contracts Design Notes (Single Source of Truth for Data and Channel Semantics)

## 1. Goals and Scope

- Serve as the single source of truth for cross-module data and channel semantics, guiding the implementation of `core/contracts.py` and interactions among modules (Trigger/Camera/Detect/Output/SystemRuntime).
- Contain only lightweight type definitions and conventions; do not introduce heavy dependencies (e.g. use `Any` instead of numpy types).
- Define time and channel semantics to reduce duplicated parsing and format divergence.

## 2. Naming and Dependencies

- Suggested filename: `core/contracts.py` (a single file is sufficient).
- Use `@dataclass` or `TypedDict`; default to dataclass for better type checking and IDE assistance.
- Image field runtime type uses `Any | None` to avoid hard dependency on numpy.
- BGR/grayscale shape, dtype, and related channel conventions are defined only in this file; other module docs reference them and do not repeat details.
- Forbidden: importing business modules at the contracts layer; allowed: `datetime`, `typing`, `dataclasses`.

## 3. Data Objects and Fields

### 3.1 TriggerEvent

- `trigger_seq: int`: global trigger sequence number, monotonically increasing.
- `source: str`: trigger source identifier (runtime-defined string, case-sensitive; current implementation uses values such as `"TCP"` / `"WEB"` / `"MODBUS"` / `"TEST"`).
- `triggered_at: datetime | None`: trigger occurrence time, UTC, millisecond precision (may be None at creation).
- `monotonic_ms: int`: monotonic milliseconds for debouncing and ordering.
- `payload: Any | None`: sanitized/clipped business summary carried by the source.

### 3.2 CaptureResult (camera-side output)

- `trigger_seq: int`: upstream trigger sequence for pipeline alignment.
- `source: str`
- `device_id: str`: camera identity (e.g. serial number or configured name).
- `success: bool`: capture success flag.
- `error: str | None`: preserve raw SDK error text; should be None on success.
- `image: Any | None`: mono8 (H x W) or bgr8 (H x W x 3); uses `Any` to avoid hard dependency.
- `path: str | None`: saved path if saved successfully; None if not saved or on failure.
- `timings: Dict[str, float] | None`: optional timing metrics (e.g. grab_ms).
- Timestamps:
  - `triggered_at: datetime | None`: trigger time (from Trigger).
  - `captured_at: datetime | None`: SDK grab completion time.

### 3.3 OutputRecord (for HMI/Modbus/CSV)

- `trigger_seq`
- `source: str`
- `device_id: str`
- `result: str`: `"OK"` / `"NG"` / `"TIMEOUT"` / `"ERROR"` etc., for external parsing and HMI display.
- `triggered_at: datetime | None`
- `captured_at: datetime | None`
- `detected_at: datetime | None`
- `message: str`: prefix classification and text.
- `data: dict | None`: optional structured supplement (may be trimmed for storage; excludes error info).
- `result_code: str | None`: optional machine-friendly code (e.g. `TIMEOUT`, `CAMERA_ERROR`).
- `duration_ms: float | None`: total end-to-end latency (ms).
- `remark: str`: free-form note (currently mirrors message in most cases).

## 4. Channel and Shape Conventions (Single Source of Truth)

- Color: 3-channel images are always BGR; if RGB is needed, explicitly convert at the usage side.
- Grayscale: 2D array H x W; do not use H x W x 1.
- dtype: default uint8; if special dtype is used, declare it in `data`.
  - Type hints: runtime uses `Any`.

## 5. Time Semantics

- All `datetime` use UTC timezone (or naive but treated as UTC), millisecond precision; serialize as ISO-8601 (`YYYY-MM-DDTHH:MM:SS.mmmZ`).
- `monotonic_ms` is only for internal interval/ordering and is not exposed externally.

## 6. Error and Status Conventions

- Do not invent error codes; preserve vendor/real error text in `error` or `message`, and use unified prefix classification (NG/Timeout/Error).
- Counting semantics (HMI/statistics): current `ResultStore` counts by `OutputRecord.result` (`OK` / `NG` / `TIMEOUT` / others). HMI reports `error = error_count + timeout_count`. On the PLC/Modbus side, business NG and Timeout/Error are all mapped to the NG signal; OK maps to the OK signal.
- Queue overflow: synthesized overflow results use an `"Error: queue overflow"` message prefix and `result_code="QUEUE_OVERFLOW"`. Timestamp preservation depends on where overflow occurs (trigger-queue overflow uses trigger time; detect-queue overflow preserves `captured_at` from the dropped task).
- `ok/success` semantics:
  - Camera side: `success` means whether acquisition succeeded.
  - Detect side: `ok` means whether the business decision passed.

## 7. Serialization and Transport

- For JSON, include only serializable fields; binary preview is delivered via a separate `/preview` endpoint.
- For CSV/Modbus and other persistence, use only scalar fields in `OutputRecord`; `data` can be ignored or trimmed.
