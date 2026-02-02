# Camera Module Design Notes

## 1. Scope and Constraints

- Supported adapters: `opt`, `hik`, and `mock`. All are single-frame capture on soft trigger; no concurrent grabbing.
- Camera internal parameters are preset via vendor tools; this program does not modify exposure/gain/trigger mode.
- Responsibilities: select device, soft trigger → grab → convert to target output format (bgr8/mono8), optionally save to disk; does not manage queues/backpressure.
- Image channel/shape/dtype conventions follow `core/contracts` and are not repeated here.
- Configuration items (device selection, grab timeout, retries, format, saving switch/path, etc.) are defined in `config.md`.

## 2. CaptureResult (Aligned with contracts)

- Must populate fields such as `device_id`, `captured_at`, etc., and keep BGR/mono8 shapes consistent.

## 3. BaseCamera Abstraction

- Interface: `session()` (context manager) + `capture_once(idx)`; optional `get_stats()` is not used.
- `capture_once`: serialized and thread-safe; retry per configuration; flow is soft trigger → grab → SDK conversion → optional save → fill `CaptureResult` (save failure does not affect the return; just log a warning).
- Adapter-internal locking/buffering strategy is up to the implementation, but external semantics must remain stable.

## 4. Queues and Backpressure

- Camera only captures and returns a single frame; it does not maintain an acquisition queue. `trigger_seq` is bound by the upper layer before enqueue.
- Backpressure/frame dropping is handled uniformly by the main Camera→Detect queue (DropHead) in `runtime`.

## 5. Mock Camera (Optional)

- A mock adapter is available; it reads frames from a directory. After registration, switch via config; no changes to the main flow.
- Suggested config fields for mock (camera block):
  - `type: "mock"`
  - `image_dir`: relative to project root (or absolute path). Required.
  - `order`: `name_asc` / `name_desc` / `name_natural` / `mtime_asc` / `mtime_desc` / `random`.
  - `end_mode`: `loop` / `stop` / `hold`.
- Constraints: read-only; only current directory (no recursion). Supports common image extensions (`.jpg/.jpeg/.png/.bmp`).

## 6. Adapter Notes

- `opt`: loads vendor SDK at import time; errors are raised if the SDK library is missing.
- `hik`: loads vendor SDK at import time; errors are raised if the SDK library is missing.
- `mock`: uses OpenCV to read local images and respects `order`/`end_mode`.

## 7. Logging and Errors

- Terminal logs only: device discovery/open/close results, SDK text for trigger/grab failures, retry hints, save-failure path and error.
  Use `CaptureResult.success=False` and preserve the raw error text; do not invent error codes.
