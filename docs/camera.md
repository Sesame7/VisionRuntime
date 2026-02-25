# Camera Module Design Notes

## 1. Scope and Constraints

- Supported adapters: `opt`, `hik`, `raspi`, and `mock`. In the current runtime, all operate in single-frame capture mode; no concurrent grabbing.
- Camera parameter behavior depends on adapter:
  - `opt` / `hik`: many settings are typically preset by vendor tools/SDK-side configuration.
  - `raspi`: Picamera2 controls (AE/AWB/exposure/gain/frame duration, etc.) may be applied from config during `session()`.
- Responsibilities: select device, soft trigger → grab → convert to target output format (bgr8/mono8), optionally save to disk; does not manage queues/backpressure.
- Image channel/shape/dtype conventions follow `core/contracts` and are not repeated here.
- Configuration items (device selection, grab timeout, retries, format, saving switch/path, etc.) are defined in `config.md`.

## 2. CaptureResult (Aligned with contracts)

- Must populate fields such as `device_id`, `captured_at`, etc., and keep BGR/mono8 shapes consistent.

## 3. BaseCamera Abstraction

- Interface: `session()` (context manager) + `capture_once(idx)`; optional `get_stats()` is not used.
- `capture_once`: serialized and thread-safe; retries follow configuration; the flow is soft trigger → grab → SDK conversion → optional save → populate `CaptureResult` (save failure does not affect the return value; log a warning only).
- Adapter-internal locking/buffering strategy is up to the implementation, but external semantics must remain stable.

## 4. Queues and Backpressure

- Camera only captures and returns a single frame; it does not maintain an acquisition queue. `trigger_seq` is bound by the upper layer before enqueue.
- Backpressure/frame dropping is handled uniformly by the main Camera→Detect queue (DropHead) in `runtime`.

## 5. Mock Camera (Optional)

- A mock adapter is available; it reads frames from a directory. After registration, switch via config; no changes to the main flow.
- Suggested config fields for mock (split camera config):
  - `camera.type: "mock"`
  - `camera.common`: shared fields such as `capture_output_format`, `save_images`, `save_ext`
  - `camera.mock.image_dir`: relative to the current working directory (`os.getcwd()`) or an absolute path. Required. In normal usage this is typically the project root because the service is started from the repo directory.
  - `camera.mock.order`: `name_asc` / `name_desc` / `name_natural` / `mtime_asc` / `mtime_desc` / `random`.
  - `camera.mock.end_mode`: `loop` / `stop` / `hold`.
- Constraints: read-only; only current directory (no recursion). Supports common image extensions (`.jpg/.jpeg/.png/.bmp`).

## 6. Adapter Notes

- `opt`: loads vendor SDK at import time; errors are raised if the SDK library is missing.
- `hik`: loads vendor SDK at import time; errors are raised if the SDK library is missing.
- `raspi`: depends on Picamera2; camera controls may be applied from config.
- `mock`: uses OpenCV to read local images and respects `order`/`end_mode`.

## 7. Logging and Errors

- Terminal logs only: device discovery/open/close results, SDK text for trigger/grab failures, retry hints, save-failure path and error.
  Use `CaptureResult.success=False` and preserve the raw error text; do not invent error codes.
