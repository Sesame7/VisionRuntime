# Detect Module Design Notes

## 1. Goals and Constraints

- Single-plugin architecture: one detection plugin per project, implemented against `detect/base.py` and any required third-party libraries.
- Single detection thread: currently supports one `DetectWorker`; no concurrency/multi-instance support yet (future extension).
- Queue decoupling: consumes the Camera→Detect queue; Detect→Output has no queue—after processing, results fan out directly.
- Independent configuration: the main config specifies plugin name and its config file; detector-specific validation (if any) is done inside the detector implementation.
- Unified color: 3-channel images are always BGR (OpenCV semantics). If RGB is needed, the plugin must explicitly convert.
- Preview is display-only: original images are saved by Camera if enabled; Detect does not persist original images.

## 2. Contracts and Validation

- OutputRecord fields, time semantics, and channel conventions use `core/contracts` as the single source of truth and are not repeated here. The pipeline converts detection output into OutputRecord before fan-out.
- Input validation: dtype/channel (mono8/bgr8), dimensions, ROI bounds are detector responsibilities.
- Result constraints: top-level keys are fixed; extensions go into `data`. Plugins must not modify `trigger_seq/source/device_id`.
- Exceptions/timeouts: the worker classifies elapsed detect time as `TIMEOUT`, but detector/preview-encode/publish exceptions currently follow fail-fast semantics (the `DetectWorker` thread exits and runtime health checks stop the service).

## 3. Preview and Overlay

- The detector may return an overlay image; the worker optionally encodes it as JPEG for HMI preview.
- Preview encoding is controlled by `detect.preview_enabled`. Overlay generation follows the same switch when creating the detector.
- `detect.preview_max_edge` limits preview image size by the longest edge (default `1280`, `0` disables downscale). Detection itself still runs on full resolution input.
- Detect does not persist original images; preview is display-only.

## 4. DetectWorker Behavior

- Long-running thread: pulls tasks from the queue; when idle, block with a short timeout; respond to stop.
- Per-frame timeout: evaluated by the worker; when exceeding the threshold, the result is marked `TIMEOUT`. The thread is not forcibly aborted.
- Queue overflow: handled upstream by DropHead; DetectWorker will not see frames that were dropped.
- Shutdown: no detector-level close is currently invoked by the worker; detector teardown (if any) should be handled by the creator.

## 5. Logging

- Detection information is returned upstream through the result (`message` / `data`). Terminal logs follow the global logging policy.

## 6. Future Extensions

- If multi-thread detection or more structured output is needed, implement by adding Workers or extending the `data` field, without changing the primary contract.
