# core/worker Design Notes (camera/detect workers + queue)

## 1. Goals and Scope

- Implement the concrete worker loops for acquisition and detection.
- Own the Camera→Detect queue and its drop strategy.
- Convert camera/detect outputs into `OutputRecord` consistently.

## 2. Key Components

- `CameraWorker`: pulls triggers, grabs frames, enqueues detect tasks.
- `DetectWorker`: consumes detect tasks, runs detector, emits results.
- `DetectQueueManager`: queue + DropHead policy (drop oldest on full).
- `BaseWorker`: shared worker thread lifecycle wrapper (implemented in `core/worker.py`).

## 3. Flow (Simplified)

1) Trigger arrives → `CameraWorker` captures once  
2) Build `AcqTask` → enqueue into `DetectQueueManager`  
3) `DetectWorker` runs detector → build `OutputRecord`  
4) Optional preview encoding → `OutputManager.publish`

## 4. Drop Strategy

- When queue is full, drop oldest and emit an ERROR record.
- Generated record uses `result="ERROR"` and `result_code="QUEUE_OVERFLOW"`.

## 5. Preview and Timing

- Preview encoding is attempted only for OK/NG if enabled.
- `duration_ms` is end-to-end time (trigger → detect done).

## 6. Notes

- Workers are single-threaded and rely on `BaseWorker` in `core/worker.py`.
- `trigger_seq` comes from `TriggerGateway` (no worker-side global ID allocator).
- Detector/publish/preview-encode exceptions currently follow fail-fast semantics (the worker thread exits and runtime stops).
- No backpressure beyond the Camera→Detect queue.
