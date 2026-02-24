# core/worker Design Notes (camera/detect workers + queue)

## 1. Goals and Scope

- Implement the concrete worker loops for acquisition and detection.
- Own the Camera→Detect queue and its drop strategy.
- Convert camera/detect outputs into `OutputRecord` consistently.

## 2. Key Components

- `CameraWorker`: pulls triggers, grabs frames, enqueues detect tasks.
- `DetectWorker`: consumes detect tasks, runs detector, emits results.
- `DetectQueueManager`: queue + DropHead policy (drop oldest on full).
- `GlobalIdManager`: monotonic trigger ID that resets daily.

## 3. Flow (Simplified)

1) Trigger arrives → `CameraWorker` captures once  
2) Build `AcqTask` → enqueue into `DetectQueueManager`  
3) `DetectWorker` runs detector → `_to_output_record`  
4) Optional preview encoding → `OutputManager.publish`

## 4. Drop Strategy

- When queue is full, drop oldest and emit an ERROR record.
- Generated record uses `result="ERROR"` and `result_code="QUEUE_OVERFLOW"`.

## 5. Preview and Timing

- Preview encoding is attempted only for OK/NG if enabled.
- `duration_ms` is end-to-end time (trigger → detect done).

## 6. Notes

- Workers are single-threaded and rely on `BaseWorker` from `core/runtime`.
- No backpressure beyond the Camera→Detect queue.
