# core/runtime Design Notes (workers, queues, async helpers, SystemRuntime)

## 1. Goals and Scope

- Provide a unified wrapper and lifecycle management for core synchronous threads such as camera acquisition and image detection.
- `SystemRuntime` is responsible for starting/stopping Workers, emitting a lightweight heartbeat, and uses async helpers (`run_async` / `spawn_background_task` / `shutdown_loop`) from this module.
- Non-goals: do not directly operate camera SDKs / industrial protocol libraries; do not implement Modbus/TCP/HTTP details; do not manage across processes; no auto-restart in the first version.

## 2. BaseWorker

- Public interface: `start()` / `stop()`; `stop()` sets a flag and waits briefly for thread exit.
- Subclass hook: override `run()` with the worker loop; the loop should check the stop flag regularly.

## 3. Queues and Backpressure

- The only main queue: Camera → Detect, using a DropHead strategy (when full, drop the oldest and immediately generate an NG result to Output).
- Detect → Output has no queue; once detection finishes, it fans out directly.
- DropHead synthesized result: for a dropped frame, generate an OutputRecord with `result="ERROR"` and `message` prefixed with `"Error: queue overflow"`; keep `captured_at` from the original result and set `detected_at` to the current time to distinguish counting.
- Queue items and result fields follow `contracts` and do not need to be expanded in this file.

## 4. Constraints for Concrete Workers (see `docs/worker.md`)

- CameraWorker: single-thread acquisition; serial `capture_once`; optional lightweight preprocessing before enqueue.
- DetectWorker: single-thread detection; when the queue is idle, block with a short timeout; per-frame timeout returns `ok=False` + message.
- Concurrency / multi-instance: currently one acquisition thread + one DetectWorker; backpressure relies entirely on the main DropHead queue.

## 5. Heartbeat (Lightweight)

- Emits a 1 Hz heartbeat tick to OutputManager.
- Heartbeat only represents "runtime online"; it does not evaluate module health or restart logic.

## 6. SystemRuntime

- Responsibilities: create the main queue and Camera/Detect Workers; start/stop everything uniformly; manage the Trigger set and OutputManager; use `core/runtime` async helpers to start/stop async services; emit heartbeat ticks.
- Startup order: load config and registries → build queues and Workers → start Trigger → start Camera/Detect → start Output → start async services (via `run_async`/`spawn_background_task`, return handles after services are bound).
- Shutdown order: stop trigger sources first → stop acquisition → stop detection/Output → close async services (gracefully close handles) → `shutdown_loop()` → exit.
- External interface: `start()` / `stop()`.

## 7. Logging and Error Semantics

- Start/stop: INFO; shutdown timeout: WARNING; uncaught Worker exceptions or startup failures: ERROR.
- In production, log only to the console (global `log_level`); file logging follows Output debugging strategies; this module does not write files.

## 8. Future Extensions

- Optional auto-restart, more Worker concurrency, finer-grained metrics, etc. are future evolutions and are not in the current implementation.
