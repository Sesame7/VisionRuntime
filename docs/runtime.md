# core/runtime Design Notes (workers, queues, SystemRuntime)

## 1. Goals and Scope

- Provide a unified wrapper and lifecycle management for core synchronous threads such as camera acquisition and image detection.
- `SystemRuntime` is responsible for starting/stopping Workers/triggers/output, emitting a lightweight heartbeat, and uses shared async helpers from `core/lifecycle.py`.
- Non-goals: do not directly operate camera SDKs / industrial protocol libraries; do not implement Modbus/TCP/HTTP protocol details; do not manage cross-process behavior; no auto-restart in the current implementation.

## 2. BaseWorker

- Public interface: `start()` / `stop()`; `stop()` sets a flag and waits briefly for thread exit.
- Subclass hook: override `run()` with the worker loop; the loop should check the stop flag regularly.
- Current implementation location: `core/worker.py` (not `core/runtime.py`).

## 3. Queues and Backpressure

- Main runtime queues:
  - Trigger queue (`TriggerGateway` → `CameraWorker`), bounded (current default capacity `2`, internal runtime parameter); on overflow, drop the oldest trigger and synthesize an ERROR result.
  - Camera → Detect queue (`DetectQueueManager`), bounded DropHead; on overflow, drop the oldest detect task and synthesize an ERROR result.
- Detect → Output has no queue; once detection finishes, it fans out directly.
- DropHead synthesized result: for a dropped frame, generate an OutputRecord with `result="ERROR"` and `message` prefixed with `"Error: queue overflow"`; keep `captured_at` from the original result and set `detected_at` to the current time to distinguish counting.
- Queue items and result fields follow `contracts` and do not need to be expanded in this file.

## 4. Constraints for Concrete Workers (see `docs/worker.md`)

- CameraWorker: single-thread acquisition; serial `capture_once`; optional lightweight preprocessing before enqueue.
- DetectWorker: single-thread detection; when the queue is idle, block with a short timeout; per-frame timeout is classified in the worker as `TIMEOUT` in the emitted `OutputRecord`.
- Concurrency / multi-instance: currently one acquisition thread + one DetectWorker; backpressure relies entirely on the main DropHead queue.

## 5. Heartbeat (Lightweight)

- Emits a 1 Hz heartbeat tick to OutputManager from the `SystemRuntime.run()` loop (no dedicated heartbeat thread).
- Heartbeat only represents "runtime online"; it does not evaluate module health or restart logic.

## 6. SystemRuntime

- Responsibilities: coordinate start/stop order uniformly; manage the Trigger set and OutputManager; emit heartbeat ticks; provide reset/queue-drain behavior and health checks. Queue/Worker construction and wiring are performed by `build_runtime(...)`.
- Startup order (current implementation): load config and registries → `build_runtime(...)` constructs queues/workers/channels → start `CameraWorker`/`DetectWorker` → start output channels → start trigger sources.
- Shutdown order: stop trigger sources first → stop acquisition → stop detection/Output → close async services (gracefully close handles) → `shutdown_loop()` → exit.
- External interface: `start()` / `run()` / `request_stop()` / `stop()`.
- `SystemRuntime` is single-use (no restart support after `start()`/`stop()`).

## 7. Unified Lifecycle Conventions

- Ownership boundaries:
  - `main.py` owns process-level concerns (CLI/config/bootstrap) and top-level exception handling.
  - `SystemRuntime` owns runtime start/stop orchestration for workers, triggers, output channels, camera session entry/exit, queue draining, heartbeat ticking, and shared async-loop shutdown.
  - Output/trigger implementations own their protocol handles/servers, but must not own the global async loop lifecycle.
- Async loop ownership:
  - The shared asyncio loop is created via helpers in `core/lifecycle.py` and is closed only by `shutdown_loop()`.
  - Plugins/channels/triggers must not call `loop.close()` or stop/replace the shared loop directly.
- Start/stop contracts (component-level):
  - Adapters/channels/triggers should keep `start()` idempotent when practical (safe to call more than once without duplicating resources/tasks).
  - `stop()` should be best-effort and bounded (request cancellation/closure and wait briefly, but avoid unbounded blocking).
  - Re-entrant `stop()` should be tolerated when practical (especially for output/trigger adapters).
  - `SystemRuntime` itself is intentionally single-use and not restartable.
- Task ownership and cleanup:
  - Background tasks created by output-related services should be adopted through `OutputManager.adopt_task(...)` so `OutputManager.stop()` can perform final cancel/await cleanup.
  - If task registration is unavailable or fails, the component must retain local ownership and clean up its own fallback tasks.
  - Avoid double ownership of the same task (duplicate cancel/wait calls are usually safe but make shutdown behavior harder to reason about).
- Shutdown sequencing (runtime-level convention):
  - Stop trigger sources first (prevent new events entering queues).
  - Stop acquisition/detect workers next.
  - Drain pending detect tasks / synthesize terminal results as needed.
  - Stop output channels.
  - Finally call `shutdown_loop()` to close the shared async loop.
- Timeout policy:
  - All waits in shutdown paths should be bounded (thread join, task await, server cleanup).
  - Timeouts are operational safeguards, not a "hard kill" mechanism; on timeout, log a warning and continue shutdown progress when possible.
- Shutdown observability policy:
  - Routine shutdown timing diagnostics and pending-task summaries from `shutdown_loop()` are DEBUG-level by default.
  - Pending-task details remain available in DEBUG (including task/coroutine names) for field diagnostics when needed.

## 8. Logging and Error Semantics

- Start/stop: INFO; shutdown timeout: WARNING; uncaught Worker exceptions or startup failures: ERROR.
- In production, log only to the console (global `log_level`); file logging follows Output debugging strategies; this module does not write files.

## 9. Future Extensions

- Optional auto-restart, more Worker concurrency, finer-grained metrics, etc. are future evolutions and are not in the current implementation.
