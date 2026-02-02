# Output Module Design Notes

## 1. Goals and Boundaries

- After detection completes, fan out to Output immediately. Output has no internal queue/backpressure/timeout; the only backpressure point is Camera→Detect (see `runtime`).
- Heartbeat: Output channels may expose a 1 Hz heartbeat indicating "channel online". This does not represent full pipeline health.
- Keep the latest result and a small optional history for HMI/debugging; do not manage health.
- Do not re-compress or resize previews; by default, upstream already provides a compressed preview.
- File writing boundary: only Output may optionally write logs/CSV-like result summaries; images are saved only by Camera; other modules do not write files.
- Config fields and defaults are defined in `config.md` and are not expanded here; log switches/paths are also configuration-driven. Communication endpoints (host/port/offsets) are under `comm`.

## 2. Interfaces and Responsibilities

- `OutputChannel`: `start()/stop()` manage resources; `publish(result)` is a synchronous entry. Any IO inside the channel must be made asynchronous by the channel itself (prefer helpers from `core/runtime` such as `spawn_background_task` or `run_async`). A channel must not call `shutdown_loop`.
- Optional: `publish_heartbeat(ts)` to expose an online heartbeat (e.g., toggle Modbus bit or update HMI indicator).
- `OutputManager`: holds the channel list and recent history (size from `runtime.history_size`). On `publish`, it receives OutputRecord, updates last/history/preview under a lock, then calls channels in order; exceptions must not block later channels. Track fire-and-forget tasks created by channels; on `stop()`, cancel/await them one by one before returning.

## 3. Channel Semantics (Summary)

- HMI: HTTP query only (no push). Endpoints `/status` `/preview/latest`. Data sources are OutputManager’s latest records/preview. Static resource `web/index.html`.
  - Web layout: left side is the overlay preview occupying most of the area; right side shows runtime time, manual trigger, counters (OK/NG/ERROR/TOTAL, where NG does not include ERROR), history table, and a runtime online indicator.
  - Display logic: when there is no recent result, summary areas show an explicit empty state (e.g., “Idle”). When `result=ERROR` or `TIMEOUT`, `/preview/latest` returns a red SVG placeholder for on-site visibility.
  - Empty-state API: `/preview/latest` returns 404 when no preview exists.
- Modbus: `ModbusOutput` maps OutputRecord to result codes/bits, and `ModbusIO` performs register writes only. Write 1x Discrete Inputs (OK/NG/ERR + toggles) and 3x Input Registers (timestamps/seq/codes). For PLC it only distinguishes OK/NG; Timeout/Error also map to NG (business NG and ERROR are unified as an NG signal; naming still uses NG/ERROR prefixes for upper-layer display). When emitting a result, update IR → DI bits → toggle result.
- CSV: asynchronously write result summaries, no higher than about ~5 Hz.

## 4. Integration Order

- Startup: SystemRuntime constructs channels and OutputManager → `start()` channels → begin accepting `publish`.
- Shutdown: stop channels first and clean up their async tasks; closing the async loop is done uniformly by SystemRuntime via `shutdown_loop()`; Output must not close the loop directly.
- `publish` is called synchronously by DetectWorker, ensuring the detect thread is not blocked by channel IO (channels are responsible for async-ifying their own IO).

## 5. Logging

- Default is terminal logs. If enabled in config, a channel may persist its own logs (such as result summaries). Paths/formats are channel-defined, but must not include raw preview data.

## 6. Test Points

- OutputManager: after multiple `publish` calls, last/history/preview remain consistent; an exception in one channel does not affect other channels.
- HMI: API data are consistent; when there is no preview, `/preview/latest` returns 404.
- Modbus/TCP/GPIO/CSV: each channel output follows semantics and does not block the detect thread; after channel stop, no dangling tasks remain.
