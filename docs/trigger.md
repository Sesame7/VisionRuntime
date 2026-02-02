# Trigger Module Design Notes

## 1. Goals and Boundaries

- Unify trigger sources such as TCP text, Modbus TCP, and HMI web triggers into internal `TriggerEvent` (fields and time semantics are defined in `contracts`).
- Provide global mutual exclusion and filtering (minimum interval, high-priority cooldown, IP whitelist), and support plugin-style extension of trigger source types.
- Decouple from acquisition/detection: only deliver compliant events to the upper-layer handler; lifecycle is managed uniformly by SystemRuntime.
- Config fields and defaults are defined in `config.md` and are not expanded here.
- Communication endpoints (host/port) are configured under `comm`, while trigger enablement and polling live under `trigger`.

## 2. Core Components

- `TriggerGateway`: filtering / sequence assignment / event generation; holds the handler callback; does not record logs or history.
- `BaseTrigger` plugin: implementations of each trigger source (tcp/modbus), unified `start/stop`.
  - Modbus trigger reads coils (CMD_TRIG_TOGGLE/CMD_RESET) via the shared ModbusIO server and toggles DI on acceptance; no Modbus client is used for DI updates.
- HMI web trigger bypasses BaseTrigger and calls `TriggerGateway.report_raw_trigger("WEB", ...)` directly.
- Multiple trigger sources can be enabled concurrently (e.g., TCP + Modbus).

## 3. Data and Contracts

- Event fields and time semantics follow `core/contracts.TriggerEvent` as the single specification, and are not repeated here.
- Filtering uses a monotonic clock; external exposure/logging uses UTC `triggered_at`. Payload must be clipped/desensitized inside the trigger source.

## 4. Filtering and Mutual Exclusion

- Rules: global minimum interval; reject low-priority sources during the high-priority cooldown window; optional IP whitelist.
- On rejection, do not dispatch upstream; on acceptance, increment `trigger_seq` and enqueue the event.
- Filter parameter definitions are in `config.md`.

## 5. TriggerGateway Behavior

- Thread safety: protect key state such as `trigger_seq` and latest trigger time with a lock; inside the lock, only do decision-making and metadata updates; call `handler(event)` outside the lock to avoid blocking.
- `report_raw_trigger(source, payload, remote_ip=None)`: input trigger source name + payload, perform filtering, and generate an event or reject.

## 6. BaseTrigger Plugin Conventions

- Interface: `start()` (bring up listening thread/task; whether it blocks must be stated), `stop()` (request stop and wait for exit; re-entrant).
- Responsibility: after receiving a physical/protocol trigger, call `gateway.report_raw_trigger(self.name, payload, remote_ip)`.
- Async dependency: if HTTP/Modbus/TCP async servers are needed, they must reuse the shared loop from `core/runtime` and must not create/close private loops.

## 7. Test Points

- Filter rule unit tests: global interval, high-priority cooldown, IP whitelist.
- Concurrency: when multiple threads call `report_raw_trigger`, `trigger_seq` must increase monotonically and filtering must be correct.
- End-to-end: after different trigger sources report, generate correct `TriggerEvent` and invoke the handler; time fields satisfy the contract.
