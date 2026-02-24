# Debug Tool Design Notes (tools/)

## 1. Goals and Boundaries

- Provide lightweight debug/integration tools independent of the production pipeline, intended for development or on-site troubleshooting.
- Tools live under `tools/` and are CLI-driven; keep them simple. They do not read production config files and do not write to production data/log directories.
- By default, logs are terminal-only. When needed, provide an optional `--log-file` to write logs to a user-specified path.

## 2. Tool List (Current)

- `tools/modbus_read.py`: read Modbus registers to validate Output Modbus mapping.
- `tools/modbus_write_trigger.py`: write/toggle Modbus trigger coil (CMD_TRIG_TOGGLE).
- `tools/modbus_sim_server.py`: Modbus TCP simulator for the v2.2 point table.
- `tools/tcp_listen.py`: TCP listener to inspect incoming trigger payloads.
- `tools/tcp_send_once.py`: send one TCP trigger payload.

## 3. modbus_read.py

- Purpose: read coils, discrete inputs, and input registers to validate register layout and result codes.
- Main arguments: `--host`, `--port`, `--device-id`, `--coil-offset`, `--di-offset`, `--ir-offset`, plus their counts.
- Quick run: `python tools/modbus_read.py --host 127.0.0.1 --port 1502`
- Output: print raw values to the terminal; no decoding.
- Behavior constraint: read-only; never write registers.

## 4. modbus_write_trigger.py

- Purpose: write/toggle a trigger coil to validate the Modbus trigger path.
- Main arguments: `--host`, `--port`, `--device-id`, `--coil-offset`, `--value` (optional).
- Quick run: `python tools/modbus_write_trigger.py --host 127.0.0.1 --port 1502`
- Behavior constraint: write-only; toggles by default if `--value` is omitted.

## 5. tcp_send_once.py

- Purpose: send a trigger word to a TCP listener (e.g., `tools/tcp_listen.py`).
- Main arguments: `--host`, `--port`, `--word`.
- Quick run: `python tools/tcp_send_once.py --host 127.0.0.1 --port 9000`
- Behavior constraint: single-shot sending by default.

## 6. tcp_listen.py

- Purpose: listen on a TCP port and print incoming payloads in hex/text for inspection.
- Main arguments: `--host`, `--port`, `--max-bytes`, `--max-preview`, `--encoding`, `--no-text`.
- Quick run: `python tools/tcp_listen.py --host 0.0.0.0 --port 9000`
- Behavior constraint: passive listener only.

## 7. modbus_sim_server.py

- Purpose: simulate a Modbus TCP server using the v2.2 point table (coils + DI + IR).
- Main arguments: `--host`, `--port`, `--coil-offset`, `--di-offset`, `--ir-offset`, `--heartbeat-ms`, `--poll-ms`, `--process-ms`, `--result`.
- Quick run: `python tools/modbus_sim_server.py --host 0.0.0.0 --port 1502`

## 8. General Notes

- No production config dependency: all parameters come from CLI with conservative defaults.
- Do not occupy production ports: host/port should be configurable; defaults should differ from production ports where practical.
- Do not modify production data: do not write into `data/`, `logs/`, or other production directories. If writing results is needed, require an explicit user-specified path.
- Dependencies: prefer the same dependency versions as the main project; if extra dependencies are required, document them in README/comments.
