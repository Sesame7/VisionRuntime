# Debug Tool Design Notes (debug_tools/)

## 1. Goals and Boundaries

- Provide lightweight debug/integration tools independent of the production pipeline, intended for development or on-site troubleshooting.
- Tools live under `debug_tools/` and are CLI-driven; keep them simple. They do not read production config files and do not write to production data/log directories.
- By default, logs are terminal-only. Current tools do not provide a `--log-file` option.

## 2. Tool List (Current)

- `debug_tools/modbus_read.py`: read Modbus registers to validate Output Modbus mapping.
- `debug_tools/modbus_write_trigger.py`: write/toggle Modbus trigger coil (CMD_TRIG_TOGGLE).
- `debug_tools/modbus_write_hr_once.py`: write holding registers once (generic mode or `RECIPE_SLOT/RECIPE_SEQ` shortcut).
- `debug_tools/modbus_sim_server.py`: Modbus TCP simulator for the v3 point table.
- `debug_tools/tcp_listen.py`: TCP listener to inspect incoming trigger payloads.
- `debug_tools/tcp_send_once.py`: send one TCP trigger payload.
- `debug_tools/streamlit_hmi.py`: backup Streamlit HMI client (polls the built-in HTTP HMI API; legacy fallback script).

## 3. modbus_read.py

- Purpose: read coils, discrete inputs, input registers, and holding registers to validate register layout and result/recipe command areas.
- Main arguments: `--host`, `--port`, `--device-id`, `--coil-offset`, `--di-offset`, `--ir-offset`, `--hr-offset`, plus their counts.
- Quick run: `python debug_tools/modbus_read.py --host 127.0.0.1 --port 1502`
- Output: print raw values to the terminal; no decoding.
- Behavior constraint: read-only; never write registers.

## 4. modbus_write_trigger.py

- Purpose: write/toggle a trigger coil to validate the Modbus trigger path.
- Main arguments: `--host`, `--port`, `--device-id`, `--coil-offset`, `--value` (optional).
- Quick run: `python debug_tools/modbus_write_trigger.py --host 127.0.0.1 --port 1502`
- Behavior constraint: write-only; toggles by default if `--value` is omitted.

## 5. modbus_write_hr_once.py

- Purpose: write Modbus holding registers once, typically for recipe command registers (`RECIPE_SLOT` + `RECIPE_SEQ`).
- Main arguments: `--host`, `--port`, `--device-id`, `--hr-offset`; generic mode uses `--address --values ...`; recipe shortcut uses `--slot --seq`.
- Quick run (recipe): `python debug_tools/modbus_write_hr_once.py --host 127.0.0.1 --port 1502 --hr-offset 50 --slot 2 --seq 1 --verify`
- Quick run (generic): `python debug_tools/modbus_write_hr_once.py --host 127.0.0.1 --port 1502 --hr-offset 50 --address 0 --values 2 1 --verify`
- Behavior constraint: one-shot write, then exit.

## 6. tcp_send_once.py

- Purpose: send a trigger word to a TCP listener (e.g., `debug_tools/tcp_listen.py`).
- Main arguments: `--host`, `--port`, `--word`.
- Quick run: `python debug_tools/tcp_send_once.py --host 127.0.0.1 --port 9000`
- Behavior constraint: single-shot sending by default.

## 7. tcp_listen.py

- Purpose: listen on a TCP port and print incoming payloads in hex/text for inspection.
- Main arguments: `--host`, `--port`, `--max-bytes`, `--max-preview`, `--encoding`, `--no-text`.
- Quick run: `python debug_tools/tcp_listen.py --host 0.0.0.0 --port 9000`
- Behavior constraint: passive listener only.

## 8. modbus_sim_server.py

- Purpose: simulate a Modbus TCP server using the v3 point table (`0x/1x/3x/4x`) with trigger toggle + recipe command/ack flow.
- Main arguments: `--host`, `--port`, `--coil-offset`, `--di-offset`, `--ir-offset`, `--hr-offset`, `--heartbeat-ms`, `--poll-ms`, `--process-ms`, `--recipe-ms`, `--result`.
- Quick run: `python debug_tools/modbus_sim_server.py --host 0.0.0.0 --port 1502`

## 9. General Notes

- No production config dependency: all parameters come from CLI with conservative defaults.
- Do not occupy production ports: host/port should be configurable; defaults should differ from production ports where practical.
- Do not modify production data: do not write into `data/`, `logs/`, or other production directories. If writing results is needed, require an explicit user-specified path.
- Dependencies: prefer the same dependency versions as the main project; if extra dependencies are required, document them in README/comments.
- `debug_tools/streamlit_hmi.py` optional deps are installed manually: `pip install streamlit streamlit-autorefresh requests`.
- `debug_tools/streamlit_hmi.py` reads `/status` timing fields with `triggered_at_ms` priority and keeps legacy `triggered_at` fallback.
