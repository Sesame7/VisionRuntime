# main and Directory Structure Design Notes

## 1. Goals and Principles

- Make the directory itself a structured document: configuration entrypoints, core modules, extension modules, and data/log locations should be clear and easy to find.
- Design docs only describe responsibilities and entrypoints, and do not repeat internal logic of each module (see each module’s design doc).
- Keep only necessary persistence: during debugging, Output may write file logs; in production, only terminal logs; camera disk saving is organized by date directories.

## 2. Top-level Directory Overview

```text
.
├─config/            # main_*.yaml (single-load), detect_*.yaml, example_main_*.yaml, demo_* detect examples
├─core/              # runtime.py, worker.py, contracts.py, config.py, modbus_io.py
├─camera/            # base.py + adapters
├─detect/            # base.py + detection plugins
├─output/            # base.py, manager.py, channel implementations
├─trigger/           # base.py, gateway.py, trigger source implementations
├─tools/             # optional, scripts/helpers
├─data/              # camera image dumping (only written by camera module)
├─logs/              # only usable during output debugging; may be empty in production
├─main.py            # entrypoint: load config and start SystemRuntime
└─docs/              # design docs (optional, where this file set lives)
```

## 3. Configuration and Entrypoint

- Configurations are centralized under `config/`. The main config follows the single-file rule `main_<PROJECT>_<SITE>.yaml`; detection parameters are carried by `detect_*.yaml`. Field meanings and validations are described in `config.md`.
- The entrypoint is responsible for: load config → initialize SystemRuntime → start trigger/camera/detect/output. Global data contracts are defined by `core/contracts`.
- Communication endpoints (host/port/offsets) are configured under the `comm` section and referenced by trigger/output enablement.
- The top-level `imports` list is used to populate registries (camera/detect/output/trigger). Each item is a Python import path. An empty list is allowed; import failure causes startup failure.

## 4. Data and Logs

- Image dumping: `data/YYYYMMDD/HHMMSS_00001.ext` (or by `filename_pattern`). Only the Camera acquisition thread may optionally save raw images; other modules do not write images.
- Log/result files: production runtime prints to terminal only; during debugging, if file dumping is needed, only the Output module may write (logs/CSV, etc.) to `logs/` or a channel-specific path according to channel strategy. Camera SDK auto-generated directories such as `DrvLog`/`System`/`SystemLog` are out of scope of this design.

## 5. Module Responsibility Anchors

- `core/`: threads and async runtime, system status and queues; see `runtime.md`.
- `camera/`: acquisition and saving implementation; config entrypoints are in `config.md`.
- `detect/`: detection plugins and preview generation; config entrypoints are in `config.md` plus per-plugin self-validation.
- `output/`: fan-out and channel implementations; the only module allowed to write debug log files.
- `trigger/`: trigger sources and filtering.

## 6. Plugin Pattern Conventions

- Applicable modules: camera / detect / output / trigger all use registry + factory pattern.
- Registration and imports: plugins self-register when the module is imported (e.g., via `register_*` decorators). The main config’s `imports` list is used to populate registries. Import failure or missing registration type → startup failure and list available types.
- Instantiation: instantiate via `create_*` factory based on config; missing required params or type mismatch must raise a clear error.
- Lifecycle: hot reload is not supported; do not auto-rebuild after close; plugins are forbidden to create/close event loops by themselves (must reuse `core/runtime` helpers).
- Config boundary: config only provides type/imports/paths/required params; field definitions are in `config.md`; plugin internal configs must validate themselves.

## 7. main Module Design Key Points

- Responsibilities: parse required CLI/environment parameters (e.g., optional `--config-dir`, `--log-level`), locate the unique `main_*.yaml`, call Loader to produce the config object, preload module registries according to `imports`.
- Startup sequence: build SystemRuntime → start trigger/camera/detect/output → start async services (via `core/runtime` helpers) → enter a blocking main loop or signal listener.
- Shutdown sequence: receive signal/exception → request each module to stop → wait for Workers/channels to close → call `shutdown_loop()` to close the async loop → exit process; stop/join timeouts can reuse runtime defaults.
- Logging: follow global `log_level`; production prints to terminal only; during debugging, Output may optionally dump file logs (`logs/`).
- Exception handling: if config/import/startup fails at any step, exit immediately and print a clear error; uncaught runtime exceptions are caught by main, then trigger an orderly shutdown.
