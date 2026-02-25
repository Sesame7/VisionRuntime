# main and Directory Structure Design Notes

## 1. Goals and Principles

- Treat the directory layout itself as structured documentation: configuration entry points, core modules, extension modules, and data/log locations should be clear and easy to find.
- Design docs only describe responsibilities and entrypoints, and do not repeat internal logic of each module (see each module’s design doc).
- Keep only necessary persistence: during debugging, Output may write file logs; in production, only terminal logs; camera image persistence is configuration-driven.

## 2. Top-level Directory Overview

```text
.
├─config/            # main_*.yaml (single-load), detect_*.yaml, example_main_*.yaml, demo detect examples
├─core/              # runtime.py, worker.py, contracts.py, config.py, modbus_io.py
├─camera/            # base.py + adapters
├─detect/            # base.py + detection plugins
├─output/            # manager.py, channel implementations, web assets
├─trigger/           # base.py, gateway.py, trigger source implementations
├─tools/             # optional, scripts/helpers
├─data/              # runtime outputs (e.g., images/YYYY-MM-DD/* and same-day records.csv)
├─logs/              # optional debug/output logs; may be absent or empty in production
├─main.py            # entrypoint: load config and start SystemRuntime
└─docs/              # design docs (optional)
```

## 3. Configuration and Entrypoint

- Configurations are centralized under `config/`. Recommended main config naming is `main_<PROJECT>_<SITE>.yaml`; runtime loader behavior is “exactly one `main_*.yaml` in the selected config directory”. Detection parameters are carried by `detect_*.yaml`. Field meanings and validations are described in `config.md`.
- The entrypoint is responsible for: load config → initialize runtime components (`build_runtime`) → create enabled trigger sources → start runtime (workers/output/heartbeat/triggers). Global data contracts are defined by `core/contracts`.
- Communication endpoints (host/port/offsets) are configured under the `comm` section and referenced by trigger/output enablement.
- The top-level `imports` list is an optional preload hook for plugin modules / side-effect registrations (mainly custom extensions). Built-in camera/detect/trigger modules can be lazily imported by factory name, and output channels are wired directly in `build_runtime(...)`. Each item is a Python import path. An empty list is allowed; import failure causes startup failure.

## 4. Data and Logs

- Image dumping: when `camera.save_images` is enabled, images are written under `<runtime.save_dir>/images` (extension controlled by camera config). Only the Camera acquisition path may optionally save raw images; other modules do not write image files.
- Log/result files: production runtime prints to terminal only; during debugging or production result archiving, Output may write result summaries (for example `<runtime.save_dir>/images/YYYY-MM-DD/records.csv`) and optional channel-specific logs. Camera SDK auto-generated directories such as `DrvLog`/`System`/`SystemLog` are out of scope of this design.

## 5. Module Responsibility Anchors

- `core/`: threads and async runtime, system status and queues; see `runtime.md`.
- `camera/`: acquisition and saving implementation; config entrypoints are in `config.md`.
- `detect/`: detection plugins and preview generation; config entrypoints are in `config.md` plus per-plugin self-validation.
- `output/`: fan-out and channel implementations; the only module allowed to write result-summary/debug files.
- `trigger/`: trigger sources and filtering.

## 6. Plugin Pattern Conventions

- Applicable modules: camera / detect / trigger use registry + factory pattern. Output channels are currently wired directly in `build_runtime(...)` (not registry-based).
- Registration and imports: plugins self-register when the module is imported (e.g., via `register_*` decorators). Registries can be populated either by the optional main-config `imports` preload list or by factory lazy imports of built-in modules. Import failure or missing registration type → startup failure and list available types.
- Instantiation: instantiate via `create_*` factory based on config; missing required params or type mismatch must raise a clear error.
- Lifecycle: hot reload is not supported; plugins should not auto-rebuild after close; plugins must not create/close event loops on their own (must reuse `core/runtime` helpers).
- Config boundary: config only provides type/imports/paths/required params; field definitions are in `config.md`; plugin internal configs must validate themselves.

## 7. main Module Design Key Points

- Responsibilities: parse required CLI/environment parameters (e.g., optional `--config-dir`, `--log-level`), locate the unique `main_*.yaml`, call Loader to produce the config object, and optionally preload modules declared in `imports`.
- Startup sequence (current implementation): load config/imports → build detector/camera/runtime → create enabled trigger sources → `runtime.start(...)` (Camera/Detect Workers → Output channels → heartbeat → triggers) → enter blocking main loop.
- Shutdown sequence: receive signal/exception → request each module to stop → wait for Workers/channels to close → call `shutdown_loop()` to close the async loop → exit process; stop/join timeouts can reuse runtime defaults.
- Logging: follow global `log_level`; production prints to terminal only; during debugging, Output may optionally dump file logs (`logs/`).
- Exception handling: if config/import/startup fails at any step, exit immediately and print a clear error; uncaught runtime exceptions are caught by main, then trigger an orderly shutdown.
