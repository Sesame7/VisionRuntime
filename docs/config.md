# Configuration Design Notes (core/config.py)

## 1. Goals and Boundaries

- Manage configurations through a unified Loader plus directory/naming conventions, reducing drift and duplicated parsing.
- Cover runtime/trigger/camera/output/detection algorithm parameters; the data contracts are still defined **only** in `core/contracts` as the single source of truth, and field definitions are NOT duplicated in config files.
- Configs do not support hot reload; changes require restart.

## 2. Configuration Layers (3 layers)

- **L1 Built-in defaults**: dataclass default values in code (queue capacity, Modbus heartbeat interval, grab timeout, etc.), project-wide and site-independent.
- **L2 Site main config (main)**: describes the complete runtime environment of the project + site (runtime/communication/camera/trigger/output/which detection algorithm to use and its config file).
- **L3 Detection algorithm parameters (detect)**: frequently tuned detection parameters (thresholds/ROI/switches, etc.). Hot reload is not supported in the current implementation (restart required).

## 3. Directory and Naming Conventions

- All configs live under `config/`; it stores only files and examples for the current project/site.
- Main config (single load): recommended naming `config/main_<PROJECT>_<SITE>.yaml`. Loader rule: there must be **exactly 1** `main_*.yml/yaml` in the selected config directory; 0 or >1 → startup failure, and list the discovered files.
- Main config template: `config/example_main_*.yaml` (e.g. `example_main_weigao_tray.yaml`), for copying only; it is never loaded.
- Detection config (actual use): `config/detect_<PROJECT>_<SITE>.yaml`, explicitly specified by `detect.config_file` in the main config (typically relative to `config/`; absolute paths are also supported and validated at startup).
- Detection config examples: `config/detect_overexposure.yaml` (never auto-loaded; examples only).
- Test main config lives in `config/tests/main_test.yaml` to avoid `main_*.yaml` collisions.

## 4. Loader Responsibilities and Flow

1) Construct the default config object (dataclass), apply known fields, and warn on unknown fields to prevent typos being silently ignored.  
2) Scan and load the main config `main_*.yaml`, bind blocks into Runtime/Camera/Trigger/Output/Detect metadata, etc.; on validation failure, provide “file + field path”.  
2.5) Optionally import extra plugin modules according to the `imports` list from the main config (string Python import paths). Import failure is a startup error (must include filename + failing import path). An empty/missing list is allowed because built-in camera/trigger/detect modules can be lazily imported when instantiated, and output channels are assembled directly in runtime wiring.  
3) Read `detect.config_file` from the main config; load the detection-parameter YAML (typically under `config/`) into `detect_params`.  
4) Pass the loaded config object to the runtime assembly path; use `detect.impl` to choose the detector implementation and pass `detect_params` to it.  
5) Detection params are not validated here; detectors should validate their own parameters if needed.

## 5. Configuration Format Rules (Restricted YAML Subset)

- Indent with 2 spaces; do not use anchors/aliases, tags, or custom types; booleans must be lowercase `true`/`false`; numbers are decimal; use a single document per file (do not use `---`).
- Config describes only behavior parameters and wiring (ports/IPs, queue sizes, which detection config to choose); do not duplicate data structure fields from `core/contracts` in config.
- Sensitive info (passwords/keys, etc.) must not be written into YAML; use environment variables or a separate secrets file.

## 6. Key Parameter Notes

- `comm`
  - `http`: `host`, `port`
  - `tcp`: `host`, `port`
  - `modbus`: `host`, `port`, `coil_offset`, `di_offset`, `ir_offset`, `heartbeat_ms`
    - Units/effective constraints: `heartbeat_ms` is milliseconds (startup validation requires `>0`, and `ModbusIO` further clamps to `>=100`); offsets require `>=0` at startup (and are also clamped to `>=0` in `ModbusIO`).
- `runtime`
  - `save_dir`: base directory for runtime outputs (images/CSV).
  - `max_runtime_s`: auto-stop after N seconds (unit: seconds; `0` = unlimited; startup validation requires `>=0`).
  - `detect_queue_capacity`: Camera→Detect queue capacity (detect task backlog; startup validation requires `>0`).
  - Trigger-event queue capacity is currently an internal runtime parameter (default `2`) and is not exposed in the YAML config.
  - `log_level`: global log level (unknown values fall back to `info`).
- `imports`
  - A top-level list; each element is a string Python import path (e.g. `"my_project.custom_detector"`).
  - Purpose: preload optional registries / side-effect registrations (mostly custom plugins); built-in modules usually do not need it. Import failures cause startup failure (error should include the specific path and exception text). Empty list is allowed.
- `camera`
  - Structure: `camera.type` selects the adapter, `camera.common` carries shared fields, and `camera.<type>` carries adapter-specific overrides (e.g. `camera.mock`, `camera.raspi`, `camera.opt`, `camera.hik`).
    - Merge order at load time: dataclass defaults -> `camera.common` -> selected `camera.<type>`.
  - `type`: adapter registry name.
  - `device_index`, `grab_timeout_ms`, `max_retry_per_frame`, `capture_output_format` (bgr8/mono8).
    - Units/effective constraints: `grab_timeout_ms` is milliseconds (startup validation requires `>0`; adapter/SDK semantics still apply). `max_retry_per_frame` requires `>0` at startup.
  - `save_images`, `save_ext` (naming/rules are described in the camera document; config only contains switches and paths).
    - `save_ext` is normalized at startup: empty -> `.bmp`; missing leading dot is auto-added.
    - `capture_output_format` is normalized to lowercase; empty -> `bgr8`.
  - Generic capture/adapter parameters: `width`, `height`.
    - Effective behavior: adapter-specific. For `raspi`, width/height override is applied only when both are non-zero; otherwise adapter defaults are used.
  - Camera-control related fields (primarily used by adapters that support runtime control, e.g. `raspi`): `ae_enable`, `awb_enable`, `exposure_us`, `analogue_gain`, `frame_duration_us`, `settle_ms`, `use_still`.
    - Units/effective constraints (raspi): `exposure_us` / `frame_duration_us` are microseconds and only applied when `>0`; `analogue_gain` only applied when `>0`; `settle_ms` is milliseconds and is clamped to `>=0`.
  - `image_dir`, `order`, `end_mode` (mock camera only; see camera doc).
    - Mock-camera constraints: `image_dir` is required; `order` / `end_mode` are validated against fixed enumerations in `camera.mock`.
- `trigger`
  - `debounce_ms`: trigger debounce window (unit: ms; startup validation requires `>=0`).
  - Global filters: `global_min_interval_ms`, `high_priority_cooldown_ms`, `high_priority_sources`/`low_priority_sources`, `ip_whitelist`.
    - Units/effective constraints: interval/cooldown are milliseconds and are clamped to `>=0` inside `TriggerGateway`; empty `ip_whitelist` disables allowlist enforcement.
    - `ip_whitelist` currently applies only to network-origin triggers (`TCP` / `WEB`) and does not block local/internal sources such as `MODBUS`.
  - Source names used by `high_priority_sources` / `low_priority_sources` are case-sensitive runtime identifiers (current implementation uses `"TCP"`, `"WEB"`, `"MODBUS"`, etc.).
  - Source enablement and required params are grouped by source:
    - `trigger.tcp`: `enabled`, `word` (`word` is UTF-8 encoded at startup; non-string values are stringified before encoding)
    - `trigger.modbus`: `enabled`, `poll_ms` (`poll_ms` unit: ms; startup validation requires `>0`, and `ModbusTrigger` clamps to `>=5`)
- `detect`
  - `impl`: detection plugin registry name.
  - `config_file`: points to `detect_*.yaml`, validate existence at startup.
  - `timeout_ms`: per-frame detection timeout (unit: ms).
    - Effective behavior: `0` disables timeout classification in the worker. Startup validation requires `>=0`.
  - Preview control: `preview_enabled`.
    - Effective behavior: when enabled, detector overlay generation and HMI preview encoding are both enabled in the runtime startup path.
- `output`
  - `output.hmi`: HMI output settings (e.g., `enabled`, `history_size`).
    - `history_size`: number of recent records kept in memory for HMI (startup validation requires `>0`).
  - `output.modbus`: Modbus output settings (e.g., `enabled`).
  - `write_csv`: enable CSV result writing.
  - Modbus IO: address/offset settings live under `comm.modbus`.

## 7. Effective Changes and Operations Notes

- Change application: restart required.  
- Error observability: on main/detect config load failure, provide clear errors (missing, duplicate main files, unknown fields). Main startup performs unified validation for key numeric fields across `runtime/camera/trigger/comm/detect`; some module-specific constraints still use additional clamping/validation inside runtime modules/adapters.  
- Examples and docs: add comments in `config/example_main_*.yaml` and demo detect configs; field meaning and default values should match the Loader’s fields.

## 8. Alignment with Other Modules

- Data contracts/channels/time semantics: `core/contracts` is the single source of truth.  
- Backpressure and queues: Detect→Output has no queue. Current runtime also uses a bounded trigger queue before CameraWorker (default capacity `2`, internal parameter, not YAML-configurable yet); `runtime.detect_queue_capacity` controls the Camera→Detect queue.  
- Async boundary: config contains only business parameters like network/timeouts/retries; threading/loop/shutdown strategy is uniformly managed by `SystemRuntime` (`core/runtime.py`) and shared loop helpers in `core/lifecycle.py`.  
- Trigger/Camera/Detect/Output read their own config blocks and do not parse other modules’ fields.
