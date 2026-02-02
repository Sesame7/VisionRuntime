# Detection Parameter Tuning Tool Design Notes

## 1. Goals and Boundaries

- Tune detection parameters offline, reusing production detection code and configuration formats, but without connecting to the running process/queues.
- A standalone application (pure GUI). All operations (select code/config, tune, export, launch batch runs) are done within the UI.
- Do not write into production data/log directories. Logs and errors are shown in an embedded UI view (optional export); by default, do not write to disk.

## 2. Inputs and Loading

- Detection plugin and config are selected via a front-end file picker: choose the detection code file and the corresponding `detect_*.yaml`.
- Image source: import a folder in the UI for batch loading, sorted by filename or manually selected.
- Loading flow reuses the production Loader/schema and import pipeline via `core/config`, import plugins based on `imports`.
- During plugin initialization, load/validate config per production logic, initialize model/templates; on failure, show detailed errors in the UI dialog.

## 3. Interactive Features

- Parameter panel: list tunable items in the config such as thresholds/ROI/switches, support input boxes/sliders, re-run detection after updates in real time.
- ROI adjustment: interactively box-select/drag ROI on the preview (if the plugin supports ROI), and write back into the parameter panel.
- Preview: show original/scaled image and overlays; present key output fields such as `ok/message/data`; support before/after comparison.
- Batch run: run on the selected directory with the current params, summarize OK/NG stats and latency distribution; allow drill-down into per-image results.

## 4. Output and Export

- Export tuned parameters as YAML (same format as production detect config); output path is chosen via the UI file picker.
- Optional export of batch report (CSV including filename, OK/NG, latency, message).

## 5. Performance and Safety

- Default single-thread/single-instance detection, focusing on interactive feedback; if acceleration is needed, optionally add a batch thread pool but not by default.
- Do not access production data/queues/loop; do not write to production directories; logs are shown only in UI, writing to disk requires explicit user path selection.

## 6. Validation

- Preview and result semantics follow `core/contracts` (BGR channels, time/field naming) to avoid RGB/shape mixing.
- Parameter edits must be validated immediately for type/range to prevent exporting illegal configs; exported files must pass the same schema validation.

## 7. Reserved

- Can be extended with multi-config comparison, parameter versioning, grid/random search, and other advanced tuning capabilities.
