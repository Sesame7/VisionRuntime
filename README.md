# Smart Camera

Config-driven industrial vision service with HMI (web), Modbus output, and pluggable camera/trigger/detector modules.

## Features

- Multi-camera drivers (mock, raspi, hik/opt when available)
- Trigger inputs (TCP, Modbus)
- Outputs (HMI web, Modbus, CSV)
- Config-driven runtime with YAML

## Quick Start

```bash
python -m venv .venv
source .venv/bin/activate
pip install -r requirements.txt
python main.py --config-dir config
```

## Configuration

Main config lives in `config/main_*.yaml`. Detect config is referenced by `detect.config_file`.

### Raspberry Pi camera (Picamera2)

```yaml
imports:
  - "camera.raspi"

camera:
  type: "raspi"
  width: 4056
  height: 3040
  ae_enable: false
  awb_enable: false
  exposure_us: 100000
  analogue_gain: 1.0
  frame_duration_us: 100000
  settle_ms: 200
  use_still: true
```

Notes:

- `exposure_us` is in microseconds.
- When `ae_enable` is false, `frame_duration_us` must be >= `exposure_us`.
- For headless systems, consider `opencv-python-headless`.

### Mock camera

```yaml
imports:
  - "camera.mock"

camera:
  type: "mock"
  image_dir: "data/mock_images"
```

## Run

```bash
python main.py --config-dir config
```

## License

MIT License. See `LICENSE`.
