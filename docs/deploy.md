# Deployment Guide

This guide covers Linux/Ubuntu and Windows deployments using a release package under a fixed root directory.

## Common configuration notes

- `config/` must contain exactly one `main_*.yaml`.
- The file referenced by `detect.config_file` must exist.
- Set `runtime.save_dir` to an absolute path to keep runtime data outside the release directory.
  - Linux example: `/opt/visionruntime/data`
  - Windows example: `C:\VisionRuntime\data`

## Linux / Ubuntu

### 1) Install system dependencies

```bash
sudo apt update
sudo apt install python3-aiohttp python3-yaml python3-pymodbus python3-opencv
```

### 2) Prepare directories

```bash
sudo mkdir -p /opt/visionruntime/{releases,config,data}
sudo chown -R $USER:$USER /opt/visionruntime
```

### 3) Unpack a release

```bash
VER="20260204_a7cf6bb"
mkdir -p /opt/visionruntime/releases/$VER
tar -xzf /path/to/VisionRuntime-$VER.tar.gz -C /opt/visionruntime/releases/$VER
cd /opt/visionruntime
ln -sfn releases/$VER current
```

If your archive contains a top-level directory (e.g. `VisionRuntime-20260204_a7cf6bb/`),
use this command so `/opt/visionruntime/current` points to the actual project root:

```bash
tar -xzf /path/to/VisionRuntime-$VER.tar.gz -C /opt/visionruntime/releases/$VER --strip-components=1
```

### 4) Start the service (manual)

If you use `/opt/visionruntime/config`:

```bash
cd /opt/visionruntime/current
python3 main.py --config-dir /opt/visionruntime/config
```

If you use the package's default `config/`:

```bash
cd /opt/visionruntime/current
python3 main.py
```

### 5) systemd unit (optional)

Create `/etc/systemd/system/visionruntime.service`:

```ini
[Unit]
Description=VisionRuntime Service
After=network.target

[Service]
Type=simple
User=YOUR_USER
WorkingDirectory=/opt/visionruntime/current
ExecStart=/usr/bin/python3 /opt/visionruntime/current/main.py --config-dir /opt/visionruntime/config
Restart=always
RestartSec=3

[Install]
WantedBy=multi-user.target
```

Enable and start:

```bash
sudo systemctl daemon-reload
sudo systemctl enable visionruntime
sudo systemctl start visionruntime
```

If you do not use `/opt/visionruntime/config`, remove `--config-dir` from `ExecStart`.

## Windows

### 1) Prepare directories

```powershell
New-Item -ItemType Directory -Force -Path C:\VisionRuntime\releases | Out-Null
New-Item -ItemType Directory -Force -Path C:\VisionRuntime\config | Out-Null
New-Item -ItemType Directory -Force -Path C:\VisionRuntime\data | Out-Null
```

### 2) Unpack a release

```powershell
$ver = "20260204_a7cf6bb"
$dst = "C:\VisionRuntime\releases\$ver"
New-Item -ItemType Directory -Force -Path $dst | Out-Null
Expand-Archive -Path C:\path\VisionRuntime-$ver.zip -DestinationPath $dst
```

If the zip contains a top-level directory (e.g. `VisionRuntime-20260204_a7cf6bb\`),
move its contents up so `C:\VisionRuntime\releases\$ver\main.py` exists.

### 3) Switch versions by renaming

```powershell
cd C:\VisionRuntime
Rename-Item current current_old -ErrorAction SilentlyContinue
Rename-Item releases\$ver current
```

### 4) Install dependencies (one-time)

```powershell
py -3.13 -m pip install -r C:\VisionRuntime\current\requirements.txt
```

### 5) Start the service (manual)

If you use `C:\VisionRuntime\config`:

```powershell
cd C:\VisionRuntime\current
py -3.13 main.py --config-dir C:\VisionRuntime\config
```

If you use the package's default `config\`:

```powershell
cd C:\VisionRuntime\current
py -3.13 main.py
```

### 6) Optional: Startup batch file

If you prefer a simple auto-start without services or Task Scheduler, create a batch
file and place a shortcut to it in the Windows Startup folder.

Example `start_visionruntime.bat`:

```bat
@echo off
set "WORKDIR=C:\VisionRuntime\current"
start "" /min /d "%WORKDIR%" cmd /c "py -3.13 main.py --config-dir C:\VisionRuntime\config"
exit /b
```

If you do not use `C:\VisionRuntime\config`, remove the `--config-dir` argument.

### 7) Optional: HMI desktop shortcut (Edge app mode)

You can create a Windows shortcut that opens the HMI in Edge app mode.
Set the shortcut target to:

```text
"C:\Program Files (x86)\Microsoft\Edge\Application\msedge.exe" --app=http://<HMI_HOST>:8000
```

In the shortcut properties, you may set "Run" to "Maximized" if desired.
