# Container Setup Notes

## 1. Create and Start the Container

```bash
docker run -d \
  --net=host \
  --shm-size=4096m \
  --gpus all \
  --privileged \
  -e DISPLAY=$DISPLAY \
  -v /:/app \
  --device /dev/snd \
  --device /dev/dri \
  -v /dev/shm:/dev/shm \
  -v /tmp/.X11-unix/:/tmp/.X11-unix \
  --name ultralytics \
  --restart=always \
  -it \
  ultralytics/ultralytics:8.3.89 \
  bash
```

### What each argument means:

| Argument | Meaning |
|---|---|
| `docker run` | Create and start a new container |
| `-d` | Run in background (detached mode) |
| `--net=host` | Container shares the host's network — same IP, same ports |
| `--shm-size=4096m` | Shared memory size set to 4GB — required for PyTorch multi-worker dataloading |
| `--gpus all` | Give the container access to ALL GPUs on the host |
| `--privileged` | Give container full access to host devices (needed for GPU, display etc.) |
| `-e DISPLAY=$DISPLAY` | Pass the host display variable so GUI apps can show on screen |
| `-v /:/app` | Mount the ENTIRE host filesystem at `/app` inside container — so `/app/home/beast/...` = `/home/beast/...` on host |
| `--device /dev/snd` | Give access to host sound device |
| `--device /dev/dri` | Give access to host GPU render device |
| `-v /dev/shm:/dev/shm` | Mount host shared memory into container |
| `-v /tmp/.X11-unix/:/tmp/.X11-unix` | Mount X11 socket for display forwarding |
| `--name ultralytics` | Name the container `ultralytics` so you can reference it by name |
| `--restart=always` | Auto-restart container if it crashes or on system reboot |
| `-it` | Interactive + TTY — keeps the container running with a terminal |
| `ultralytics/ultralytics:8.3.89` | The Docker image to use (Ultralytics official image, version 8.3.89) |
| `bash` | Command to run inside the container on start |

---

## 2. Connect to the Running Container

```bash
docker exec -it ultralytics bash
```

Or connect via VS Code → Remote Explorer → Dev Containers → select `ultralytics`.

---

## 3. Clone the Custom Ultralytics Repo

```bash
git clone https://github.com/ArslanRobo123/ultralytics.git /usr/src/app
```

The custom code is on the `MultiDatasets` branch of `bnm-ai/ultralytics`.

---

## 4. Install PyTorch Nightly (Required for RTX 5090)

The default PyTorch in this container (`2.5.1+cu124`) does NOT support RTX 5090 (`sm_120`).
You MUST reinstall with the nightly build:

```bash
pip install --upgrade --pre torch torchvision torchaudio \
  --index-url https://download.pytorch.org/whl/nightly/cu128 \
  --force-reinstall
```

| Argument | Meaning |
|---|---|
| `--upgrade` | Upgrade even if already installed |
| `--pre` | Allow pre-release / nightly versions |
| `--index-url` | Use PyTorch's nightly download server with CUDA 12.8 builds |
| `--force-reinstall` | Force reinstall even if pip thinks it's already satisfied |

Verify GPU works after install:
```bash
python -c "import torch; print(torch.__version__); print('CUDA:', torch.cuda.is_available()); print('GPU:', torch.cuda.get_device_name(0))"
```

Expected output:
```
2.12.0.dev20260312+cu128
CUDA: True
GPU: NVIDIA GeForce RTX 5090
```

---

## 5. Install polars (Required for Saving Training Results)

```bash
pip install polars
```

Without this, training crashes after each epoch when trying to save `results.csv`.

---

## 6. Set PYTHONPATH

Every time you open a new terminal in the container, set this so Python can find the custom ultralytics package:

```bash
export PYTHONPATH=/app/home/beast/trainings/arslan/ultralytics
```

Or prefix every command with it:
```bash
PYTHONPATH=/app/home/beast/trainings/arslan/ultralytics yolo detect train ...
```

---

## 7. Dataset Paths Inside Container

Since the host filesystem is mounted at `/app`, all host paths are accessible as:

| Host path | Container path |
|---|---|
| `/home/beast/trainings/arslan/ultralytics/` | `/app/home/beast/trainings/arslan/ultralytics/` |
| `/home/beast/trainings/arslan/person/` | `/app/home/beast/trainings/arslan/person/` |
| `/home/beast/trainings/noor/.../7_to_10ft/` | `/app/home/beast/trainings/noor/.../7_to_10ft/` |

---

## 8. If Container is Stopped or Removed

If the container is stopped (not removed):
```bash
docker start ultralytics
docker exec -it ultralytics bash
```

If the container is removed (you need to recreate it):
- Re-run the `docker run` command from Step 1
- Re-install PyTorch nightly (Step 4) — pip installs don't persist after container removal
- Re-install polars (Step 5)
- The code and datasets are safe — they live on the host filesystem under `/app/home/beast/`

> **Important:** pip installs inside the container are NOT persistent if the container is deleted. Only files on mounted volumes (`/app/...`) persist. Consider saving the pip install commands somewhere or building a custom Docker image to avoid reinstalling every time.
