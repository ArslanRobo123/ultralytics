# Ultralytics — Multi-Dataset Harmonized Training

This is a custom fork of [Ultralytics](https://github.com/ultralytics/ultralytics) extended with **multi-dataset harmonized training** support. It allows training a single YOLO model on multiple datasets simultaneously, with automatic class ID remapping, per-dataset class filtering, and built-in dataloader verification tools.

---

## What Was Added

| Feature | Description |
|---|---|
| `HarmonizedClassMap` | Core class that merges multiple dataset YAMLs into a single unified class map |
| `classes_to_train` | Per-dataset YAML key to select only specific classes for training |
| Automatic class remapping | Local dataset class IDs are remapped to global contiguous train IDs in memory |
| Multi-path merging | Train/val image paths from all datasets are merged automatically |
| `test_dataloader_harmonize.py` | Standalone script to verify class remapping and data loading before training |
| Epoch-0 debug images | First 100 training images saved individually with `classID:ClassName` boxes drawn |
| `sample_images/` | Annotated sample images saved by the dataloader test script per dataset source |

---

## Modified Files

| File | Changes |
|---|---|
| `ultralytics/data/build.py` | Added `HarmonizedClassMap` class; `build_yolo_dataset()` accepts `harmonizer=` |
| `ultralytics/data/dataset.py` | `YOLODataset` stores harmonizer, tracks source IDs, applies remapping in `get_labels()` |
| `ultralytics/models/yolo/detect/train.py` | `get_dataset()` builds harmonizer; `preprocess_batch()` saves debug images |
| `ultralytics/models/yolo/detect/val.py` | `build_dataset()` and `get_dataloader()` pass harmonizer for standalone val |
| `ultralytics/cfg/default.yaml` | Added `harmonize_yaml_paths: ""` config key |

---

## How It Works

Each dataset has its own YAML with its own local class IDs (0-indexed). When you pass multiple YAMLs via `harmonize_yaml_paths`, the `HarmonizedClassMap`:

1. Reads all YAMLs and collects every class name
2. Applies `classes_to_train` filter per dataset (if specified)
3. Assigns new contiguous global train IDs to the active classes
4. Builds a per-dataset remap dict: `local_id → global_train_id`
5. At load time, each label's class ID is remapped in memory — label `.txt` files on disk are **never modified**

**Class identity is determined by name** — if two datasets both have a class called `person`, they get the same global train ID regardless of their local IDs.

> **Case sensitive:** `car` and `Car` are treated as different classes. Keep naming consistent across all YAMLs.

---

## Dataset YAML Format

Each dataset needs its own YAML file:

```yaml
# vehicle.yaml

# Absolute path to dataset root
path: /path/to/datasets/vehicle_data

# Splits (relative to path)
train: images/train
val:   images/val
test:  images/test          # optional

# Total number of classes in THIS dataset
nc: 7

# Local class names (0-indexed, must match your .txt label files)
names:
  0: license plate
  1: car
  2: bike
  3: truck
  4: bus
  5: van
  6: rickshaw

# Only train on these classes from this dataset.
# If omitted or all commented out → ALL classes are used.
classes_to_train:
  - license plate
  - truck
  # - bus        ← commented out = excluded
```

```yaml
# head_dataset.yaml

path: /path/to/datasets/head_data

train: images/train
val:   images/val

nc: 2

names:
  0: head
  1: person

classes_to_train:
  - head          # only head, skip person
```

```yaml
# coco_person.yaml  (full 80-class COCO, only train on person)

path: /path/to/coco

train: images/train
val:   images/val

nc: 80

names:
  0: person
  1: bicycle
  2: car
  # ... all 80 COCO classes ...
  79: toothbrush

classes_to_train:
  - person        # only person — all other 79 classes are dropped
```

**Rules:**
- `classes_to_train` values must exactly match strings in `names`
- Omit `classes_to_train` entirely to use all classes from that dataset
- Local class IDs in `.txt` files stay unchanged on disk — remapping is in-memory only

---

## Step 1 — Verify Dataloader Before Training

Always run this first to confirm class remapping and data loading are correct:

```bash
PYTHONPATH=/path/to/ultralytics python test_dataloader_harmonize.py \
  --yaml-files /path/to/dataset_a.yaml /path/to/dataset_b.yaml \
  --split train \
  --visualize
```

**All available arguments:**

| Argument | Default | Description |
|---|---|---|
| `--yaml-files` | required | Space-separated paths to all dataset YAMLs |
| `--split` | `train` | Split to inspect: `train`, `val`, `test` |
| `--stats-samples` | `500` | Max images per dataset to scan for label counts |
| `--visualize` | off | Save annotated sample images to `sample_images/` |
| `--num-vis-per-source` | `2` | Number of sample images to save per dataset |
| `--log-file` | `harmonize_test_log.txt` | Path to save the output log |

**Example output:**
```
HARMONIZED CLASS MAP
  Source YAMLs       : ['vehicle', 'head_dataset']
  All detected names : ['license plate', 'car', 'bike', ..., 'head']
  Classes to train   : ['license plate', 'head']
  Train indices      : [0, 1]
  Train names        : ['license plate', 'head']
  nc (active)        : 2

  [0] vehicle
       classes_to_train : ['license plate']
       local→global remap: {0: 0}
  [1] head_dataset
       classes_to_train : ['head']
       local→global remap: {0: 1}

IMAGE COUNT PER SOURCE
  vehicle        : 20595 images
  head_dataset   : 18601 images
  TOTAL          : 39196 images

LABEL DISTRIBUTION PER SOURCE
  [vehicle]  (scanned 500 images)
    Remapped class counts : {'license plate': 412}
  [head_dataset]  (scanned 500 images)
    Remapped class counts : {'head': 1716}
```

**`sample_images/` folder** — contains annotated PNG images saved per dataset, showing the remapped class labels drawn on bounding boxes. Use these to visually confirm correct class filtering.

---

## Step 2 — Training

```bash
PYTHONPATH=/path/to/ultralytics yolo detect train \
  model=yolo11n.pt \
  data=/path/to/dataset_a.yaml \
  harmonize_yaml_paths="/path/to/dataset_a.yaml,/path/to/dataset_b.yaml" \
  epochs=100 \
  batch=16 \
  imgsz=640 \
  project=/path/to/runs \
  name=my_run
```

**Key arguments:**

| Argument | Description |
|---|---|
| `model` | Pretrained weights. Auto-downloads if not present locally |
| `data` | Any one of your dataset YAMLs (used as entry point; paths overridden by harmonizer) |
| `harmonize_yaml_paths` | Comma-separated list of ALL dataset YAMLs — activates harmonization |
| `epochs` | Number of training epochs |
| `batch` | Batch size |
| `imgsz` | Input image size (default 640) |
| `project` | Root folder where run results are saved |
| `name` | Sub-folder name inside project |
| `exist_ok=True` | Reuse existing run folder instead of creating `name2`, `name3` etc. |
| `mosaic=0.0` | Disable mosaic augmentation (useful for debugging individual images) |

> **Note:** `harmonize_yaml_paths` is optional. If you omit it, the model trains normally using only the single dataset defined in `data=` — no harmonization or class remapping is applied. This means the fork is fully backwards-compatible with standard single-dataset Ultralytics training.

**What happens automatically:**
- Model head is resized to `nc` = number of active classes across all datasets
- Train/val image paths from all YAMLs are merged
- Class IDs remapped in memory — disk files untouched
- First 100 training images saved to `{project}/{name}/debug_epoch0/` with boxes drawn

**Supported models:**

| Model | Params | Speed |
|---|---|---|
| `yolov5nu.pt` | 2.5M | Fastest (v5 architecture) |
| `yolov8n.pt` | 3.2M | Fast (v8 architecture) |
| `yolov10n.pt` | 2.3M | Fast (v10 architecture) |
| `yolo11n.pt` | 2.6M | Fast (v11 architecture) |
| `yolo11s.pt` | 9.4M | Small |
| `yolo11m.pt` | 20M | Medium |
| `yolo11l.pt` | 25M | Large |
| `yolo11x.pt` | 56M | Extra-large — most accurate |

---

## Step 3 — Validation (Standalone)

Evaluate a trained model on the val split:

```bash
PYTHONPATH=/path/to/ultralytics yolo detect val \
  model=/path/to/runs/my_run/weights/best.pt \
  data=/path/to/dataset_a.yaml \
  harmonize_yaml_paths="/path/to/dataset_a.yaml,/path/to/dataset_b.yaml" \
  imgsz=640 \
  batch=16
```

---

## Step 4 — Inference / Prediction

Run a trained model on new images. **No `harmonize_yaml_paths` needed** — class names are baked into the `.pt` file at training time.

```bash
# On a folder of images
PYTHONPATH=/path/to/ultralytics yolo detect predict \
  model=/path/to/runs/my_run/weights/best.pt \
  source=/path/to/images/ \
  imgsz=640 \
  conf=0.25 \
  save=True \
  project=/path/to/runs \
  name=my_inference

# On a single image
PYTHONPATH=/path/to/ultralytics yolo detect predict \
  model=/path/to/runs/my_run/weights/best.pt \
  source=/path/to/image.jpg \
  conf=0.25 \
  save=True
```

**Verify class names baked into a trained model:**
```python
from ultralytics import YOLO
model = YOLO('/path/to/runs/my_run/weights/best.pt')
print(model.names)   # {0: 'license plate', 1: 'head'}
print(model.nc)      # 2
```

The model only knows the active classes — it will never predict classes excluded via `classes_to_train`.

---

## Debug Epoch-0 Images

During the first epoch of training, the first 100 images (after augmentation) are saved individually to:

```
{project}/{name}/debug_epoch0/img_0000.jpg
{project}/{name}/debug_epoch0/img_0001.jpg
...
{project}/{name}/debug_epoch0/img_0099.jpg
```

Each image has bounding boxes drawn with `classID:ClassName` labels. Use these to verify:
- Correct global class IDs are being used (0, 1, 2... not original local IDs)
- Class names match expectations
- Images from both datasets appear in the batches

> **Note:** If `mosaic=1.0` (default during training), each debug image shows 4 source images stitched together — this is normal. Add `mosaic=0.0` to the training command to see individual images.

---

## Sample Images

The `test_dataloader_harmonize.py` script saves annotated sample images to `sample_images/` when `--visualize` is passed:

```
sample_images/
  vehicle_0.png          ← image from vehicle dataset with boxes drawn
  vehicle_1.png
  head_dataset_0.png     ← image from head dataset with boxes drawn
  head_dataset_1.png
```

Class labels drawn show the **global train ID and name** (e.g. `0:license plate`, `1:head`), confirming remapping is correct before training starts.

---

## Quick Reference

```bash
# Set PYTHONPATH once
export PYTHONPATH=/path/to/ultralytics

# 1. Verify dataloader
python test_dataloader_harmonize.py \
  --yaml-files dataset_a.yaml dataset_b.yaml \
  --split train --visualize

# 2. Train
yolo detect train \
  model=yolo11n.pt \
  data=dataset_a.yaml \
  harmonize_yaml_paths="dataset_a.yaml,dataset_b.yaml" \
  epochs=100 batch=16 imgsz=640 \
  project=runs name=my_run

# 3. Validate
yolo detect val \
  model=runs/my_run/weights/best.pt \
  data=dataset_a.yaml \
  harmonize_yaml_paths="dataset_a.yaml,dataset_b.yaml"

# 4. Predict
yolo detect predict \
  model=runs/my_run/weights/best.pt \
  source=/path/to/images/ \
  conf=0.25 save=True
```

---

## Requirements

- Python 3.8+
- PyTorch with CUDA support
  - RTX 5090 / `sm_120` requires PyTorch nightly:
    ```bash
    pip install --pre torch torchvision --index-url https://download.pytorch.org/whl/nightly/cu128
    ```
- `pip install polars` (required for saving training results CSV)

---

## Troubleshooting

| Issue | Cause | Fix |
|---|---|---|
| `FileNotFoundError: images not found` | `path:` in YAML points to wrong location | Update `path:` in YAML to absolute path |
| `CUDA error: no kernel image` | PyTorch version doesn't support your GPU | Install nightly PyTorch (see Requirements) |
| `ModuleNotFoundError: polars` | polars not installed | `pip install polars` |
| Debug images show 4 merged scenes | Mosaic augmentation is on by default | Add `mosaic=0.0` to training command |
| Class X flagged as corrupt in cache | Stale `.cache` file from before harmonizer | Delete `.cache` files in labels folders and retrain |
| `car` and `Car` get different IDs | Class names are case-sensitive | Standardize names across all YAMLs |
| `exist_ok` error on rerun | Run folder already exists | Add `exist_ok=True` to training command |
