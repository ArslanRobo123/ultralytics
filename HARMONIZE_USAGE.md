# Multi-Dataset Harmonized Training — Usage Guide

---

## 1. Per-Dataset YAML Syntax

Each dataset gets its own YAML file. Place one per dataset root.

```yaml
# dataset_a.yaml

# Absolute path to dataset root
path: /usr/src/app/datasets/dataset_a

# Splits (relative to path)
train: images/train
val:   images/val
test:  images/test          # optional

# Number of classes in THIS dataset (all local classes)
nc: 7

# Local class names (0-indexed, must match your .txt label files)
names:
  0: license plate
  1: car
  2: bike
  3: truck
  4: bus
  5: van
  6: ricksahw

# Which classes from this dataset to include in training.
# If omitted or commented out → ALL classes from this dataset are used.
classes_to_train:
  - truck
  - bus
  - license plate
```

```yaml
# dataset_b.yaml

path: /usr/src/app/datasets/dataset_b

train: images/train
val:   images/val

nc: 2

names:
  0: head
  1: person

classes_to_train:
  - head
```

**Rules:**
- `classes_to_train` values must exactly match strings in `names`
- Omit `classes_to_train` to use all classes from that dataset
- Class IDs in `.txt` files are always local to their dataset — the harmonizer handles remapping automatically

---

## 2. Test Dataloader (No Training)

Validates that class remapping, filtering, and image loading work correctly before starting a real training run.

```bash
# Basic — print class map and label stats
PYTHONPATH=/usr/src/app python test_dataloader_harmonize.py \
  --yaml-files /path/to/dataset_a.yaml /path/to/dataset_b.yaml

# Inspect val split instead of train
PYTHONPATH=/usr/src/app python test_dataloader_harmonize.py \
  --yaml-files /path/to/dataset_a.yaml /path/to/dataset_b.yaml \
  --split val

# Also save annotated sample images to sample_images/
PYTHONPATH=/usr/src/app python test_dataloader_harmonize.py \
  --yaml-files /path/to/dataset_a.yaml /path/to/dataset_b.yaml \
  --visualize

# Scan more images for accurate label stats
PYTHONPATH=/usr/src/app python test_dataloader_harmonize.py \
  --yaml-files /path/to/dataset_a.yaml /path/to/dataset_b.yaml \
  --stats-samples 2000

# Full example with all options
PYTHONPATH=/usr/src/app python test_dataloader_harmonize.py \
  --yaml-files /usr/src/app/vehicle.yaml /usr/src/app/custom_yolo_7_to_10ft.yaml \
  --split train \
  --stats-samples 2000 \
  --visualize \
  --num-vis-per-source 3 \
  --log-file harmonize_test_log.txt
```

**Arguments:**

| Argument | Default | Description |
|---|---|---|
| `--yaml-files` | required | Space-separated paths to per-dataset YAMLs |
| `--split` | `train` | Which split to inspect: `train`, `val`, `test` |
| `--stats-samples` | `500` | Max images per source to scan for label counts |
| `--visualize` | off | Save annotated sample images to `sample_images/` |
| `--num-vis-per-source` | `2` | How many sample images to save per dataset |
| `--log-file` | `harmonize_test_log.txt` | Path to save the output log |

**Output log shows:**
- Unified class map (local → global → train ID remapping)
- Image count per dataset
- Class annotation counts per dataset
- (Optional) saved annotated PNGs per source

---

## 3. Training

```bash
PYTHONPATH=/usr/src/app yolo detect train \
  model=yolo11n.pt \
  data=/usr/src/app/vehicle.yaml \
  harmonize_yaml_paths="/usr/src/app/vehicle.yaml,/usr/src/app/custom_yolo_7_to_10ft.yaml" \
  epochs=100 \
  batch=16 \
  imgsz=640 \
  project=/usr/src/app/runs \
  name=my_run
```

**Key arguments:**

| Argument | Description |
|---|---|
| `model` | Pretrained weights or architecture. `yolo11n.pt` = lightest, auto-downloads |
| `data` | Any one of your dataset YAMLs (used as placeholder; paths are overridden by harmonizer) |
| `harmonize_yaml_paths` | Comma-separated list of ALL dataset YAMLs. This activates harmonization |
| `epochs` | Number of training epochs |
| `batch` | Batch size |
| `imgsz` | Input image size |
| `project` | Root folder for saving runs |
| `name` | Sub-folder name inside project |

**What happens automatically:**
- Model head is resized to `nc` = number of active classes
- Train/val image paths from all YAMLs are merged
- Class IDs are remapped in memory — label `.txt` files on disk are never modified
- First ~100 training images saved to `{project}/{name}/debug_epoch0/` for verification

**Available models (lightest → heaviest):**

| Model | Params | Notes |
|---|---|---|
| `yolo11n.pt` | 2.6M | Nano — fastest |
| `yolo11s.pt` | 9.4M | Small |
| `yolo11m.pt` | 20M | Medium |
| `yolo11l.pt` | 25M | Large |
| `yolo11x.pt` | 56M | Extra-large |

---

## 4. Validation (Standalone)

Run after training to evaluate a saved model on the val split.

```bash
PYTHONPATH=/usr/src/app yolo detect val \
  model=/usr/src/app/runs/my_run/weights/best.pt \
  data=/usr/src/app/vehicle.yaml \
  harmonize_yaml_paths="/usr/src/app/vehicle.yaml,/usr/src/app/custom_yolo_7_to_10ft.yaml" \
  imgsz=640 \
  batch=16
```

---

## 5. Inference / Testing on Images

Run a trained model on new images (no labels required).

```bash
# On a folder of images
PYTHONPATH=/usr/src/app yolo detect predict \
  model=/usr/src/app/runs/my_run/weights/best.pt \
  source=/path/to/images/ \
  imgsz=640 \
  conf=0.25 \
  save=True

# On a single image
PYTHONPATH=/usr/src/app yolo detect predict \
  model=/usr/src/app/runs/my_run/weights/best.pt \
  source=/path/to/image.jpg \
  conf=0.25 \
  save=True
```

> `harmonize_yaml_paths` is NOT needed for inference — the trained `.pt` file already has the harmonized class names baked in.

---

## 6. Quick Reference

```bash
# 1. Verify dataloader before training
PYTHONPATH=/usr/src/app python test_dataloader_harmonize.py \
  --yaml-files dataset_a.yaml dataset_b.yaml --visualize

# 2. Train
PYTHONPATH=/usr/src/app yolo detect train \
  model=yolo11n.pt data=dataset_a.yaml \
  harmonize_yaml_paths="dataset_a.yaml,dataset_b.yaml" \
  epochs=100 batch=16

# 3. Validate
PYTHONPATH=/usr/src/app yolo detect val \
  model=runs/my_run/weights/best.pt data=dataset_a.yaml \
  harmonize_yaml_paths="dataset_a.yaml,dataset_b.yaml"

# 4. Predict
PYTHONPATH=/usr/src/app yolo detect predict \
  model=runs/my_run/weights/best.pt source=/path/to/images/
```
