"""
Test script for Ultralytics multi-dataset dataloader harmonization.

Validates HarmonizedClassMap + YOLODataset remapping WITHOUT training.
Prints a log to console and saves it to harmonize_test_log.txt.
Optionally saves annotated sample images to sample_images/.

Usage examples:
    # Basic check — just print class map and label stats
    python test_dataloader_harmonize.py --yaml-files data_A.yaml data_B.yaml

    # Inspect val split instead of train
    python test_dataloader_harmonize.py --yaml-files data_A.yaml data_B.yaml --split val

    # Also save sample images with bounding boxes drawn
    python test_dataloader_harmonize.py --yaml-files data_A.yaml data_B.yaml --visualize

    # Control how many samples are scanned for stats
    python test_dataloader_harmonize.py --yaml-files data_A.yaml data_B.yaml --stats-samples 1000
"""

import argparse
import os
import sys
from collections import Counter
from pathlib import Path

import yaml

# Make sure ultralytics package is importable from this directory
ROOT = Path(__file__).resolve().parent
sys.path.insert(0, str(ROOT))

from ultralytics.data.build import HarmonizedClassMap, build_yolo_dataset
from ultralytics.cfg import get_cfg
from ultralytics.utils import DEFAULT_CFG


# ── Helpers ───────────────────────────────────────────────────────────────────

def load_yaml(path):
    with open(path) as f:
        return yaml.safe_load(f)


def resolve_split_path(ypath, ydata, split):
    """Return the absolute image directory for the requested split from one YAML."""
    if split not in ydata:
        raise KeyError(f"Split '{split}' missing in {ypath}")
    val = ydata[split]
    if isinstance(val, list):
        if len(val) > 1:
            print(f"  [WARN] {Path(ypath).name} has multiple '{split}' entries — using the first.")
        val = val[0]
    val = str(val)
    if os.path.isabs(val):
        return val
    if "path" not in ydata:
        raise KeyError(f"YAML {ypath} has a relative '{split}' path but no 'path' key")
    return os.path.join(str(ydata["path"]), val)


def build_test_dataset(img_dirs, yaml_files, img_size, batch_size, harmonizer):
    """Build a YOLODataset with harmonizer active (augment=False, no caching)."""
    cfg = get_cfg(DEFAULT_CFG)
    cfg.imgsz = img_size
    cfg.rect = False
    cfg.cache = False
    cfg.single_cls = False
    cfg.classes = None
    cfg.task = "detect"
    cfg.fraction = 1.0

    # Minimal data dict — nc/names come from harmonizer
    data = {
        "names": {i: n for i, n in enumerate(harmonizer.train_names)},
        "nc": harmonizer.nc,
        "channels": 3,
    }

    from ultralytics.data.dataset import YOLODataset
    return YOLODataset(
        img_path=img_dirs,
        imgsz=img_size,
        batch_size=batch_size,
        augment=False,
        hyp=cfg,
        rect=False,
        cache=False,
        single_cls=False,
        stride=32,
        pad=0.5,
        prefix="[test] ",
        task="detect",
        classes=None,
        data=data,
        fraction=1.0,
        harmonizer=harmonizer,
    )


# ── Log sections ──────────────────────────────────────────────────────────────

def log_class_map(hcm, lines):
    lines += [
        "",
        "=" * 60,
        "  HARMONIZED CLASS MAP",
        "=" * 60,
        f"  Source YAMLs       : {hcm.source_names}",
        f"  All detected names : {hcm.all_names}",
        f"  Classes to train   : {sorted(hcm.classes_to_train)}",
        f"  Train indices      : {hcm.train_indices}",
        f"  Train names        : {hcm.train_names}",
        f"  nc (active)        : {hcm.nc}",
        "",
    ]
    for i, (src_name, dm, tc) in enumerate(
        zip(hcm.source_names, hcm.dataset_maps, hcm.per_dataset_train_classes)
    ):
        lines.append(f"  [{i}] {src_name}")
        lines.append(f"       classes_to_train : {sorted(tc)}")
        lines.append(f"       local→global remap: {dm}")
    lines.append("")


def log_dataset_stats(dataset, hcm, stats_samples, lines):
    """Scan up to stats_samples images per source and report class distributions."""
    n_sources = len(hcm.dataset_maps)
    per_src_raw = {i: Counter() for i in range(n_sources)}
    per_src_remapped = {i: Counter() for i in range(n_sources)}
    per_src_seen = {i: 0 for i in range(n_sources)}

    for i in range(len(dataset)):
        src_id = dataset.source_ids[i]
        if per_src_seen[src_id] >= stats_samples:
            if all(v >= stats_samples for v in per_src_seen.values()):
                break
            continue
        per_src_seen[src_id] += 1
        for box in dataset.labels[i]["cls"].flatten().astype(int):
            per_src_remapped[src_id][hcm.train_names[box]] += 1

    lines += [
        "=" * 60,
        "  LABEL DISTRIBUTION PER SOURCE",
        f"  (first {stats_samples} images per source)",
        "=" * 60,
    ]
    for src_id in range(n_sources):
        lines.append(f"  [{hcm.source_names[src_id]}]  (scanned {per_src_seen[src_id]} images)")
        lines.append(f"    Remapped class counts : {dict(per_src_remapped[src_id])}")
    lines.append("")

    total = Counter()
    for c in per_src_remapped.values():
        total.update(c)
    lines += ["  COMBINED totals (across all sources):"]
    for name, cnt in sorted(total.items()):
        lines.append(f"    {name:30s}: {cnt}")
    lines.append("")


def log_source_summary(dataset, hcm, lines):
    counts = Counter(dataset.source_ids)
    lines += [
        "=" * 60,
        "  IMAGE COUNT PER SOURCE",
        "=" * 60,
    ]
    for src_id, src_name in enumerate(hcm.source_names):
        lines.append(f"  {src_name:30s}: {counts.get(src_id, 0)} images")
    lines.append(f"  {'TOTAL':30s}: {len(dataset)} images")
    lines.append("")


# ── Visualisation (optional) ──────────────────────────────────────────────────

def visualize_samples(dataset, hcm, num_per_source, out_dir, lines):
    try:
        import matplotlib
        matplotlib.use("Agg")
        import matplotlib.pyplot as plt
        import matplotlib.patches as mpatches
    except ImportError:
        lines.append("[WARN] matplotlib not installed — skipping visualisation")
        return

    COLORS = ["red", "lime", "cyan", "yellow", "magenta", "orange", "white", "pink"]
    os.makedirs(out_dir, exist_ok=True)
    found = Counter()

    lines += [
        "=" * 60,
        f"  SAMPLE IMAGES  (saving to {out_dir}/)",
        "=" * 60,
    ]

    for i in range(len(dataset)):
        src_id = dataset.source_ids[i]
        if found[src_id] >= num_per_source:
            if all(found[s] >= num_per_source for s in range(len(hcm.source_names))):
                break
            continue

        item = dataset[i]
        img_tensor = item[0]  # CHW uint8
        labels_tensor = item[1]  # (N, 6): batch_idx | cls | x | y | w | h

        if labels_tensor.shape[0] == 0:
            continue  # skip backgrounds

        img_np = img_tensor.permute(1, 2, 0).cpu().numpy()
        fig, ax = plt.subplots(figsize=(10, 8))
        ax.imshow(img_np.astype("uint8"))
        ih, iw = img_np.shape[:2]

        for row in labels_tensor:
            cls_id = int(row[1].item())
            x, y, w, h = row[2].item(), row[3].item(), row[4].item(), row[5].item()
            name = hcm.train_names[cls_id] if cls_id < len(hcm.train_names) else str(cls_id)
            color = COLORS[cls_id % len(COLORS)]
            x1 = int((x - w / 2) * iw)
            y1 = int((y - h / 2) * ih)
            x2 = int((x + w / 2) * iw)
            y2 = int((y + h / 2) * ih)
            ax.add_patch(plt.Rectangle((x1, y1), x2 - x1, y2 - y1,
                                        edgecolor=color, facecolor="none", lw=2))
            ax.text(x1, max(y1 - 4, 0), f"{cls_id}:{name}", color=color, fontsize=10,
                    bbox=dict(facecolor="black", alpha=0.4, pad=1, edgecolor="none"))

        src_name = hcm.source_names[src_id]
        n = found[src_id]
        ax.set_title(f"Source: {src_name} | img #{i} | {labels_tensor.shape[0]} box(es)", fontsize=11)
        ax.axis("off")
        out_path = os.path.join(out_dir, f"{src_name}_{n}.png")
        plt.tight_layout()
        plt.savefig(out_path, dpi=100)
        plt.close()

        cls_names = [hcm.train_names[int(r[1])] for r in labels_tensor]
        lines.append(f"  Saved : {out_path}")
        lines.append(f"  Classes in image: {cls_names}")
        found[src_id] += 1

    for src_id, src_name in enumerate(hcm.source_names):
        if found[src_id] == 0:
            lines.append(f"  [WARN] No labeled samples found for source: {src_name}")
    lines.append("")


# ── Main ──────────────────────────────────────────────────────────────────────

def parse_args():
    p = argparse.ArgumentParser(description="Test Ultralytics harmonized dataloader (no training)")
    p.add_argument("--yaml-files", nargs="+", required=True,
                   help="Per-dataset YAML paths (with names + optional classes_to_train)")
    p.add_argument("--split", default="train", choices=["train", "val", "test"],
                   help="Dataset split to inspect")
    p.add_argument("--img-size", type=int, default=640)
    p.add_argument("--batch-size", type=int, default=4)
    p.add_argument("--stats-samples", type=int, default=500,
                   help="Max images per source to scan for label stats")
    p.add_argument("--visualize", action="store_true",
                   help="Save annotated sample images to sample_images/")
    p.add_argument("--num-vis-per-source", type=int, default=2,
                   help="How many sample images to save per source (requires --visualize)")
    p.add_argument("--log-file", default="harmonize_test_log.txt",
                   help="Path to save the output log")
    return p.parse_args()


def main():
    args = parse_args()
    lines = ["Ultralytics Harmonized Dataloader Test", ""]

    # Resolve YAML paths
    yaml_files = [
        str(Path(y).resolve()) if not os.path.isabs(y) else y
        for y in args.yaml_files
    ]
    yamls = [load_yaml(y) for y in yaml_files]

    # Build image directory list for the requested split
    img_dirs = [resolve_split_path(yp, yd, args.split) for yp, yd in zip(yaml_files, yamls)]
    lines.append(f"Split        : {args.split}")
    lines.append(f"Image dirs   : {img_dirs}")
    lines.append("")

    # Build harmonizer
    hcm = HarmonizedClassMap(yaml_files)
    log_class_map(hcm, lines)

    # Build dataset
    print("Building dataset (may scan/cache labels)...")
    dataset = build_test_dataset(img_dirs, yaml_files, args.img_size, args.batch_size, hcm)

    log_source_summary(dataset, hcm, lines)
    log_dataset_stats(dataset, hcm, args.stats_samples, lines)

    if args.visualize:
        visualize_samples(dataset, hcm, args.num_vis_per_source, "sample_images", lines)

    # Print + save log
    output = "\n".join(lines)
    print(output)
    log_path = Path(args.log_file)
    log_path.write_text(output, encoding="utf-8")
    print(f"\nLog saved → {log_path.resolve()}")


if __name__ == "__main__":
    main()
