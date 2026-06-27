import csv
import hashlib
import math
from pathlib import Path

import numpy as np
from PIL import Image


MIN_DEPTH_M = 0.10
DEPTH_SUMMARY_METRICS = (
    "valid_pixel_ratio",
    "abs_rel",
    "rmse",
    "mae",
    "delta1",
    "delta2",
    "delta3",
    "scale_aligned_rmse",
    "scale_aligned_mae",
    "median_scale_ratio",
    "least_squares_scale_ratio",
)


def stable_hash01(value):
    digest = hashlib.sha1(str(value).encode("utf-8")).hexdigest()
    return int(digest[:12], 16) / float(16**12)


def assign_scene_splits(scenes, train_fraction=0.80, val_fraction=0.10):
    scenes = sorted(set(map(str, scenes)))
    if not scenes:
        return {}

    ranked = sorted(scenes, key=lambda scene: (stable_hash01(scene), scene))
    if len(ranked) < 3:
        return {scene: "train" for scene in ranked}

    train_count = max(1, int(round(len(ranked) * float(train_fraction))))
    val_count = max(1, int(round(len(ranked) * float(val_fraction))))
    if train_count + val_count >= len(ranked):
        train_count = max(1, len(ranked) - 2)
        val_count = 1

    splits = {}
    for index, scene in enumerate(ranked):
        if index < train_count:
            split = "train"
        elif index < train_count + val_count:
            split = "val"
        else:
            split = "test"
        splits[scene] = split
    return splits


def read_metric_depth(path):
    depth = np.asarray(Image.open(path))
    if depth.ndim == 3:
        depth = depth[..., 0]
    depth = depth.astype(np.float32)
    if depth.size and float(np.nanmax(depth)) > 100.0:
        depth /= 1000.0
    depth[~np.isfinite(depth)] = 0.0
    return depth


def resize_depth(depth, width, height):
    depth = np.asarray(depth, dtype=np.float32).squeeze()
    if depth.ndim != 2:
        raise ValueError(f"Expected 2D depth, got shape={depth.shape}")
    image = Image.fromarray(depth, mode="F")
    return np.asarray(
        image.resize((int(width), int(height)), resample=Image.BICUBIC),
        dtype=np.float32,
    )


def valid_depth_mask(source_depth, predicted_depth, min_depth_m=MIN_DEPTH_M):
    source = np.asarray(source_depth, dtype=np.float32)
    predicted = np.asarray(predicted_depth, dtype=np.float32)
    if source.shape != predicted.shape:
        raise ValueError(
            f"Depth shape mismatch: source={source.shape}, predicted={predicted.shape}"
        )
    return (
        np.isfinite(source)
        & np.isfinite(predicted)
        & (source > float(min_depth_m))
        & (predicted > 1e-6)
    )


def depth_scale_factors(predicted, source):
    predicted = np.asarray(predicted, dtype=np.float64)
    source = np.asarray(source, dtype=np.float64)
    scales = source / np.maximum(predicted, 1e-8)
    median_scale = float(np.median(scales))
    least_squares_scale = float(
        np.sum(predicted * source) / max(np.sum(predicted * predicted), 1e-12)
    )
    return median_scale, least_squares_scale


def depth_error_metrics(
    predicted_depth,
    source_depth,
    min_depth_m=MIN_DEPTH_M,
):
    predicted = np.asarray(predicted_depth, dtype=np.float32)
    source = np.asarray(source_depth, dtype=np.float32)
    valid = valid_depth_mask(source, predicted, min_depth_m=min_depth_m)
    if int(valid.sum()) < 16:
        raise ValueError(f"Insufficient valid depth pixels: {int(valid.sum())}")

    pred = predicted[valid].astype(np.float64)
    src = source[valid].astype(np.float64)
    error = pred - src
    abs_error = np.abs(error)
    ratio = np.maximum(
        pred / np.maximum(src, 1e-8),
        src / np.maximum(pred, 1e-8),
    )
    median_scale, least_squares_scale = depth_scale_factors(pred, src)
    aligned_error = pred * median_scale - src

    return {
        "num_pixels": int(source.size),
        "num_valid_pixels": int(valid.sum()),
        "valid_pixel_ratio": float(valid.mean()),
        "abs_rel": float(np.mean(abs_error / np.maximum(src, 1e-8))),
        "rmse": float(np.sqrt(np.mean(error * error))),
        "mae": float(np.mean(abs_error)),
        "delta1": float(np.mean(ratio < 1.25)),
        "delta2": float(np.mean(ratio < 1.25**2)),
        "delta3": float(np.mean(ratio < 1.25**3)),
        "scale_aligned_rmse": float(
            np.sqrt(np.mean(aligned_error * aligned_error))
        ),
        "scale_aligned_mae": float(np.mean(np.abs(aligned_error))),
        "median_scale_ratio": median_scale,
        "least_squares_scale_ratio": least_squares_scale,
        "source_depth_mean_m": float(np.mean(src)),
        "predicted_depth_mean_m": float(np.mean(pred)),
    }


def finite_metric_values(rows, name):
    return [
        float(row[name])
        for row in rows
        if math.isfinite(float(row[name]))
    ]


def summarize_metric_columns(rows, metric_names=DEPTH_SUMMARY_METRICS):
    summary = {}
    for name in metric_names:
        values = finite_metric_values(rows, name)
        summary[f"mean_{name}"] = float(np.mean(values))
        summary[f"median_{name}"] = float(np.median(values))
    return summary


def summarize_depth_rows(rows, split):
    selected = rows if split == "all" else [row for row in rows if row["split"] == split]
    if not selected:
        return None
    summary = {
        "split": split,
        "num_frames": len(selected),
        "num_scenes": len({row["scene"] for row in selected}),
    }
    summary.update(summarize_metric_columns(selected))
    return summary


def discover_complete_rgbd_pose_frames(posed_images_root, scene_splits=None):
    root = Path(posed_images_root)
    if not root.exists():
        raise FileNotFoundError(f"Missing posed_images root: {root}")
    scene_dirs = sorted(path for path in root.iterdir() if path.is_dir())
    if scene_splits is None:
        scene_splits = assign_scene_splits(path.name for path in scene_dirs)

    rows = []
    for scene_dir in scene_dirs:
        jpg_by_stem = {path.stem: path for path in scene_dir.glob("*.jpg")}
        png_by_stem = {path.stem: path for path in scene_dir.glob("*.png")}
        txt_by_stem = {path.stem: path for path in scene_dir.glob("*.txt")}
        complete_stems = sorted(
            set(jpg_by_stem) & set(png_by_stem) & set(txt_by_stem)
        )
        for stem in complete_stems:
            rows.append(
                {
                    "scene": scene_dir.name,
                    "split": scene_splits.get(scene_dir.name, "train"),
                    "frame": stem,
                    "rgb_path": str(jpg_by_stem[stem]),
                    "depth_path": str(png_by_stem[stem]),
                    "pose_path": str(txt_by_stem[stem]),
                }
            )
    return rows


def split_summary(frame_rows):
    summary = []
    for split in ["train", "val", "test", "all"]:
        rows = frame_rows if split == "all" else [
            row for row in frame_rows if row["split"] == split
        ]
        if not rows:
            continue
        summary.append(
            {
                "split": split,
                "num_scenes": len({row["scene"] for row in rows}),
                "num_frames": len(rows),
                "first_scene": sorted({row["scene"] for row in rows})[0],
                "last_scene": sorted({row["scene"] for row in rows})[-1],
            }
        )
    return summary


def write_csv(path, rows):
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    if not rows:
        path.write_text("", encoding="utf-8")
        return
    fields = []
    for row in rows:
        for key in row:
            if key not in fields:
                fields.append(key)
    with path.open("w", newline="", encoding="utf-8") as handle:
        writer = csv.DictWriter(handle, fieldnames=fields)
        writer.writeheader()
        writer.writerows(rows)
