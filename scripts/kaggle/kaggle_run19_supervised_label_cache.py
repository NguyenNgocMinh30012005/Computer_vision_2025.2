import csv
import json
import math
import os
import random
import sys
import time
import urllib.request
from collections import defaultdict
from pathlib import Path

import numpy as np
from PIL import Image


RAW_BASE = "https://raw.githubusercontent.com/NguyenNgocMinh30012005/Computer_vision_2025.2/main/scripts/kaggle/kaggle_run11_final_validation_3seeds.py"


def ensure_helper_module(module_name, raw_url):
    try:
        return __import__(module_name)
    except ModuleNotFoundError:
        helper_path = Path(f"{module_name}.py")
        print(f"Downloading helper module {module_name} from GitHub raw")
        urllib.request.urlretrieve(raw_url, helper_path)
        sys.path.insert(0, str(Path.cwd()))
        return __import__(module_name)


base = ensure_helper_module("kaggle_run11_final_validation_3seeds", RAW_BASE)


RUN_NAME = "run_19_supervised_label_cache"
SEED = 1919
VIEW_COUNTS = [3, 4, 5]
POLICIES = ["hybrid", "diversity_aware"]
MAX_SCENES = int(os.environ.get("RUN19_MAX_SCENES", "30"))
MAX_PAIRS_PER_GROUP = int(os.environ.get("RUN19_MAX_PAIRS_PER_GROUP", "6"))
SAMPLES_PER_PAIR = int(os.environ.get("RUN19_SAMPLES_PER_PAIR", "2048"))
DEPTH_TOL_M = float(os.environ.get("RUN19_DEPTH_TOL_M", "0.05"))
MIN_DEPTH_M = 0.10


def write_csv_union(path, rows):
    path.parent.mkdir(parents=True, exist_ok=True)
    if not rows:
        path.write_text("")
        return
    fieldnames = []
    for row in rows:
        for key in row:
            if key not in fieldnames:
                fieldnames.append(key)
    with path.open("w", newline="") as f:
        writer = csv.DictWriter(f, fieldnames=fieldnames)
        writer.writeheader()
        writer.writerows(rows)


def read_depth(path):
    depth = np.array(Image.open(path))
    if depth.ndim == 3:
        depth = depth[..., 0]
    depth = depth.astype(np.float32)
    depth_max = float(depth.max()) if depth.size else 0.0
    if depth_max > 100:
        depth = depth / 1000.0
    depth[~np.isfinite(depth)] = 0.0
    return depth


def intrinsics_for_depth(depth):
    h, w = depth.shape
    fx = 577.870605 * (w / 640.0)
    fy = 577.870605 * (h / 480.0)
    cx = 319.5 * (w / 640.0)
    cy = 239.5 * (h / 480.0)
    return fx, fy, cx, cy


def depth_to_cam(x, y, z, depth_shape):
    h, w = depth_shape
    fx, fy, cx, cy = intrinsics_for_depth(np.zeros((h, w), dtype=np.float32))
    return np.array([(x - cx) / fx * z, (y - cy) / fy * z, z, 1.0], dtype=np.float32)


def cam_to_pixel(pt_cam, depth_shape):
    h, w = depth_shape
    z = float(pt_cam[2])
    if z <= MIN_DEPTH_M or not np.isfinite(z):
        return None
    fx, fy, cx, cy = intrinsics_for_depth(np.zeros((h, w), dtype=np.float32))
    x = float(pt_cam[0] / z * fx + cx)
    y = float(pt_cam[1] / z * fy + cy)
    return x, y, z


def sample_valid_pixels(depth, max_samples, rng):
    valid = np.argwhere(np.isfinite(depth) & (depth > MIN_DEPTH_M))
    if len(valid) == 0:
        return np.zeros((0, 2), dtype=np.int32)
    if len(valid) > max_samples:
        idx = rng.choice(len(valid), max_samples, replace=False)
        valid = valid[idx]
    # Return x, y.
    return valid[:, ::-1].astype(np.int32)


def candidate_depths(src_depth, rng):
    if rng.random() < 0.5:
        wrong_depth = max(MIN_DEPTH_M, float(src_depth) * rng.uniform(0.55, 0.85))
        wrong_type = "synthetic_floating_front"
    else:
        wrong_depth = float(src_depth) + rng.uniform(0.25, 0.85)
        wrong_type = "synthetic_behind_surface"
    return [("gt_surface", float(src_depth), 1.0), (wrong_type, wrong_depth, 0.0)]


def visibility_label(projected_z, target_depth):
    if target_depth <= MIN_DEPTH_M or not np.isfinite(target_depth):
        return "unknown_depth", math.nan
    residual = float(projected_z - target_depth)
    if abs(residual) <= DEPTH_TOL_M:
        return "visible_consistent", residual
    if residual > DEPTH_TOL_M:
        return "occluded_behind_observed_surface", residual
    return "floating_in_front_of_observed_surface", residual


def scene_splits(scene_dirs):
    n = len(scene_dirs)
    if n <= 1:
        return {scene_dirs[0].name: "train"} if scene_dirs else {}
    if n == 2:
        return {scene_dirs[0].name: "train", scene_dirs[1].name: "test"}
    if n < 5:
        return {scene.name: ("train" if i == 0 else "val" if i == 1 else "test") for i, scene in enumerate(scene_dirs)}
    train_cut = max(1, int(round(0.60 * n)))
    val_cut = max(train_cut + 1, int(round(0.80 * n)))
    val_cut = min(val_cut, n - 1)
    splits = {}
    for i, scene in enumerate(scene_dirs):
        if i < train_cut:
            splits[scene.name] = "train"
        elif i < val_cut:
            splits[scene.name] = "val"
        else:
            splits[scene.name] = "test"
    return splits


def discover_scene_dirs(posed_root):
    scenes = sorted([p for p in posed_root.glob("scene*") if p.is_dir() and list(p.glob("*.jpg"))])
    if not scenes:
        raise FileNotFoundError(f"No scene directories with JPG frames under {posed_root}")
    return scenes[:MAX_SCENES]


def pair_baseline(src_pose, tgt_pose):
    return float(np.linalg.norm(src_pose[:3, 3] - tgt_pose[:3, 3]))


def pair_rows(scene_name, split, num_views, policy, group_key, src_jpg, tgt_jpg, rng):
    src_depth = read_depth(str(src_jpg).replace(".jpg", ".png"))
    tgt_depth = read_depth(str(tgt_jpg).replace(".jpg", ".png"))
    src_pose = base.parse_pose(str(src_jpg).replace(".jpg", ".txt"))
    tgt_pose = base.parse_pose(str(tgt_jpg).replace(".jpg", ".txt"))
    tgt_pose_inv = np.linalg.inv(tgt_pose)
    baseline_m = pair_baseline(src_pose, tgt_pose)
    pixels = sample_valid_pixels(src_depth, SAMPLES_PER_PAIR, rng)
    rows = []

    for x, y in pixels:
        z_src = float(src_depth[y, x])
        for candidate_type, z_candidate, keep_label in candidate_depths(z_src, rng):
            pt_src = depth_to_cam(x, y, z_candidate, src_depth.shape)
            pt_world = src_pose @ pt_src
            pt_tgt = tgt_pose_inv @ pt_world
            projected = cam_to_pixel(pt_tgt[:3], tgt_depth.shape)
            if projected is None:
                label = "behind_target_camera"
                x_tgt = y_tgt = projected_z = target_z = depth_residual = math.nan
                in_bounds = 0.0
            else:
                x_tgt, y_tgt, projected_z = projected
                h_t, w_t = tgt_depth.shape
                in_bounds = float(0 <= x_tgt < w_t and 0 <= y_tgt < h_t)
                if not in_bounds:
                    label = "out_of_view"
                    target_z = depth_residual = math.nan
                else:
                    xi = int(np.clip(round(x_tgt), 0, w_t - 1))
                    yi = int(np.clip(round(y_tgt), 0, h_t - 1))
                    target_z = float(tgt_depth[yi, xi])
                    label, depth_residual = visibility_label(projected_z, target_z)

            rows.append(
                {
                    "run": RUN_NAME,
                    "split": split,
                    "scene": scene_name,
                    "num_views": num_views,
                    "view_policy": policy,
                    "group_key": group_key,
                    "source_image": src_jpg.name,
                    "target_image": tgt_jpg.name,
                    "candidate_type": candidate_type,
                    "keep_label": keep_label,
                    "visibility_label": label,
                    "support_label": 1.0 if label == "visible_consistent" and keep_label > 0.5 else 0.0,
                    "occlusion_label": 1.0 if label == "occluded_behind_observed_surface" else 0.0,
                    "floating_label": 1.0 if label == "floating_in_front_of_observed_surface" or keep_label < 0.5 else 0.0,
                    "match_label": 1.0 if label == "visible_consistent" and keep_label > 0.5 else 0.0,
                    "src_x_norm": float(x / max(src_depth.shape[1] - 1, 1)),
                    "src_y_norm": float(y / max(src_depth.shape[0] - 1, 1)),
                    "src_depth_m": z_src,
                    "candidate_depth_m": float(z_candidate),
                    "candidate_depth_delta_m": float(abs(z_candidate - z_src)),
                    "target_x_norm": float(x_tgt / max(tgt_depth.shape[1] - 1, 1)) if np.isfinite(x_tgt) else math.nan,
                    "target_y_norm": float(y_tgt / max(tgt_depth.shape[0] - 1, 1)) if np.isfinite(y_tgt) else math.nan,
                    "projected_target_depth_m": float(projected_z) if np.isfinite(projected_z) else math.nan,
                    "observed_target_depth_m": float(target_z) if np.isfinite(target_z) else math.nan,
                    "depth_residual_m": float(depth_residual) if np.isfinite(depth_residual) else math.nan,
                    "target_in_bounds": in_bounds,
                    "baseline_m": baseline_m,
                }
            )
    return rows


def summarize(rows):
    grouped = defaultdict(list)
    for row in rows:
        key = (row["split"], row["scene"], row["num_views"], row["view_policy"], row["group_key"])
        grouped[key].append(row)
    summary = []
    for key, items in sorted(grouped.items()):
        split, scene, num_views, policy, group_key = key
        labels = [r["visibility_label"] for r in items]
        n = len(items)
        occlusion_ratio = sum(x == "occluded_behind_observed_surface" for x in labels) / max(n, 1)
        visible_ratio = sum(x == "visible_consistent" for x in labels) / max(n, 1)
        floating_ratio = sum(x == "floating_in_front_of_observed_surface" for x in labels) / max(n, 1)
        wrong_candidate_ratio = sum(float(r["keep_label"]) < 0.5 for r in items) / max(n, 1)
        summary.append(
            {
                "run": RUN_NAME,
                "split": split,
                "scene": scene,
                "num_views": num_views,
                "view_policy": policy,
                "group_key": group_key,
                "num_labels": n,
                "visible_ratio": float(visible_ratio),
                "occlusion_ratio": float(occlusion_ratio),
                "floating_ratio": float(floating_ratio),
                "wrong_candidate_ratio": float(wrong_candidate_ratio),
                "mean_baseline_m": float(np.mean([float(r["baseline_m"]) for r in items])),
                "is_occlusion_heavy": bool(occlusion_ratio >= 0.20),
                "has_strong_wrong_depth_signal": bool(floating_ratio + wrong_candidate_ratio >= 0.35),
            }
        )
    return summary


def main():
    random.seed(SEED)
    np.random.seed(SEED)
    started = time.time()
    out_dir = Path("/kaggle/working/outputs") / RUN_NAME
    out_dir.mkdir(parents=True, exist_ok=True)

    posed_root = base.find_posed_images_root()
    scene_dirs = discover_scene_dirs(posed_root)
    splits = scene_splits(scene_dirs)
    rng = np.random.default_rng(SEED)
    print("POSED_IMAGES:", posed_root)
    print("SCENE_DIRS:", [str(p) for p in scene_dirs])
    print("SCENE_SPLITS:", splits)

    split_rows = [{"scene": scene.name, "split": splits.get(scene.name, "train")} for scene in scene_dirs]
    label_rows = []
    manifest_rows = []
    for scene_dir in scene_dirs:
        split = splits.get(scene_dir.name, "train")
        for num_views in VIEW_COUNTS:
            for policy in POLICIES:
                views = base.choose_views(scene_dir, num_views, policy, seed=SEED + num_views)
                group_key = f"{scene_dir.name}_{num_views}_{policy}"
                manifest_rows.append(
                    {
                        "run": RUN_NAME,
                        "split": split,
                        "scene": scene_dir.name,
                        "num_views": num_views,
                        "view_policy": policy,
                        "group_key": group_key,
                        "selected_images": "|".join(p.name for p in views),
                    }
                )
                pair_count = 0
                for i in range(len(views)):
                    for j in range(len(views)):
                        if i == j:
                            continue
                        if pair_count >= MAX_PAIRS_PER_GROUP:
                            break
                        rows = pair_rows(scene_dir.name, split, num_views, policy, group_key, views[i], views[j], rng)
                        label_rows.extend(rows)
                        pair_count += 1
                    if pair_count >= MAX_PAIRS_PER_GROUP:
                        break
                print(
                    "Run 19 group:",
                    {
                        "scene": scene_dir.name,
                        "split": split,
                        "num_views": num_views,
                        "policy": policy,
                        "selected": [p.name for p in views],
                        "pairs": pair_count,
                        "labels_so_far": len(label_rows),
                    },
                )

    summary_rows = summarize(label_rows)
    occlusion_rows = [r for r in summary_rows if r["is_occlusion_heavy"]]
    config = {
        "run": RUN_NAME,
        "seed": SEED,
        "view_counts": VIEW_COUNTS,
        "policies": POLICIES,
        "max_scenes": MAX_SCENES,
        "samples_per_pair": SAMPLES_PER_PAIR,
        "max_pairs_per_group": MAX_PAIRS_PER_GROUP,
        "depth_tol_m": DEPTH_TOL_M,
        "runtime_seconds": time.time() - started,
        "label_meaning": {
            "keep_label": "1 for GT-surface depth candidates, 0 for synthetic wrong-depth candidates",
            "visibility_label": "target-view relation from projected candidate depth vs observed target depth",
            "support_label": "1 only when a valid GT-surface point is visible/consistent in the target view",
            "occlusion_label": "1 when candidate lies behind observed target depth",
            "floating_label": "1 for front-depth conflict or synthetic wrong-depth candidates",
            "match_label": "1 for geometry-consistent positive correspondences; hard negatives are rows with match_label=0",
        },
        "next_runs": [
            "Run 20 mines occlusion-heavy and repeated-structure subsets from this cache.",
            "Run 21 trains OARH v2 with keep, visibility, and depth-residual multitask losses.",
            "Run 24/25 should use these labels with image-only MASt3R features for RSDH v2.",
        ],
    }
    write_csv_union(out_dir / "scene_split.csv", split_rows)
    write_csv_union(out_dir / "view_group_manifest.csv", manifest_rows)
    write_csv_union(out_dir / "label_cache.csv", label_rows)
    write_csv_union(out_dir / "label_summary.csv", summary_rows)
    write_csv_union(out_dir / "occlusion_heavy_groups.csv", occlusion_rows)
    (out_dir / "run_config.json").write_text(json.dumps(config, indent=2))

    print("Run 19 summary:")
    for row in summary_rows[:24]:
        print(row)
    print("Run 19 occlusion-heavy groups:")
    for row in occlusion_rows[:24]:
        print(row)
    print("Run 19 output dir:", out_dir)


if __name__ == "__main__":
    main()
