import csv
import json
import math
import os
import random
import sys
import time
import urllib.request
from pathlib import Path

import numpy as np
import torch
import torch.nn as nn
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


RUN_NAME = "run_22_oarh_v2_reconstruction_integration"
RUN19_SEED = 1919
SEED = 2222
MAX_GROUPS = int(os.environ.get("RUN22_MAX_GROUPS", "32"))
MAX_CANDIDATES_PER_GROUP = int(os.environ.get("RUN22_MAX_CANDIDATES_PER_GROUP", "90000"))
MIN_SELECTED_POINTS = int(os.environ.get("RUN22_MIN_SELECTED_POINTS", "100"))
GATE_MARGIN_F1 = float(os.environ.get("RUN22_GATE_MARGIN_F1", "0.005"))
BATCH_SIZE = 65536
OARH_THRESHOLDS = [0.50, 0.70, 0.90]


class OARHv2(nn.Module):
    def __init__(self, in_dim):
        super().__init__()
        self.shared = nn.Sequential(
            nn.Linear(in_dim, 160),
            nn.ReLU(inplace=True),
            nn.Dropout(0.10),
            nn.Linear(160, 96),
            nn.ReLU(inplace=True),
            nn.Dropout(0.05),
            nn.Linear(96, 48),
            nn.ReLU(inplace=True),
        )
        self.keep = nn.Linear(48, 1)
        self.visibility = nn.Linear(48, 3)
        self.depth = nn.Linear(48, 1)

    def forward(self, x):
        h = self.shared(x)
        return self.keep(h).squeeze(-1), self.visibility(h), self.depth(h).squeeze(-1)


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


def read_csv(path):
    with path.open(newline="") as f:
        return list(csv.DictReader(f))


def as_float(row, key, default=0.0):
    value = row.get(key, "")
    if value in ("", "nan", "NaN", None):
        return default
    try:
        x = float(value)
    except ValueError:
        return default
    return x if math.isfinite(x) else default


def as_int(row, key, default=0):
    try:
        return int(float(row.get(key, default)))
    except (TypeError, ValueError):
        return default


def find_file_from_kernel_source(filename, preferred_tokens):
    root = Path("/kaggle/input")
    candidates = sorted(root.rglob(filename))
    for path in candidates:
        text = str(path).lower()
        if any(token in text for token in preferred_tokens):
            print(f"Using {filename}:", path)
            return path
    if candidates:
        print(f"Using first {filename}:", candidates[0])
        return candidates[0]
    raise FileNotFoundError(f"Cannot find {filename}. Add the required Kaggle kernel source.")


def discover_scene_dirs(posed_root):
    scenes = sorted([p for p in posed_root.glob("scene*") if p.is_dir() and list(p.glob("*.jpg"))])
    if not scenes:
        raise FileNotFoundError(f"No scene directories with JPG frames under {posed_root}")
    return scenes


def read_depth(path):
    depth = np.array(Image.open(path))
    if depth.ndim == 3:
        depth = depth[..., 0]
    depth = depth.astype(np.float32)
    if depth.size and float(depth.max()) > 100:
        depth = depth / 1000.0
    depth[~np.isfinite(depth)] = 0.0
    return depth


def intrinsics(depth_shape):
    h, w = depth_shape
    fx = 577.870605 * (w / 640.0)
    fy = 577.870605 * (h / 480.0)
    cx = 319.5 * (w / 640.0)
    cy = 239.5 * (h / 480.0)
    return fx, fy, cx, cy


def project_cam_point(pt_cam, depth_shape):
    z = float(pt_cam[2])
    if z <= 0.10 or not np.isfinite(z):
        return None
    fx, fy, cx, cy = intrinsics(depth_shape)
    x = float(pt_cam[0] / z * fx + cx)
    y = float(pt_cam[1] / z * fy + cy)
    return x, y, z


def output_to_candidates(output):
    pts = [output["pred1"]["pts3d"][0].detach().cpu()]
    pts += [x["pts3d_in_other_view"][0].detach().cpu() for x in output["pred2s"]]
    conf = [output["pred1"]["conf"][0].detach().cpu()]
    conf += [x["conf"][0].detach().cpu() for x in output["pred2s"]]

    pts = torch.stack(pts, dim=0).numpy().astype(np.float32)
    conf = torch.stack(conf, dim=0).numpy().astype(np.float32)
    n_views, h, w, _ = pts.shape
    yy, xx = np.meshgrid(np.arange(h, dtype=np.float32), np.arange(w, dtype=np.float32), indexing="ij")
    xx = np.broadcast_to(xx[None, :, :], (n_views, h, w))
    yy = np.broadcast_to(yy[None, :, :], (n_views, h, w))
    view_idx = np.broadcast_to(np.arange(n_views, dtype=np.int32)[:, None, None], (n_views, h, w))

    flat_pts = pts.reshape(-1, 3)
    flat_conf = conf.reshape(-1)
    flat_x = xx.reshape(-1)
    flat_y = yy.reshape(-1)
    flat_view = view_idx.reshape(-1)
    finite = np.isfinite(flat_pts).all(axis=1) & np.isfinite(flat_conf)
    return (
        flat_pts[finite],
        flat_conf[finite],
        flat_x[finite],
        flat_y[finite],
        flat_view[finite],
        h,
        w,
    )


def load_oarh_model(checkpoint_path, device):
    ckpt = torch.load(checkpoint_path, map_location=device)
    model = OARHv2(int(ckpt["feature_dim"])).to(device)
    model.load_state_dict(ckpt["state_dict"], strict=True)
    model.eval()
    threshold = float(ckpt.get("keep_threshold", 0.70))
    print("Loaded Run 21 OARH checkpoint:", checkpoint_path)
    print("Run 21 keep threshold:", threshold)
    return model, threshold


def build_feature_matrix(points, xs, ys, view_ids, view_files, group_row, image_h, image_w):
    first_pose = base.parse_pose(str(view_files[0]).replace(".jpg", ".txt"))
    world_from_first = first_pose
    poses = [base.parse_pose(str(p).replace(".jpg", ".txt")) for p in view_files]
    inv_poses = [np.linalg.inv(p) for p in poses]
    depths = [read_depth(str(p).replace(".jpg", ".png")) for p in view_files]
    classes = row_classes(group_row)
    policy = group_row.get("view_policy", "")
    num_views = as_float(group_row, "num_views")
    priority = as_float(group_row, "priority_score")
    features = np.zeros((len(points), 21), dtype=np.float32)

    for i, pt in enumerate(points):
        src_idx = int(np.clip(view_ids[i], 0, len(view_files) - 1))
        src_depth = depths[src_idx]
        sx = int(np.clip(round(xs[i] / max(image_w - 1, 1) * (src_depth.shape[1] - 1)), 0, src_depth.shape[1] - 1))
        sy = int(np.clip(round(ys[i] / max(image_h - 1, 1) * (src_depth.shape[0] - 1)), 0, src_depth.shape[0] - 1))
        src_observed = float(src_depth[sy, sx])

        pt_first = np.array([pt[0], pt[1], pt[2], 1.0], dtype=np.float32)
        pt_world = world_from_first @ pt_first
        pt_src = inv_poses[src_idx] @ pt_world
        candidate_depth = float(pt_src[2]) if np.isfinite(pt_src[2]) else 0.0
        candidate_delta = abs(candidate_depth - src_observed) if src_observed > 0 else 0.0

        best = None
        for tgt_idx, tgt_depth in enumerate(depths):
            if tgt_idx == src_idx:
                continue
            pt_tgt = inv_poses[tgt_idx] @ pt_world
            projected = project_cam_point(pt_tgt[:3], tgt_depth.shape)
            if projected is None:
                continue
            x_t, y_t, z_t = projected
            in_bounds = 0 <= x_t < tgt_depth.shape[1] and 0 <= y_t < tgt_depth.shape[0]
            if not in_bounds:
                continue
            xi = int(np.clip(round(x_t), 0, tgt_depth.shape[1] - 1))
            yi = int(np.clip(round(y_t), 0, tgt_depth.shape[0] - 1))
            observed = float(tgt_depth[yi, xi])
            if observed <= 0 or not np.isfinite(observed):
                continue
            residual = float(z_t - observed)
            baseline = float(np.linalg.norm(poses[src_idx][:3, 3] - poses[tgt_idx][:3, 3]))
            score = abs(residual)
            if best is None or score > best["score"]:
                best = {
                    "score": score,
                    "target_x_norm": x_t / max(tgt_depth.shape[1] - 1, 1),
                    "target_y_norm": y_t / max(tgt_depth.shape[0] - 1, 1),
                    "projected": z_t,
                    "observed": observed,
                    "residual": residual,
                    "in_bounds": 1.0,
                    "baseline": baseline,
                }

        if best is None:
            best = {
                "target_x_norm": 0.0,
                "target_y_norm": 0.0,
                "projected": 0.0,
                "observed": 0.0,
                "residual": 0.0,
                "in_bounds": 0.0,
                "baseline": 0.0,
            }

        residual = best["residual"]
        features[i] = np.array(
            [
                float(xs[i] / max(image_w - 1, 1)),
                float(ys[i] / max(image_h - 1, 1)),
                best["target_x_norm"],
                best["target_y_norm"],
                math.log1p(max(src_observed, 0.0)),
                math.log1p(max(candidate_depth, 0.0)),
                candidate_delta,
                math.log1p(max(best["projected"], 0.0)),
                math.log1p(max(best["observed"], 0.0)),
                residual,
                abs(residual),
                best["in_bounds"],
                min(best["baseline"], 5.0) / 5.0,
                num_views / 5.0,
                priority,
                1.0 if policy == "hybrid" else 0.0,
                1.0 if policy == "diversity_aware" else 0.0,
                1.0 if "occlusion_core" in classes else 0.0,
                1.0 if "occlusion_borderline" in classes else 0.0,
                1.0 if "low_overlap_far" in classes else 0.0,
                1.0 if "wrong_depth_hard_negative" in classes else 0.0,
            ],
            dtype=np.float32,
        )
    return features


def row_classes(row):
    return str(row.get("group_classes", "")).split("|")


def predict_keep_prob(model, features):
    device = next(model.parameters()).device
    probs = []
    with torch.no_grad():
        for start in range(0, len(features), BATCH_SIZE):
            xb = torch.from_numpy(features[start : start + BATCH_SIZE]).to(device)
            logits, _vis, _res = model(xb)
            probs.append(torch.sigmoid(logits).detach().cpu().numpy())
    return np.concatenate(probs)


def confidence_mask(conf, conf_percent):
    threshold = float(np.quantile(conf, conf_percent / 100.0))
    return conf >= threshold, threshold


def subsample_candidates(points, conf, xs, ys, view_ids, max_candidates, seed):
    if len(points) <= max_candidates or max_candidates <= 0:
        return points, conf, xs, ys, view_ids
    rng = np.random.default_rng(seed)
    idx = rng.choice(len(points), max_candidates, replace=False)
    return points[idx], conf[idx], xs[idx], ys[idx], view_ids[idx]


def score_selection(points, mask, gt, method, group_row, extra):
    if int(mask.sum()) < MIN_SELECTED_POINTS:
        mask = np.ones(len(points), dtype=bool)
        fallback = 1
    else:
        fallback = 0
    selected = base.downsample(points[mask].astype(np.float32), base.MAX_POINTS)
    metrics = base.compute_metrics(selected, gt)
    return {
        "run": RUN_NAME,
        "method": method,
        "split": group_row.get("split"),
        "scene": group_row.get("scene"),
        "num_views": as_int(group_row, "num_views"),
        "view_policy": group_row.get("view_policy"),
        "group_key": group_row.get("group_key"),
        "group_classes": group_row.get("group_classes"),
        "priority_score": as_float(group_row, "priority_score"),
        "selected_ratio": float(mask.mean()),
        "fallback_all_points": fallback,
        **extra,
        **metrics,
    }


def evaluate_group(model, backbone, root, scene_lookup, group_row, out_dir, run21_threshold):
    scene_name = group_row["scene"]
    scene_dir = scene_lookup.get(scene_name)
    if scene_dir is None:
        raise FileNotFoundError(f"Scene {scene_name} not found in posed_images")
    num_views = as_int(group_row, "num_views")
    policy = group_row.get("view_policy")
    view_files = base.choose_views(scene_dir, num_views, policy, seed=RUN19_SEED + num_views)
    print("Run 22 group views:", {"group": group_row.get("group_key"), "views": [p.name for p in view_files]})

    output, _glb, runtime = base.run_inference(backbone, root, view_files, out_dir)
    gt, _stats = base.build_gt_cloud(view_files)
    points, conf, xs, ys, view_ids, image_h, image_w = output_to_candidates(output)
    points, conf, xs, ys, view_ids = subsample_candidates(
        points,
        conf,
        xs,
        ys,
        view_ids,
        MAX_CANDIDATES_PER_GROUP,
        SEED + as_int(group_row, "num_views") + abs(hash(group_row.get("group_key", ""))) % 100000,
    )
    features = build_feature_matrix(points, xs, ys, view_ids, view_files, group_row, image_h, image_w)
    keep_prob = predict_keep_prob(model, features)

    final_percent = base.FIXED_FINAL_CONF_BY_VIEW_COUNT.get(num_views, base.CONF_PERCENT)
    conf_final, conf_threshold = confidence_mask(conf, final_percent)
    rows = [
        score_selection(
            points,
            conf_final,
            gt,
            "confidence_fixed_final",
            group_row,
            {
                "runtime_seconds": runtime,
                "conf_percent": final_percent,
                "conf_threshold": conf_threshold,
                "oarh_threshold": "",
            },
        )
    ]
    for threshold in OARH_THRESHOLDS:
        rows.append(
            score_selection(
                points,
                keep_prob >= threshold,
                gt,
                f"oarh_v2_threshold_{threshold:.2f}",
                group_row,
                {
                    "runtime_seconds": runtime,
                    "conf_percent": "",
                    "conf_threshold": "",
                    "oarh_threshold": threshold,
                    "mean_keep_prob": float(keep_prob.mean()),
                },
            )
        )
    rows.append(
        score_selection(
            points,
            (keep_prob >= run21_threshold) & conf_final,
            gt,
            "oarh_v2_and_confidence_guard",
            group_row,
            {
                "runtime_seconds": runtime,
                "conf_percent": final_percent,
                "conf_threshold": conf_threshold,
                "oarh_threshold": run21_threshold,
                "mean_keep_prob": float(keep_prob.mean()),
            },
        )
    )
    del output
    torch.cuda.empty_cache()
    for row in rows:
        print("Run 22 metric row:", row)
    return rows


def summarize(rows):
    out = []
    keys = sorted({(r["split"], r["method"]) for r in rows})
    final_lookup = {}
    for split in sorted({r["split"] for r in rows}):
        vals = [r["fscore"] for r in rows if r["split"] == split and r["method"] == "confidence_fixed_final"]
        final_lookup[split] = float(np.mean(vals)) if vals else 0.0
    for split, method in keys:
        items = [r for r in rows if r["split"] == split and r["method"] == method]
        f = np.array([r["fscore"] for r in items], dtype=np.float32)
        out.append(
            {
                "run": RUN_NAME,
                "split": split,
                "method": method,
                "num_groups": len(items),
                "mean_fscore": float(f.mean()) if len(f) else 0.0,
                "std_fscore": float(f.std()) if len(f) else 0.0,
                "delta_vs_confidence_fixed_final": float(f.mean() - final_lookup.get(split, 0.0)) if len(f) else 0.0,
                "mean_precision": float(np.mean([r["precision"] for r in items])) if items else 0.0,
                "mean_recall": float(np.mean([r["recall"] for r in items])) if items else 0.0,
                "mean_selected_ratio": float(np.mean([r["selected_ratio"] for r in items])) if items else 0.0,
            }
        )
    return out


def make_gate_decision(summary_rows):
    val_rows = [r for r in summary_rows if r["split"] == "val"]
    final = next((r for r in val_rows if r["method"] == "confidence_fixed_final"), None)
    learned = [r for r in val_rows if r["method"] != "confidence_fixed_final"]
    if not final or not learned:
        return [
            {
                "run": RUN_NAME,
                "selected_method": "confidence_fixed_final",
                "reason": "No validation rows available for learned gate.",
            }
        ]
    best = max(learned, key=lambda r: r["mean_fscore"])
    delta = best["mean_fscore"] - final["mean_fscore"]
    selected = best["method"] if delta >= GATE_MARGIN_F1 else "confidence_fixed_final"
    return [
        {
            "run": RUN_NAME,
            "selected_method": selected,
            "best_learned_method": best["method"],
            "validation_confidence_fscore": final["mean_fscore"],
            "validation_best_learned_fscore": best["mean_fscore"],
            "delta_fscore": float(delta),
            "gate_margin_f1": GATE_MARGIN_F1,
            "recommendation": (
                "Use learned OARH v2 integration for the next full test run."
                if selected != "confidence_fixed_final"
                else "Keep fixed confidence as reconstruction policy until learned OARH wins validation."
            ),
        }
    ]


def select_groups(manifest_rows):
    rows = [r for r in manifest_rows if str(r.get("is_final_eval_candidate", "0")) in {"1", "True", "true"}]
    rows = sorted(rows, key=lambda r: (r.get("split") != "val", -as_float(r, "priority_score"), r.get("scene", ""), as_int(r, "num_views")))
    if MAX_GROUPS > 0:
        rows = rows[:MAX_GROUPS]
    return rows


def main():
    random.seed(SEED)
    np.random.seed(SEED)
    torch.manual_seed(SEED)
    started = time.time()
    out_dir = Path("/kaggle/working/outputs") / RUN_NAME
    out_dir.mkdir(parents=True, exist_ok=True)

    base.require_t4x2()
    run20_manifest = find_file_from_kernel_source(
        "final_eval_group_manifest.csv",
        ["run-20", "run_20", "subset-mining"],
    )
    run21_ckpt = find_file_from_kernel_source(
        "oarh_v2_multitask_head.pt",
        ["run-21", "run_21", "oarh-v2-multitask"],
    )
    manifest_rows = read_csv(run20_manifest)
    eval_groups = select_groups(manifest_rows)
    print("Run 22 eval groups:", len(eval_groups))
    for row in eval_groups:
        print("Run 22 selected group:", row)

    root = base.clone_repo()
    base.install_deps(root)
    posed_root = base.find_posed_images_root()
    scene_dirs = discover_scene_dirs(posed_root)
    scene_lookup = {p.name: p for p in scene_dirs}
    print("POSED_IMAGES:", posed_root)
    print("SCENE_DIRS:", [str(p) for p in scene_dirs])

    ckpt_path = base.download_checkpoint(root)
    print("Checkpoint:", ckpt_path)
    backbone = base.load_model(root, ckpt_path)
    device = "cuda" if torch.cuda.is_available() else "cpu"
    oarh, run21_threshold = load_oarh_model(run21_ckpt, device)

    metric_rows = []
    for group in eval_groups:
        group_out = out_dir / "groups" / group["group_key"]
        metric_rows.extend(evaluate_group(oarh, backbone, root, scene_lookup, group, group_out, run21_threshold))

    summary_rows = summarize(metric_rows)
    gate_rows = make_gate_decision(summary_rows)
    write_csv_union(out_dir / "metrics.csv", metric_rows)
    write_csv_union(out_dir / "summary.csv", summary_rows)
    write_csv_union(out_dir / "gate_decision.csv", gate_rows)
    (out_dir / "run_config.json").write_text(
        json.dumps(
            {
                "run": RUN_NAME,
                "source_run20_manifest": str(run20_manifest),
                "source_run21_checkpoint": str(run21_ckpt),
                "num_eval_groups": len(eval_groups),
                "max_groups": MAX_GROUPS,
                "max_candidates_per_group": MAX_CANDIDATES_PER_GROUP,
                "oarh_thresholds": OARH_THRESHOLDS,
                "run21_checkpoint_threshold": run21_threshold,
                "gate_margin_f1": GATE_MARGIN_F1,
                "runtime_seconds": time.time() - started,
                "note": (
                    "Reconstruction-level OARH v2 integration test. It compares fixed-confidence "
                    "selection with Run 21 OARH keep probabilities on Run 20 final-eval groups. "
                    "Feature construction still uses posed RGB-D depth as a proxy for cross-view "
                    "visibility, so the result is a reconstruction integration benchmark, not a "
                    "deployment-ready image-only occlusion module."
                ),
            },
            indent=2,
        )
    )

    print("Run 22 summary:")
    for row in summary_rows:
        print(row)
    print("Run 22 gate decision:")
    for row in gate_rows:
        print(row)


if __name__ == "__main__":
    main()
