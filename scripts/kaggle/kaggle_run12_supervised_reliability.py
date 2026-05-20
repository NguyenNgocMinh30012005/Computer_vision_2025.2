import json
import random
import sys
import time
import urllib.request
from pathlib import Path

import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
from scipy.spatial import cKDTree

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


RUN_NAME = "run_12_supervised_reliability"
VIEW_COUNTS = [2, 3, 4, 5]
TRAIN_SCENE_INDEX = 0
TEST_SCENE_INDEX = 1
MAX_TRAIN_POINTS_PER_CASE = 40000
MAX_EVAL_POINTS_PER_CASE = 80000
GOOD_DISTANCE_M = 0.05
BAD_DISTANCE_M = 0.10
EPOCHS = 18
BATCH_SIZE = 8192
LR = 1e-3
SEED = 1234


class ReliabilityMLP(nn.Module):
    def __init__(self, in_dim):
        super().__init__()
        self.net = nn.Sequential(
            nn.Linear(in_dim, 128),
            nn.ReLU(inplace=True),
            nn.Dropout(0.10),
            nn.Linear(128, 64),
            nn.ReLU(inplace=True),
            nn.Linear(64, 1),
        )

    def forward(self, x):
        return self.net(x).squeeze(-1)


def set_seed(seed=SEED):
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)


def output_to_candidates(output):
    pts = [output["pred1"]["pts3d"][0].detach().cpu()]
    pts += [x["pts3d_in_other_view"][0].detach().cpu() for x in output["pred2s"]]
    conf = [output["pred1"]["conf"][0].detach().cpu()]
    conf += [x["conf"][0].detach().cpu() for x in output["pred2s"]]

    pts = torch.stack(pts, dim=0).numpy().astype(np.float32)
    conf = torch.stack(conf, dim=0).numpy().astype(np.float32)
    n_views, height, width, _ = pts.shape

    yy, xx = np.meshgrid(
        np.linspace(0.0, 1.0, height, dtype=np.float32),
        np.linspace(0.0, 1.0, width, dtype=np.float32),
        indexing="ij",
    )
    xx = np.broadcast_to(xx[None, :, :], (n_views, height, width))
    yy = np.broadcast_to(yy[None, :, :], (n_views, height, width))
    view_id = np.broadcast_to(
        np.linspace(0.0, 1.0, n_views, dtype=np.float32)[:, None, None],
        (n_views, height, width),
    )

    flat_pts = pts.reshape(-1, 3)
    flat_conf = conf.reshape(-1)
    finite = np.isfinite(flat_pts).all(axis=1) & np.isfinite(flat_conf)
    flat_pts = flat_pts[finite]
    flat_conf = flat_conf[finite]
    xx = xx.reshape(-1)[finite]
    yy = yy.reshape(-1)[finite]
    view_id = view_id.reshape(-1)[finite]

    xyz_centered = flat_pts - np.nanmean(flat_pts, axis=0, keepdims=True)
    xyz_scale = np.nanstd(xyz_centered, axis=0, keepdims=True) + 1e-6
    xyz_norm = xyz_centered / xyz_scale
    radius = np.linalg.norm(xyz_norm, axis=1)
    log_conf = np.log1p(np.maximum(flat_conf, 0.0))
    conf_z = (log_conf - log_conf.mean()) / (log_conf.std() + 1e-6)

    feats = np.column_stack(
        [
            conf_z,
            log_conf,
            xyz_norm[:, 0],
            xyz_norm[:, 1],
            xyz_norm[:, 2],
            radius,
            xx,
            yy,
            view_id,
        ]
    ).astype(np.float32)
    return flat_pts.astype(np.float32), flat_conf.astype(np.float32), feats


def make_labels(points, gt):
    aligned = base.center_scale_align(points, gt)
    dists, _ = cKDTree(gt).query(aligned, k=1, workers=-1)
    usable = (dists <= GOOD_DISTANCE_M) | (dists >= BAD_DISTANCE_M)
    labels = (dists <= GOOD_DISTANCE_M).astype(np.float32)
    return labels, usable, dists.astype(np.float32)


def subsample_case(points, conf, feats, labels, usable, dists, max_points, seed):
    idx = np.flatnonzero(usable)
    if len(idx) > max_points:
        rng = np.random.default_rng(seed)
        pos = idx[labels[idx] > 0.5]
        neg = idx[labels[idx] <= 0.5]
        half = max_points // 2
        pos_take = rng.choice(pos, min(len(pos), half), replace=False) if len(pos) else np.array([], dtype=int)
        neg_take = rng.choice(neg, min(len(neg), max_points - len(pos_take)), replace=False)
        idx = np.concatenate([pos_take, neg_take])
        if len(idx) < max_points:
            rest = np.setdiff1d(np.flatnonzero(usable), idx, assume_unique=False)
            add = rng.choice(rest, min(len(rest), max_points - len(idx)), replace=False)
            idx = np.concatenate([idx, add])
    return points[idx], conf[idx], feats[idx], labels[idx], dists[idx]


def collect_case(model, root, scene_dir, view_count, policy, out_dir, max_points, seed):
    view_files = base.choose_views(scene_dir, view_count, policy, seed=seed)
    output, _, runtime = base.run_inference(model, root, view_files, out_dir)
    gt, _ = base.build_gt_cloud(view_files)
    points, conf, feats = output_to_candidates(output)
    labels, usable, dists = make_labels(points, gt)
    points, conf, feats, labels, dists = subsample_case(
        points, conf, feats, labels, usable, dists, max_points, seed + view_count
    )
    del output
    torch.cuda.empty_cache()
    return {
        "scene": scene_dir.name,
        "view_count": view_count,
        "policy": policy,
        "view_files": [str(p) for p in view_files],
        "points": points,
        "conf": conf,
        "features": feats,
        "labels": labels,
        "dists": dists,
        "gt": gt,
        "runtime": runtime,
    }


def confidence_grid_threshold(train_cases):
    candidates = np.concatenate([c["conf"] for c in train_cases])
    labels = np.concatenate([c["labels"] for c in train_cases])
    quantiles = np.linspace(0.01, 0.20, 20)
    best = {"threshold": float(np.quantile(candidates, 0.01)), "f1": -1.0}
    for q in quantiles:
        thr = float(np.quantile(candidates, q))
        pred = candidates >= thr
        tp = float(((pred == 1) & (labels == 1)).sum())
        fp = float(((pred == 1) & (labels == 0)).sum())
        fn = float(((pred == 0) & (labels == 1)).sum())
        precision = tp / (tp + fp + 1e-8)
        recall = tp / (tp + fn + 1e-8)
        f1 = 2 * precision * recall / (precision + recall + 1e-8)
        if f1 > best["f1"]:
            best = {"threshold": thr, "f1": float(f1), "quantile": float(q)}
    return best


def train_model(train_features, train_labels, val_features, val_labels):
    device = "cuda" if torch.cuda.is_available() else "cpu"
    model = ReliabilityMLP(train_features.shape[1]).to(device)
    opt = torch.optim.AdamW(model.parameters(), lr=LR, weight_decay=1e-4)

    x_train = torch.from_numpy(train_features).float().to(device)
    y_train = torch.from_numpy(train_labels).float().to(device)
    x_val = torch.from_numpy(val_features).float().to(device)
    y_val = torch.from_numpy(val_labels).float().to(device)

    pos = float(train_labels.sum())
    neg = float(len(train_labels) - pos)
    pos_weight = torch.tensor([neg / max(pos, 1.0)], device=device)
    history = []
    rng = np.random.default_rng(SEED)

    for epoch in range(1, EPOCHS + 1):
        model.train()
        order = rng.permutation(len(train_labels))
        losses = []
        for start in range(0, len(order), BATCH_SIZE):
            idx = torch.from_numpy(order[start : start + BATCH_SIZE]).long().to(device)
            logits = model(x_train[idx])
            loss = F.binary_cross_entropy_with_logits(logits, y_train[idx], pos_weight=pos_weight)
            opt.zero_grad(set_to_none=True)
            loss.backward()
            opt.step()
            losses.append(float(loss.detach().cpu()))

        model.eval()
        with torch.no_grad():
            val_prob = torch.sigmoid(model(x_val)).detach().cpu().numpy()
        val_best = probability_grid_threshold(val_prob, val_labels)
        row = {
            "epoch": epoch,
            "train_loss": float(np.mean(losses)),
            "val_best_f1": val_best["f1"],
            "val_prob_threshold": val_best["threshold"],
        }
        history.append(row)
        print("Run 12 train row:", row)

    return model, history


def probability_grid_threshold(prob, labels):
    best = {"threshold": 0.5, "f1": -1.0}
    for thr in np.linspace(0.05, 0.95, 19):
        pred = prob >= thr
        tp = float(((pred == 1) & (labels == 1)).sum())
        fp = float(((pred == 1) & (labels == 0)).sum())
        fn = float(((pred == 0) & (labels == 1)).sum())
        precision = tp / (tp + fp + 1e-8)
        recall = tp / (tp + fn + 1e-8)
        f1 = 2 * precision * recall / (precision + recall + 1e-8)
        if f1 > best["f1"]:
            best = {"threshold": float(thr), "f1": float(f1)}
    return best


def eval_selected(case, mask, method):
    selected = case["points"][mask]
    if len(selected) < 100:
        selected = case["points"]
    metrics = base.compute_metrics(base.downsample(selected, base.MAX_POINTS), case["gt"])
    return {
        "run": RUN_NAME,
        "method": method,
        "scene": case["scene"],
        "num_views": case["view_count"],
        "view_policy": case["policy"],
        "selected_ratio": float(mask.mean()),
        **metrics,
    }


def write_json(path, obj):
    path.write_text(json.dumps(obj, indent=2))


def main():
    set_seed()
    base.require_t4x2()
    root = base.clone_repo()
    base.install_deps(root)
    posed_root = base.find_posed_images_root()
    scene_dirs = base.discover_scene_dirs(posed_root)
    print("POSED_IMAGES:", posed_root)
    print("SCENE_DIRS:", [str(p) for p in scene_dirs])
    if len(scene_dirs) < 2:
        raise RuntimeError("Run 12 needs at least two scenes for train/test split.")

    ckpt_path = base.download_checkpoint(root)
    print("Checkpoint:", ckpt_path)
    model = base.load_model(root, ckpt_path)

    out_dir = Path("/kaggle/working/outputs") / RUN_NAME
    out_dir.mkdir(parents=True, exist_ok=True)

    train_scene = scene_dirs[TRAIN_SCENE_INDEX]
    test_scene = scene_dirs[TEST_SCENE_INDEX]
    train_cases = []
    test_cases = []

    started = time.time()
    for vc in VIEW_COUNTS:
        train_policy = base.BEST_POLICY_BY_VIEW_COUNT[vc]
        test_policy = base.BEST_POLICY_BY_VIEW_COUNT[vc]
        train_cases.append(
            collect_case(
                model,
                root,
                train_scene,
                vc,
                train_policy,
                out_dir / "train" / train_scene.name / f"{vc}_views",
                MAX_TRAIN_POINTS_PER_CASE,
                SEED,
            )
        )
        test_cases.append(
            collect_case(
                model,
                root,
                test_scene,
                vc,
                test_policy,
                out_dir / "test" / test_scene.name / f"{vc}_views",
                MAX_EVAL_POINTS_PER_CASE,
                SEED,
            )
        )

    x = np.concatenate([c["features"] for c in train_cases], axis=0)
    y = np.concatenate([c["labels"] for c in train_cases], axis=0)
    rng = np.random.default_rng(SEED)
    order = rng.permutation(len(y))
    split = int(0.80 * len(order))
    train_idx, val_idx = order[:split], order[split:]

    conf_best = confidence_grid_threshold(
        [
            {
                "conf": np.concatenate([c["conf"][train_idx[:0]] for c in []])
                if False
                else c["conf"],
                "labels": c["labels"],
            }
            for c in train_cases
        ]
    )
    learned_model, history = train_model(x[train_idx], y[train_idx], x[val_idx], y[val_idx])
    learned_model.eval()
    with torch.no_grad():
        val_prob = torch.sigmoid(learned_model(torch.from_numpy(x[val_idx]).float().cuda())).cpu().numpy()
    prob_best = probability_grid_threshold(val_prob, y[val_idx])

    rows = []
    device = "cuda" if torch.cuda.is_available() else "cpu"
    for case in test_cases:
        rows.append(eval_selected(case, case["conf"] >= conf_best["threshold"], "confidence_threshold_val_fixed"))
        with torch.no_grad():
            prob = torch.sigmoid(
                learned_model(torch.from_numpy(case["features"]).float().to(device))
            ).detach().cpu().numpy()
        rows.append(eval_selected(case, prob >= prob_best["threshold"], "OARH_learned_reliability"))
        print("Run 12 latest rows:", rows[-2], rows[-1])

    base.write_csv(out_dir / "metrics.csv", rows)
    base.write_csv(out_dir / "training_history.csv", history)
    torch.save(
        {
            "state_dict": learned_model.state_dict(),
            "feature_dim": int(x.shape[1]),
            "prob_threshold": prob_best["threshold"],
            "confidence_threshold": conf_best["threshold"],
        },
        out_dir / "oarh_reliability_head.pt",
    )
    write_json(
        out_dir / "run_config.json",
        {
            "run": RUN_NAME,
            "train_scene": train_scene.name,
            "test_scene": test_scene.name,
            "view_counts": VIEW_COUNTS,
            "good_distance_m": GOOD_DISTANCE_M,
            "bad_distance_m": BAD_DISTANCE_M,
            "features": [
                "confidence_z",
                "log_confidence",
                "x_norm",
                "y_norm",
                "z_norm",
                "radius_norm",
                "pixel_x_norm",
                "pixel_y_norm",
                "view_id_norm",
            ],
            "confidence_selection": conf_best,
            "learned_selection": prob_best,
            "runtime_seconds": time.time() - started,
            "note": "Stage-A supervised proxy: MV-DUSt3R+ is frozen; only a small reliability MLP is trained.",
        },
    )
    print("Run 12 metrics:")
    for row in rows:
        print(row)


if __name__ == "__main__":
    main()
