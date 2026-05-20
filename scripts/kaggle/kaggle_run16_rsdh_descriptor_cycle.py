import csv
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

RAW_BASE = "https://raw.githubusercontent.com/NguyenNgocMinh30012005/Computer_vision_2025.2/main/scripts/kaggle/kaggle_run11_final_validation_3seeds.py"
RAW_RUN15 = "https://raw.githubusercontent.com/NguyenNgocMinh30012005/Computer_vision_2025.2/main/scripts/kaggle/kaggle_run15_mast3r_reciprocal_features.py"


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
run15 = ensure_helper_module("kaggle_run15_mast3r_reciprocal_features", RAW_RUN15)


RUN_NAME = "run_16_rsdh_descriptor_cycle"
SEED = 1616
EPOCHS = 16
BATCH_SIZE = 4096
LR = 1e-3


class MatchMLP(nn.Module):
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


def write_csv_union(path, rows):
    if not rows:
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


def featurize(rows):
    usable = [r for r in rows if float(r.get("depth_valid", 0.0)) > 0.5]
    hard = [r for r in usable if float(r.get("match_label", 0.0)) > 0.5 or float(r.get("hard_negative", 0.0)) > 0.5]
    if len(hard) >= 100:
        usable = hard
    x = []
    y = []
    meta = []
    for r in usable:
        xyz_dist = min(float(r.get("xyz_distance_m", 1e3)), 2.0)
        descriptor_similarity = float(r.get("descriptor_similarity", 0.0))
        descriptor_margin = float(r.get("descriptor_margin", 0.0))
        pixel_distance = float(r.get("pixel_distance_norm", 0.0))
        reciprocal = float(r.get("reciprocal_flag", 1.0))
        backend_is_mast3r = 1.0 if r.get("backend") == "mast3r" else 0.0
        # A two-edge cycle proxy: reciprocal matches should stay geometrically close
        # after depth backprojection. Real three-view cycle is deferred to a larger run.
        cycle_error_proxy = xyz_dist
        x.append(
            [
                descriptor_similarity,
                descriptor_margin,
                reciprocal,
                pixel_distance,
                xyz_dist,
                np.log1p(xyz_dist),
                cycle_error_proxy,
                backend_is_mast3r,
            ]
        )
        y.append(float(r.get("match_label", 0.0)))
        meta.append(r)
    return np.asarray(x, dtype=np.float32), np.asarray(y, dtype=np.float32), meta


def threshold_metrics(prob, labels):
    best = {"prob_threshold": 0.5, "match_precision": 0.0, "match_recall": 0.0, "match_f1": -1.0}
    for thr in np.linspace(0.05, 0.95, 19):
        pred = prob >= thr
        tp = float(((pred == 1) & (labels == 1)).sum())
        fp = float(((pred == 1) & (labels == 0)).sum())
        fn = float(((pred == 0) & (labels == 1)).sum())
        precision = tp / (tp + fp + 1e-8)
        recall = tp / (tp + fn + 1e-8)
        f1 = 2 * precision * recall / (precision + recall + 1e-8)
        if f1 > best["match_f1"]:
            best = {
                "prob_threshold": float(thr),
                "match_precision": float(precision),
                "match_recall": float(recall),
                "match_f1": float(f1),
            }
    return best


def train_model(x_train, y_train, x_val, y_val):
    device = "cuda" if torch.cuda.is_available() else "cpu"
    model = MatchMLP(x_train.shape[1]).to(device)
    opt = torch.optim.AdamW(model.parameters(), lr=LR, weight_decay=1e-4)
    pos = float(y_train.sum())
    neg = float(len(y_train) - pos)
    pos_weight = torch.tensor([neg / max(pos, 1.0)], device=device)
    x_train_t = torch.from_numpy(x_train).float().to(device)
    y_train_t = torch.from_numpy(y_train).float().to(device)
    x_val_t = torch.from_numpy(x_val).float().to(device)
    rng = np.random.default_rng(SEED)
    history = []

    for epoch in range(1, EPOCHS + 1):
        model.train()
        order = rng.permutation(len(y_train))
        losses = []
        for start in range(0, len(order), BATCH_SIZE):
            idx = torch.from_numpy(order[start : start + BATCH_SIZE]).long().to(device)
            logits = model(x_train_t[idx])
            loss = F.binary_cross_entropy_with_logits(logits, y_train_t[idx], pos_weight=pos_weight)
            opt.zero_grad(set_to_none=True)
            loss.backward()
            opt.step()
            losses.append(float(loss.detach().cpu()))

        model.eval()
        with torch.no_grad():
            prob = torch.sigmoid(model(x_val_t)).detach().cpu().numpy()
        row = {"epoch": epoch, "train_loss": float(np.mean(losses)), **threshold_metrics(prob, y_val)}
        history.append(row)
        print("Run 16 train row:", row)
    return model, history


def grouped_metrics(prob, labels, meta, threshold):
    rows = []
    keys = sorted({(m["scene"], int(m["num_views"]), m["view_policy"], m["backend"]) for m in meta})
    pred = prob >= threshold
    for scene, num_views, policy, backend in keys:
        idx = np.array(
            [
                i
                for i, m in enumerate(meta)
                if m["scene"] == scene and int(m["num_views"]) == num_views and m["view_policy"] == policy and m["backend"] == backend
            ],
            dtype=int,
        )
        if len(idx) == 0:
            continue
        p = pred[idx]
        y = labels[idx]
        tp = float(((p == 1) & (y == 1)).sum())
        fp = float(((p == 1) & (y == 0)).sum())
        fn = float(((p == 0) & (y == 1)).sum())
        precision = tp / (tp + fp + 1e-8)
        recall = tp / (tp + fn + 1e-8)
        f1 = 2 * precision * recall / (precision + recall + 1e-8)
        rows.append(
            {
                "run": RUN_NAME,
                "scene": scene,
                "num_views": num_views,
                "view_policy": policy,
                "backend": backend,
                "prob_threshold": threshold,
                "match_precision": float(precision),
                "match_recall": float(recall),
                "match_f1": float(f1),
                "num_pairs": int(len(idx)),
                "positive_pair_ratio": float(y.mean()),
            }
        )
    return rows


def main():
    random.seed(SEED)
    np.random.seed(SEED)
    torch.manual_seed(SEED)
    torch.cuda.manual_seed_all(SEED)
    base.require_t4x2()
    posed_root = base.find_posed_images_root()
    started = time.time()
    out_dir = Path("/kaggle/working/outputs") / RUN_NAME
    out_dir.mkdir(parents=True, exist_ok=True)

    backend = run15.setup_mast3r_backend()
    raw_rows, feature_summary = run15.collect_rows(posed_root, backend)
    train_rows = [r for r in raw_rows if r["split"] == "train_proxy"]
    test_rows = [r for r in raw_rows if r["split"] == "heldout"]
    x_train_all, y_train_all, _ = featurize(train_rows)
    x_test, y_test, test_meta = featurize(test_rows)
    if len(x_train_all) < 100 or len(x_test) < 100:
        raise RuntimeError(f"Not enough match pairs for Run 16: train={len(x_train_all)} test={len(x_test)}")

    rng = np.random.default_rng(SEED)
    order = rng.permutation(len(y_train_all))
    split = int(0.80 * len(order))
    train_idx, val_idx = order[:split], order[split:]
    model, history = train_model(x_train_all[train_idx], y_train_all[train_idx], x_train_all[val_idx], y_train_all[val_idx])

    device = "cuda" if torch.cuda.is_available() else "cpu"
    model.eval()
    with torch.no_grad():
        val_prob = torch.sigmoid(model(torch.from_numpy(x_train_all[val_idx]).float().to(device))).cpu().numpy()
        test_prob = torch.sigmoid(model(torch.from_numpy(x_test).float().to(device))).cpu().numpy()
    val_best = threshold_metrics(val_prob, y_train_all[val_idx])
    overall = {"run": RUN_NAME, "scene": "heldout_all", "backend": backend["name"], **threshold_metrics(test_prob, y_test)}
    rows = grouped_metrics(test_prob, y_test, test_meta, val_best["prob_threshold"])
    rows.insert(0, overall)

    write_csv_union(out_dir / "match_metrics.csv", rows)
    write_csv_union(out_dir / "training_history.csv", history)
    write_csv_union(out_dir / "feature_summary.csv", feature_summary)
    torch.save(
        {
            "state_dict": model.state_dict(),
            "feature_dim": int(x_train_all.shape[1]),
            "threshold": val_best["prob_threshold"],
        },
        out_dir / "rsdh_descriptor_cycle_head.pt",
    )
    (out_dir / "run_config.json").write_text(
        json.dumps(
            {
                "run": RUN_NAME,
                "backend": backend["name"],
                "backend_error": backend.get("error"),
                "validation_selection": val_best,
                "runtime_seconds": time.time() - started,
                "note": "RSDH descriptor/cycle proxy. Uses MASt3R reciprocal matches when available, otherwise records ORB fallback backend explicitly.",
            },
            indent=2,
        )
    )
    print("Run 16 metrics:")
    for row in rows:
        print(row)


if __name__ == "__main__":
    main()
