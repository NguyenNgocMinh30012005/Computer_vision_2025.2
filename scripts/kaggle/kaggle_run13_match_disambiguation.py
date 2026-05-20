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
RAW_RUN12 = "https://raw.githubusercontent.com/NguyenNgocMinh30012005/Computer_vision_2025.2/main/scripts/kaggle/kaggle_run12_supervised_reliability.py"


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
run12 = ensure_helper_module("kaggle_run12_supervised_reliability", RAW_RUN12)


RUN_NAME = "run_13_match_disambiguation"
VIEW_COUNTS = [3, 4, 5]
TRAIN_SCENE_INDEX = 0
TEST_SCENE_INDEX = 1
MAX_CANDIDATES_PER_CASE = 35000
PAIRS_PER_CASE = 50000
POS_DIST_M = 0.05
NEG_DIST_M = 0.30
EPOCHS = 14
BATCH_SIZE = 8192
LR = 1e-3
SEED = 4321


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


def set_seed(seed=SEED):
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)


def aligned_candidates(case):
    aligned = base.center_scale_align(case["points"], case["gt"])
    tree = cKDTree(case["gt"])
    dists, nn_idx = tree.query(aligned, k=1, workers=-1)
    reliable = dists <= POS_DIST_M
    return aligned.astype(np.float32), dists.astype(np.float32), nn_idx.astype(np.int64), reliable


def build_pair_dataset(case, seed):
    rng = np.random.default_rng(seed)
    aligned, dists, nn_idx, reliable = aligned_candidates(case)
    feats = case["features"]
    conf = case["conf"]
    idx_good = np.flatnonzero(reliable)
    if len(idx_good) < 1000:
        idx_good = np.arange(len(aligned))

    # Positive pairs: nearby candidate points that map close to the same GT surface.
    good_sample = rng.choice(idx_good, min(len(idx_good), 12000), replace=False)
    tree = cKDTree(aligned[good_sample])
    near_pairs = tree.query_pairs(r=POS_DIST_M, output_type="ndarray")
    if len(near_pairs) > 0:
        pos_a = good_sample[near_pairs[:, 0]]
        pos_b = good_sample[near_pairs[:, 1]]
        pos_keep = nn_idx[pos_a] == nn_idx[pos_b]
        pos_a, pos_b = pos_a[pos_keep], pos_b[pos_keep]
    else:
        pos_a = np.array([], dtype=np.int64)
        pos_b = np.array([], dtype=np.int64)

    if len(pos_a) > PAIRS_PER_CASE // 2:
        take = rng.choice(len(pos_a), PAIRS_PER_CASE // 2, replace=False)
        pos_a, pos_b = pos_a[take], pos_b[take]

    # Negative pairs: visually/plausibly sampled candidates far apart in 3D.
    neg_needed = PAIRS_PER_CASE - len(pos_a)
    neg_a = rng.integers(0, len(aligned), size=neg_needed * 3)
    neg_b = rng.integers(0, len(aligned), size=neg_needed * 3)
    far = np.linalg.norm(aligned[neg_a] - aligned[neg_b], axis=1) >= NEG_DIST_M
    neg_a, neg_b = neg_a[far][:neg_needed], neg_b[far][:neg_needed]
    if len(neg_a) < neg_needed:
        extra = neg_needed - len(neg_a)
        neg_a = np.concatenate([neg_a, rng.integers(0, len(aligned), size=extra)])
        neg_b = np.concatenate([neg_b, rng.integers(0, len(aligned), size=extra)])

    a = np.concatenate([pos_a, neg_a])
    b = np.concatenate([pos_b, neg_b])
    labels = np.concatenate([np.ones(len(pos_a), dtype=np.float32), np.zeros(len(neg_a), dtype=np.float32)])
    order = rng.permutation(len(labels))
    a, b, labels = a[order], b[order], labels[order]

    pair_dist = np.linalg.norm(aligned[a] - aligned[b], axis=1)
    feat_abs = np.abs(feats[a] - feats[b])
    feat_prod_conf = (conf[a] * conf[b])[:, None]
    feat_min_conf = np.minimum(conf[a], conf[b])[:, None]
    feat_pair_dist = pair_dist[:, None].astype(np.float32)
    feat_same_view = (np.isclose(feats[a, -1], feats[b, -1])).astype(np.float32)[:, None]
    x = np.concatenate([feat_abs, feat_prod_conf, feat_min_conf, feat_pair_dist, feat_same_view], axis=1).astype(np.float32)
    return x, labels.astype(np.float32)


def train_match_model(x_train, y_train, x_val, y_val):
    device = "cuda" if torch.cuda.is_available() else "cpu"
    model = MatchMLP(x_train.shape[1]).to(device)
    opt = torch.optim.AdamW(model.parameters(), lr=LR, weight_decay=1e-4)
    pos = float(y_train.sum())
    neg = float(len(y_train) - pos)
    pos_weight = torch.tensor([neg / max(pos, 1.0)], device=device)
    x_train_t = torch.from_numpy(x_train).float().to(device)
    y_train_t = torch.from_numpy(y_train).float().to(device)
    x_val_t = torch.from_numpy(x_val).float().to(device)
    y_val_t = torch.from_numpy(y_val).float().to(device)
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
        best = threshold_metrics(prob, y_val)
        row = {"epoch": epoch, "train_loss": float(np.mean(losses)), **best}
        history.append(row)
        print("Run 13 train row:", row)
    return model, history


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


def main():
    set_seed()
    base.require_t4x2()
    root = base.clone_repo()
    base.install_deps(root)
    posed_root = base.find_posed_images_root()
    scene_dirs = base.discover_scene_dirs(posed_root)
    if len(scene_dirs) < 2:
        raise RuntimeError("Run 13 needs at least two scenes for train/test split.")
    print("POSED_IMAGES:", posed_root)
    print("SCENE_DIRS:", [str(p) for p in scene_dirs])

    ckpt_path = base.download_checkpoint(root)
    print("Checkpoint:", ckpt_path)
    model = base.load_model(root, ckpt_path)
    out_dir = Path("/kaggle/working/outputs") / RUN_NAME
    out_dir.mkdir(parents=True, exist_ok=True)

    train_scene = scene_dirs[TRAIN_SCENE_INDEX]
    test_scene = scene_dirs[TEST_SCENE_INDEX]
    train_pairs = []
    test_pairs = []
    started = time.time()

    for vc in VIEW_COUNTS:
        policy = base.BEST_POLICY_BY_VIEW_COUNT[vc]
        train_case = run12.collect_case(
            model,
            root,
            train_scene,
            vc,
            policy,
            out_dir / "train" / train_scene.name / f"{vc}_views",
            MAX_CANDIDATES_PER_CASE,
            SEED,
        )
        test_case = run12.collect_case(
            model,
            root,
            test_scene,
            vc,
            policy,
            out_dir / "test" / test_scene.name / f"{vc}_views",
            MAX_CANDIDATES_PER_CASE,
            SEED,
        )
        train_pairs.append(build_pair_dataset(train_case, SEED + vc))
        test_pairs.append((vc, build_pair_dataset(test_case, SEED + 100 + vc)))

    x_all = np.concatenate([p[0] for p in train_pairs], axis=0)
    y_all = np.concatenate([p[1] for p in train_pairs], axis=0)
    rng = np.random.default_rng(SEED)
    order = rng.permutation(len(y_all))
    split = int(0.80 * len(order))
    train_idx, val_idx = order[:split], order[split:]
    match_model, history = train_match_model(x_all[train_idx], y_all[train_idx], x_all[val_idx], y_all[val_idx])

    device = "cuda" if torch.cuda.is_available() else "cpu"
    with torch.no_grad():
        val_prob = torch.sigmoid(match_model(torch.from_numpy(x_all[val_idx]).float().to(device))).cpu().numpy()
    best = threshold_metrics(val_prob, y_all[val_idx])

    rows = []
    match_model.eval()
    for vc, (x_test, y_test) in test_pairs:
        with torch.no_grad():
            prob = torch.sigmoid(match_model(torch.from_numpy(x_test).float().to(device))).cpu().numpy()
        metrics = threshold_metrics(prob, y_test)
        row = {
            "run": RUN_NAME,
            "scene": test_scene.name,
            "num_views": vc,
            "view_policy": base.BEST_POLICY_BY_VIEW_COUNT[vc],
            "selection_threshold_from_val": best["prob_threshold"],
            **metrics,
            "num_pairs": int(len(y_test)),
            "positive_pair_ratio": float(y_test.mean()),
        }
        rows.append(row)
        print("Run 13 row:", row)

    base.write_csv(out_dir / "match_metrics.csv", rows)
    base.write_csv(out_dir / "training_history.csv", history)
    torch.save(
        {"state_dict": match_model.state_dict(), "feature_dim": int(x_all.shape[1]), "threshold": best["prob_threshold"]},
        out_dir / "rsdh_match_head.pt",
    )
    (out_dir / "run_config.json").write_text(
        json.dumps(
            {
                "run": RUN_NAME,
                "train_scene": train_scene.name,
                "test_scene": test_scene.name,
                "view_counts": VIEW_COUNTS,
                "positive_distance_m": POS_DIST_M,
                "negative_distance_m": NEG_DIST_M,
                "validation_selection": best,
                "runtime_seconds": time.time() - started,
                "note": "Stage-A RSDH proxy: pair labels are generated from GT-depth nearest-surface consistency. This does not yet use MASt3R descriptors.",
            },
            indent=2,
        )
    )
    print("Run 13 metrics:")
    for row in rows:
        print(row)


if __name__ == "__main__":
    main()
