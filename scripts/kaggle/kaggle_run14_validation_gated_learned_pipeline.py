import json
import sys
import time
import urllib.request
from pathlib import Path

import numpy as np
import torch

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


RUN_NAME = "run_14_validation_gated_learned_pipeline"
VIEW_COUNTS = [2, 3, 4, 5]
TRAIN_SCENE_INDEX = 0
TEST_SCENE_INDEX = 1
GATE_MARGIN_F1 = 0.005
SEED = 2026


def score_case(case, mask, method):
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


def predict_prob(model, features):
    device = "cuda" if torch.cuda.is_available() else "cpu"
    model.eval()
    with torch.no_grad():
        x = torch.from_numpy(features).float().to(device)
        return torch.sigmoid(model(x)).detach().cpu().numpy()


def train_oarh(train_cases):
    x = np.concatenate([c["features"] for c in train_cases], axis=0)
    y = np.concatenate([c["labels"] for c in train_cases], axis=0)
    rng = np.random.default_rng(SEED)
    order = rng.permutation(len(y))
    split = int(0.80 * len(order))
    train_idx, val_idx = order[:split], order[split:]

    conf_best = run12.confidence_grid_threshold(train_cases)
    model, history = run12.train_model(x[train_idx], y[train_idx], x[val_idx], y[val_idx])
    val_prob = predict_prob(model, x[val_idx])
    prob_best = run12.probability_grid_threshold(val_prob, y[val_idx])
    return model, history, conf_best, prob_best


def make_gate_decisions(train_cases, model, conf_threshold, prob_threshold):
    decisions = []
    for case in train_cases:
        conf_row = score_case(
            case,
            case["conf"] >= conf_threshold,
            "validation_proxy_confidence_threshold",
        )
        prob = predict_prob(model, case["features"])
        oarh_row = score_case(
            case,
            prob >= prob_threshold,
            "validation_proxy_OARH",
        )
        delta = float(oarh_row["fscore"] - conf_row["fscore"])
        selected = "OARH_learned_reliability" if delta > GATE_MARGIN_F1 else "confidence_threshold_val_fixed"
        row = {
            "run": RUN_NAME,
            "scene": case["scene"],
            "num_views": case["view_count"],
            "view_policy": case["policy"],
            "confidence_fscore": conf_row["fscore"],
            "oarh_fscore": oarh_row["fscore"],
            "delta_fscore": delta,
            "gate_margin_f1": GATE_MARGIN_F1,
            "selected_method": selected,
        }
        decisions.append(row)
        print("Run 14 gate row:", row)
    return decisions


def evaluate_gated(test_cases, model, conf_threshold, prob_threshold, decisions):
    selected_by_view = {int(row["num_views"]): row["selected_method"] for row in decisions}
    rows = []
    for case in test_cases:
        prob = predict_prob(model, case["features"])
        conf_mask = case["conf"] >= conf_threshold
        oarh_mask = prob >= prob_threshold

        conf_row = score_case(case, conf_mask, "confidence_threshold_val_fixed")
        oarh_row = score_case(case, oarh_mask, "OARH_learned_reliability")
        selected_method = selected_by_view[int(case["view_count"])]
        selected_mask = oarh_mask if selected_method == "OARH_learned_reliability" else conf_mask
        gated_row = score_case(case, selected_mask, "validation_gated_learned_pipeline")
        gated_row["selected_method"] = selected_method
        gated_row["candidate_confidence_fscore"] = conf_row["fscore"]
        gated_row["candidate_oarh_fscore"] = oarh_row["fscore"]

        rows.extend([conf_row, oarh_row, gated_row])
        print("Run 14 eval rows:", conf_row, oarh_row, gated_row)
    return rows


def write_json(path, obj):
    path.write_text(json.dumps(obj, indent=2))


def main():
    run12.set_seed(SEED)
    base.require_t4x2()
    root = base.clone_repo()
    base.install_deps(root)
    posed_root = base.find_posed_images_root()
    scene_dirs = base.discover_scene_dirs(posed_root)
    print("POSED_IMAGES:", posed_root)
    print("SCENE_DIRS:", [str(p) for p in scene_dirs])
    if len(scene_dirs) < 2:
        raise RuntimeError("Run 14 needs at least two scenes for train/test split.")

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
        policy = base.BEST_POLICY_BY_VIEW_COUNT[vc]
        train_cases.append(
            run12.collect_case(
                model,
                root,
                train_scene,
                vc,
                policy,
                out_dir / "train_proxy" / train_scene.name / f"{vc}_views",
                run12.MAX_TRAIN_POINTS_PER_CASE,
                SEED,
            )
        )
        test_cases.append(
            run12.collect_case(
                model,
                root,
                test_scene,
                vc,
                policy,
                out_dir / "test" / test_scene.name / f"{vc}_views",
                run12.MAX_EVAL_POINTS_PER_CASE,
                SEED,
            )
        )

    learned_model, history, conf_best, prob_best = train_oarh(train_cases)
    gate_rows = make_gate_decisions(
        train_cases,
        learned_model,
        conf_best["threshold"],
        prob_best["threshold"],
    )
    eval_rows = evaluate_gated(
        test_cases,
        learned_model,
        conf_best["threshold"],
        prob_best["threshold"],
        gate_rows,
    )

    base.write_csv(out_dir / "metrics.csv", eval_rows)
    base.write_csv(out_dir / "gate_decisions.csv", gate_rows)
    base.write_csv(out_dir / "training_history.csv", history)
    torch.save(
        {
            "state_dict": learned_model.state_dict(),
            "feature_dim": int(train_cases[0]["features"].shape[1]),
            "prob_threshold": prob_best["threshold"],
            "confidence_threshold": conf_best["threshold"],
            "gate_margin_f1": GATE_MARGIN_F1,
        },
        out_dir / "validation_gated_oarh.pt",
    )
    write_json(
        out_dir / "run_config.json",
        {
            "run": RUN_NAME,
            "train_scene": train_scene.name,
            "test_scene": test_scene.name,
            "view_counts": VIEW_COUNTS,
            "confidence_selection": conf_best,
            "learned_selection": prob_best,
            "gate_margin_f1": GATE_MARGIN_F1,
            "runtime_seconds": time.time() - started,
            "note": (
                "Validation-gated Stage-A learned pipeline. OARH is used only "
                "for view counts where it beats confidence-only on the train-scene "
                "proxy gate by the configured F-score margin."
            ),
        },
    )
    print("Run 14 metrics:")
    for row in eval_rows:
        print(row)


if __name__ == "__main__":
    main()
