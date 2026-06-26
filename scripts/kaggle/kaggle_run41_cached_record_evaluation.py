import csv
import json
import os
import sys
import time
import urllib.request
from pathlib import Path

import numpy as np


RAW_RUN36 = "https://raw.githubusercontent.com/NguyenNgocMinh30012005/Computer_vision_2025.2/main/scripts/kaggle/kaggle_run36_predicted_depth_correction.py"


def ensure_helper_module(module_name, raw_url):
    try:
        return __import__(module_name)
    except ModuleNotFoundError:
        helper_path = Path(f"{module_name}.py")
        print(f"Downloading helper module {module_name} from GitHub raw")
        urllib.request.urlretrieve(raw_url, helper_path)
        sys.path.insert(0, str(Path.cwd()))
        return __import__(module_name)


r36 = ensure_helper_module(
    "kaggle_run36_predicted_depth_correction",
    RAW_RUN36,
)
base = r36.base

RUN_NAME = "run_41_cached_record_evaluation"
SCALE_MODES = ["raw"]
r36.RUN_NAME = RUN_NAME
r36.SCALE_MODES = SCALE_MODES


def write_csv_union(path, rows):
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


def discover_record_caches():
    roots = []
    for cache_root in sorted(Path("/kaggle/input").rglob("record_cache")):
        if not (cache_root / "eval").exists():
            continue
        run_config = cache_root.parent / "run_config.json"
        run_name = cache_root.parent.name
        if run_config.exists():
            try:
                run_name = json.loads(
                    run_config.read_text(encoding="utf-8")
                ).get("run", run_name)
            except json.JSONDecodeError:
                pass
        if "run_38" in run_name:
            label = "finetuned"
        elif "run_39" in run_name:
            label = "pretrained"
        else:
            label = run_name
        roots.append({"label": label, "run": run_name, "path": cache_root})
    if not roots:
        raise FileNotFoundError(
            "No record_cache directories found under /kaggle/input. "
            "Attach successful Run 38/Run 39 outputs as kernel sources."
        )
    return roots


def load_record(meta_path):
    meta_path = Path(meta_path)
    metadata = json.loads(meta_path.read_text(encoding="utf-8"))
    npz_path = meta_path.with_suffix(".npz")
    arrays = np.load(npz_path)
    method_inputs = {}
    for mode in metadata["scale_modes"]:
        prefix = f"{mode}__"
        method_inputs[mode] = {
            "targets": arrays[prefix + "targets"].astype(np.float32),
            "valid": arrays[prefix + "valid"].astype(bool),
            "sampled_depth": arrays[prefix + "sampled_depth"].astype(
                np.float32
            ),
            "direct_cloud": arrays[prefix + "direct_cloud"].astype(
                np.float32
            ),
        }
    gt = arrays["gt"].astype(np.float32)
    return {
        "group": metadata["group"],
        "points": arrays["points"].astype(np.float32),
        "conf": arrays["conf"].astype(np.float32),
        "method_inputs": method_inputs,
        "gt": gt,
        "gt_tree": r36.cKDTree(gt),
        "occlusion_proxy": arrays["occlusion_proxy"].astype(np.float32),
        "ambiguity_proxy": arrays["ambiguity_proxy"].astype(np.float32),
        "photo_stats": metadata["photo_stats"],
        "runtime_seconds": float(metadata["runtime_seconds"]),
        "view_files": metadata.get("view_files", []),
        "cache_npz": str(npz_path),
    }


def load_cache_records(cache_root):
    cache_root = Path(cache_root)
    output = {}
    for split_name in ["internal_val", "eval"]:
        records = [
            load_record(path)
            for path in sorted((cache_root / split_name).glob("*.json"))
        ]
        output[split_name] = records
    return output


def score_points(points, record, method, family, extra):
    group = record["group"]
    return {
        "run": RUN_NAME,
        "split": group["split"],
        "scene": group["scene"],
        "num_views": int(group["num_views"]),
        "view_policy": group["view_policy"],
        "group_key": group["group_key"],
        "method": method,
        "method_family": family,
        "occlusion_proxy_ratio": float(record["occlusion_proxy"].mean()),
        "ambiguity_proxy_ratio": float(record["ambiguity_proxy"].mean()),
        "runtime_seconds": record["runtime_seconds"],
        "uses_true_source_depth_for_inference": 0,
        "uses_predicted_depth_for_inference": int(
            family != "rgb_only_baseline"
        ),
        "uses_known_pose": 1,
        "uses_known_intrinsics": 1,
        **record["photo_stats"],
        **extra,
        **r36.compute_metrics(points, record["gt"]),
    }


def evaluate_cached_record(
    record,
    selected,
    selected_by_mode,
    label,
    include_rgb_baselines,
    include_direct,
):
    rows = []
    if include_rgb_baselines:
        conf_mask, conf_percent, conf_threshold = r36.confidence_mask(
            record["conf"],
            int(record["group"]["num_views"]),
        )
        rows.extend(
            [
                score_points(
                    record["points"],
                    record,
                    "mvdust3r_rgb_only_all_candidates",
                    "rgb_only_baseline",
                    {"selected_ratio": 1.0},
                ),
                score_points(
                    record["points"][conf_mask],
                    record,
                    "mvdust3r_confidence_fixed",
                    "rgb_only_baseline",
                    {
                        "selected_ratio": float(conf_mask.mean()),
                        "conf_percent": conf_percent,
                        "conf_threshold": conf_threshold,
                    },
                ),
            ]
        )

    for mode in SCALE_MODES:
        policy = selected_by_mode[mode]
        corrected, mask, residual = r36.corrected_points(
            record,
            mode,
            policy["tau_pred"],
            policy["alpha"],
        )
        valid = record["method_inputs"][mode]["valid"]
        rows.append(
            score_points(
                corrected,
                record,
                f"{label}_estimated_depth_correction",
                "predicted_depth_correction",
                {
                    "depth_scale_mode": mode,
                    "tau_pred": policy["tau_pred"],
                    "alpha": policy["alpha"],
                    "correction_ratio": float(mask.mean()),
                    "valid_predicted_depth_ratio": float(valid.mean()),
                    "mean_predicted_residual": float(residual[valid].mean())
                    if valid.any()
                    else float("nan"),
                },
            )
        )

    if include_direct:
        mode = selected["depth_scale_mode"]
        rows.append(
            score_points(
                record["method_inputs"][mode]["direct_cloud"],
                record,
                f"{label}_depth_direct_backprojection",
                "predicted_depth_direct",
                {
                    "depth_scale_mode": mode,
                    "tau_pred": "",
                    "alpha": "",
                    "correction_ratio": "",
                    "valid_predicted_depth_ratio": float(
                        record["method_inputs"][mode]["valid"].mean()
                    ),
                },
            )
        )
    return rows


def main():
    started = time.time()
    out_dir = Path("/kaggle/working/outputs") / RUN_NAME
    out_dir.mkdir(parents=True, exist_ok=True)

    cache_roots = discover_record_caches()
    cache_manifest = [
        {
            "run": RUN_NAME,
            "cache_label": item["label"],
            "source_run": item["run"],
            "cache_path": str(item["path"]),
        }
        for item in cache_roots
    ]
    write_csv_union(out_dir / "cache_manifest.csv", cache_manifest)

    all_metric_rows = []
    policy_rows = []
    for item in cache_roots:
        label = item["label"]
        records = load_cache_records(item["path"])
        internal_records = records["internal_val"]
        eval_records = records["eval"]
        if not internal_records or not eval_records:
            raise RuntimeError(
                f"Cache {item['path']} must contain internal_val and eval records."
            )
        selected, selected_by_mode, rows = r36.select_policies(
            internal_records
        )
        for row in rows:
            row["cache_label"] = label
            row["source_run"] = item["run"]
        policy_rows.extend(rows)
        include_rgb = label == "finetuned"
        include_direct = label == "finetuned"
        for record in eval_records:
            all_metric_rows.extend(
                evaluate_cached_record(
                    record,
                    selected,
                    selected_by_mode,
                    label,
                    include_rgb_baselines=include_rgb,
                    include_direct=include_direct,
                )
            )

    summary_rows = r36.summarize(all_metric_rows)
    limit_rows = r36.limit_summary(all_metric_rows)
    write_csv_union(out_dir / "metrics.csv", all_metric_rows)
    write_csv_union(out_dir / "summary.csv", summary_rows)
    write_csv_union(out_dir / "limit_summary.csv", limit_rows)
    write_csv_union(out_dir / "policy_selection.csv", policy_rows)
    config = {
        "run": RUN_NAME,
        "purpose": (
            "Evaluate cached MV-DUSt3R+ candidate/depth records without "
            "rerunning MV-DUSt3R+ or depth-estimator inference."
        ),
        "cache_sources": cache_manifest,
        "elapsed_seconds": time.time() - started,
    }
    (out_dir / "run_config.json").write_text(
        json.dumps(config, indent=2),
        encoding="utf-8",
    )
    print("Run 41 config:", config)


if __name__ == "__main__":
    main()
