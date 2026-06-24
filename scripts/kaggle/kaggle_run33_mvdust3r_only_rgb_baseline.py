import csv
import json
import math
import os
import time
from pathlib import Path


RUN_NAME = "run_33_mvdust3r_only_rgb_baseline"
RUN30_NAME = "run_30_rgbd_source_depth_correction"
RUN32_NAME = "run_32_direct_rgbd_backprojection_baseline"
GATE_MARGIN_F1 = float(os.environ.get("RUN33_GATE_MARGIN_F1", "0.005"))

METHOD_MAP = {
    "all_candidates": "mvdust3r_raw_all_candidates",
    "confidence_fixed_final": "mvdust3r_confidence_fixed",
}
RUN30_METHOD = "rgbd_source_depth_selected"
RUN32_DIRECT_METHOD = "direct_rgbd_backprojection"
RUN32_SAMPLED_METHOD = "direct_rgbd_backprojection_sampled"


def read_csv_rows(path):
    with Path(path).open(newline="", encoding="utf-8") as handle:
        return list(csv.DictReader(handle))


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
        for row in rows:
            writer.writerow(row)


def as_float(row, key, default=0.0):
    try:
        return float(row.get(key, default))
    except (TypeError, ValueError):
        return default


def mean(rows, key):
    values = [as_float(row, key) for row in rows]
    return sum(values) / max(1, len(values))


def candidate_output_dirs(output_name, env_var):
    if os.environ.get(env_var):
        yield Path(os.environ[env_var])

    owners = ["nguynnminh", "minhhuyen3012nguyen"]
    slugs = {
        RUN30_NAME: "mv-dust3r-run-30-rgbd-source-depth-correction",
        RUN32_NAME: "mv-dust3r-run-32-direct-rgbd-backprojection",
    }
    slug = slugs.get(output_name)
    for owner in owners:
        if slug:
            yield (
                Path("/kaggle/input/notebooks")
                / owner
                / slug
                / "outputs"
                / output_name
            )
            yield Path("/kaggle/input") / slug / "outputs" / output_name

    yield Path("downloads") / output_name
    yield Path("downloads") / f"kaggle_{output_name}" / "outputs" / output_name
    yield Path("tmp") / output_name


def find_output_dir(output_name, env_var):
    for candidate in candidate_output_dirs(output_name, env_var):
        if (candidate / "metrics.csv").exists():
            return candidate

    for root in [Path("/kaggle/input/notebooks"), Path("/kaggle/input"), Path("downloads"), Path("tmp")]:
        if not root.exists():
            continue
        for candidate in root.rglob(output_name):
            if (candidate / "metrics.csv").exists():
                return candidate
    return None


def normalize_metric_row(row):
    original_method = row["method"]
    out = dict(row)
    out["run"] = RUN_NAME
    out["source_run"] = row.get("run", RUN30_NAME)
    out["source_method"] = original_method
    out["method"] = METHOD_MAP[original_method]
    out["method_family"] = "mvdust3r_only_rgb_baseline"
    out["uses_source_depth_for_inference"] = 0
    out["uses_source_depth_for_correction"] = 0
    out["uses_direct_rgbd_backprojection"] = 0
    out["backbone"] = "MV-DUSt3R+"
    out["input_contract"] = "selected sparse RGB views only"
    out["evaluation_contract"] = "existing project reconstruction evaluator"
    out["evaluation_may_use_depth_proxy_target"] = 1
    return out


def load_run30_rgb_only_metrics(run30_dir):
    rows = read_csv_rows(run30_dir / "metrics.csv")
    selected = [
        normalize_metric_row(row)
        for row in rows
        if row.get("method") in METHOD_MAP and row.get("split") in {"val", "test"}
    ]
    if not selected:
        raise RuntimeError(f"No RGB-only baseline rows found in {run30_dir / 'metrics.csv'}")
    return selected


def summarize(metric_rows):
    output = []
    keys = sorted({(row["split"], row["method"]) for row in metric_rows})
    for split, method in keys:
        rows = [row for row in metric_rows if row["split"] == split and row["method"] == method]
        output.append(
            {
                "run": RUN_NAME,
                "split": split,
                "method": method,
                "method_family": rows[0]["method_family"],
                "num_groups": len(rows),
                "mean_fscore": mean(rows, "fscore"),
                "mean_precision": mean(rows, "precision"),
                "mean_recall": mean(rows, "recall"),
                "mean_accuracy": mean(rows, "accuracy"),
                "mean_completeness": mean(rows, "completeness"),
                "mean_chamfer": mean(rows, "chamfer"),
                "uses_source_depth_for_inference": 0,
                "uses_source_depth_for_correction": 0,
                "uses_direct_rgbd_backprojection": 0,
            }
        )
    return output


def limit_summary(metric_rows):
    output = []
    for split in sorted({row["split"] for row in metric_rows}):
        split_rows = [row for row in metric_rows if row["split"] == split]
        group_diag = {}
        for row in split_rows:
            group_diag[row["group_key"]] = (
                as_float(row, "occlusion_proxy_ratio"),
                as_float(row, "ambiguity_proxy_ratio"),
            )
        count = max(1, int(math.ceil(len(group_diag) / 3.0)))
        occ_keys = {
            item[0]
            for item in sorted(group_diag.items(), key=lambda item: (-item[1][0], item[0]))[:count]
        }
        amb_keys = {
            item[0]
            for item in sorted(group_diag.items(), key=lambda item: (-item[1][1], item[0]))[:count]
        }
        subsets = {
            "overall": lambda row: True,
            "occlusion_challenging": lambda row: row["group_key"] in occ_keys,
            "ambiguity_challenging": lambda row: row["group_key"] in amb_keys,
        }
        for subset_name, predicate in subsets.items():
            subset_rows = [row for row in split_rows if predicate(row)]
            for method in sorted({row["method"] for row in subset_rows}):
                rows = [row for row in subset_rows if row["method"] == method]
                output.append(
                    {
                        "run": RUN_NAME,
                        "split": split,
                        "limit_subset": subset_name,
                        "method": method,
                        "method_family": rows[0]["method_family"],
                        "num_groups": len(rows),
                        "mean_fscore": mean(rows, "fscore"),
                        "mean_precision": mean(rows, "precision"),
                        "mean_recall": mean(rows, "recall"),
                        "uses_source_depth_for_inference": 0,
                        "uses_source_depth_for_correction": 0,
                        "uses_direct_rgbd_backprojection": 0,
                    }
                )
    return output


def lookup(rows, split, subset, method):
    matches = [
        row
        for row in rows
        if row.get("split") == split
        and row.get("limit_subset") == subset
        and row.get("method") == method
    ]
    return matches[0] if matches else None


def load_limit_rows(output_dir):
    path = output_dir / "limit_summary.csv"
    return read_csv_rows(path) if path.exists() else []


def read_json(path):
    path = Path(path)
    if not path.exists():
        return {}
    return json.loads(path.read_text(encoding="utf-8"))


def coverage_from_metric_rows(metric_rows, source_config):
    scenes = sorted({row["scene"] for row in metric_rows if row.get("scene")})
    groups = sorted({row["group_key"] for row in metric_rows if row.get("group_key")})
    return {
        "max_eval_groups_raw": source_config.get("max_eval_groups_raw"),
        "max_eval_groups_resolved": source_config.get("max_eval_groups_resolved"),
        "num_discovered_scenes": source_config.get("num_discovered_scenes"),
        "num_total_eval_groups_before_cap": source_config.get(
            "num_total_eval_groups_before_cap"
        ),
        "num_eval_groups_after_cap": len(groups),
        "evaluated_scene_count": len(scenes),
        "evaluated_scene_ids": scenes,
    }


def choose_best_rgb_method(limit_rows):
    val_rows = [
        row
        for row in limit_rows
        if row["split"] == "val" and row["limit_subset"] == "overall"
    ]
    return max(val_rows, key=lambda row: as_float(row, "mean_fscore"))["method"]


def comparison_gate(run33_limits, run30_dir, run32_dir):
    run30_limits = load_limit_rows(run30_dir)
    run32_limits = load_limit_rows(run32_dir) if run32_dir else []
    best_rgb = choose_best_rgb_method(run33_limits)
    row = {
        "run": RUN_NAME,
        "rgb_only_method": best_rgb,
        "run30_method": RUN30_METHOD,
        "run32_direct_method": RUN32_DIRECT_METHOD,
        "run32_sampled_method": RUN32_SAMPLED_METHOD,
        "reused_existing_all_candidates_confidence_fixed_outputs": 1,
        "reran_mvdust3r_inference": 0,
        "uses_source_depth_for_inference": 0,
        "uses_source_depth_for_correction": 0,
        "uses_direct_rgbd_backprojection": 0,
        "gate_margin_f1": GATE_MARGIN_F1,
        "run30_output_dir": str(run30_dir),
        "run32_output_dir": str(run32_dir) if run32_dir else "",
    }
    subsets = ["overall", "occlusion_challenging", "ambiguity_challenging"]
    for split in ["val", "test"]:
        for subset in subsets:
            rgb = lookup(run33_limits, split, subset, best_rgb)
            run30 = lookup(run30_limits, split, subset, RUN30_METHOD)
            direct = lookup(run32_limits, split, subset, RUN32_DIRECT_METHOD)
            sampled = lookup(run32_limits, split, subset, RUN32_SAMPLED_METHOD)
            if rgb is None or run30 is None:
                raise RuntimeError(f"Missing Run 33/Run 30 comparison row for {split}/{subset}")
            prefix = f"{split}_{subset}"
            rgb_f = as_float(rgb, "mean_fscore")
            run30_f = as_float(run30, "mean_fscore")
            row[f"{prefix}_mvdust3r_only_fscore"] = rgb_f
            row[f"{prefix}_run30_fscore"] = run30_f
            row[f"{prefix}_run30_minus_mvdust3r_only"] = run30_f - rgb_f
            if direct is not None:
                row[f"{prefix}_run32_direct_fscore"] = as_float(direct, "mean_fscore")
            if sampled is not None:
                row[f"{prefix}_run32_sampled_diagnostic_fscore"] = as_float(sampled, "mean_fscore")

    val_delta = row["val_overall_run30_minus_mvdust3r_only"]
    if val_delta >= GATE_MARGIN_F1:
        outcome = "run30_adds_value_over_mvdust3r_only"
        final_claim_changed = 0
    elif val_delta <= -GATE_MARGIN_F1:
        outcome = "mvdust3r_only_beats_run30_under_evaluator"
        final_claim_changed = 1
    else:
        outcome = "run30_and_mvdust3r_only_competitive_close"
        final_claim_changed = 1
    row.update(
        {
            "comparison_outcome": outcome,
            "final_claim_changed": final_claim_changed,
            "official_benchmark_claim": 0,
            "evaluator_warning": (
                "Method inference is RGB-only, but the existing project evaluator may "
                "use depth-derived proxy targets. Direct depth diagnostics must not be "
                "treated as official mesh/laser-scan benchmark evidence."
            ),
        }
    )
    return [row]


def main():
    started = time.time()
    out_dir = Path("/kaggle/working/outputs") / RUN_NAME
    out_dir.mkdir(parents=True, exist_ok=True)

    run30_dir = find_output_dir(RUN30_NAME, "RUN33_RUN30_OUTPUT_DIR")
    if run30_dir is None:
        raise RuntimeError("Run 30 output not found. Add it as a Kaggle kernel source.")
    run32_dir = find_output_dir(RUN32_NAME, "RUN33_RUN32_OUTPUT_DIR")

    metric_rows = load_run30_rgb_only_metrics(run30_dir)
    run30_config = read_json(run30_dir / "run_config.json")
    run32_config = read_json(run32_dir / "run_config.json") if run32_dir else {}
    coverage = coverage_from_metric_rows(metric_rows, run30_config)
    summary_rows = summarize(metric_rows)
    limit_rows = limit_summary(metric_rows)
    gate_rows = comparison_gate(limit_rows, run30_dir, run32_dir)
    config = {
        "run": RUN_NAME,
        "purpose": "MV-DUSt3R+ only RGB baseline extracted from existing Run 30 RGB-only baseline rows.",
        "reuse_strategy": "Reuse Run 30 all_candidates and confidence_fixed_final rows; no new MV-DUSt3R+ inference.",
        "source_run30_dir": str(run30_dir),
        "source_run32_dir": str(run32_dir) if run32_dir else None,
        "uses_source_depth_for_inference": False,
        "uses_source_depth_for_correction": False,
        "uses_direct_rgbd_backprojection": False,
        "reran_mvdust3r_inference": False,
        "backbone": "MV-DUSt3R+",
        "input_contract": "selected sparse RGB views only",
        "evaluation_contract": "existing project reconstruction evaluator",
        "evaluation_may_use_depth_proxy_target": True,
        "method_map": METHOD_MAP,
        **coverage,
        "source_run30_coverage": {
            "max_eval_groups_raw": run30_config.get("max_eval_groups_raw"),
            "max_eval_groups_resolved": run30_config.get("max_eval_groups_resolved"),
            "num_discovered_scenes": run30_config.get("num_discovered_scenes"),
            "num_total_eval_groups_before_cap": run30_config.get(
                "num_total_eval_groups_before_cap"
            ),
            "num_eval_groups_after_cap": run30_config.get("num_eval_groups_after_cap"),
            "evaluated_scene_count": run30_config.get("evaluated_scene_count"),
            "evaluated_scene_ids": run30_config.get("evaluated_scene_ids"),
        },
        "source_run32_coverage": {
            "max_eval_groups_raw": run32_config.get("max_eval_groups_raw"),
            "max_eval_groups_resolved": run32_config.get("max_eval_groups_resolved"),
            "num_discovered_scenes": run32_config.get("num_discovered_scenes"),
            "num_total_eval_groups_before_cap": run32_config.get(
                "num_total_eval_groups_before_cap"
            ),
            "num_eval_groups_after_cap": run32_config.get("num_eval_groups_after_cap"),
            "evaluated_scene_count": run32_config.get("evaluated_scene_count"),
            "evaluated_scene_ids": run32_config.get("evaluated_scene_ids"),
        },
        "num_metric_rows": len(metric_rows),
        "num_groups": len({row["group_key"] for row in metric_rows}),
        "gate_margin_f1": GATE_MARGIN_F1,
        "runtime_seconds": time.time() - started,
    }
    write_csv_union(out_dir / "metrics.csv", metric_rows)
    write_csv_union(out_dir / "summary.csv", summary_rows)
    write_csv_union(out_dir / "limit_summary.csv", limit_rows)
    write_csv_union(out_dir / "gate_decision.csv", gate_rows)
    write_csv_union(out_dir / "qualitative_manifest.csv", [])
    (out_dir / "run_config.json").write_text(json.dumps(config, indent=2), encoding="utf-8")

    print("Run 33 config:", config)
    print("Run 33 summary:")
    for item in summary_rows:
        print(item)
    print("Run 33 gate decision:")
    for item in gate_rows:
        print(item)
    print("Run 33 output dir:", out_dir)


if __name__ == "__main__":
    main()
