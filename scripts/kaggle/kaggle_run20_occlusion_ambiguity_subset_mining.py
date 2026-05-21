import csv
import json
import math
import random
import time
from collections import Counter, defaultdict
from pathlib import Path


RUN_NAME = "run_20_occlusion_ambiguity_subset_mining"
SEED = 2020
CORE_OCCLUSION_RATIO = 0.20
BORDERLINE_OCCLUSION_RATIO = 0.15
LOW_VISIBLE_RATIO = 0.03
FAR_BASELINE_M = 1.0
MAX_OARH_ROWS_PER_BUCKET = 25000
MAX_RSDH_ROWS_PER_BUCKET = 25000


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


def find_run19_dir():
    root = Path("/kaggle/input")
    candidates = sorted(root.rglob("label_summary.csv"))
    for path in candidates:
        text = str(path).lower()
        if "run-19" in text or "run_19" in text or "supervised-label-cache" in text:
            run_dir = path.parent
            if (run_dir / "label_cache.csv").exists():
                print("Using Run 19 output:", run_dir)
                return run_dir
    for path in candidates:
        run_dir = path.parent
        if (run_dir / "label_cache.csv").exists():
            print("Using first label cache output:", run_dir)
            return run_dir
    raise FileNotFoundError(
        "Cannot find Run 19 label cache. Add the Run 19 kernel output as a Kaggle kernel source."
    )


def as_float(row, key, default=0.0):
    value = row.get(key, "")
    if value in ("", "nan", "NaN", None):
        return default
    try:
        return float(value)
    except ValueError:
        return default


def as_bool(row, key):
    value = row.get(key)
    if isinstance(value, bool):
        return value
    return str(value).lower() in {"true", "1", "yes"}


def classify_group(row):
    occlusion = as_float(row, "occlusion_ratio")
    visible = as_float(row, "visible_ratio")
    floating = as_float(row, "floating_ratio")
    wrong = as_float(row, "wrong_candidate_ratio")
    baseline = as_float(row, "mean_baseline_m")
    classes = []
    if occlusion >= CORE_OCCLUSION_RATIO:
        classes.append("occlusion_core")
    elif occlusion >= BORDERLINE_OCCLUSION_RATIO:
        classes.append("occlusion_borderline")
    if visible <= LOW_VISIBLE_RATIO and baseline >= FAR_BASELINE_M:
        classes.append("low_overlap_far")
    if floating + wrong >= 0.55:
        classes.append("wrong_depth_hard_negative")
    if not classes:
        classes.append("general_support")
    return classes


def build_group_manifest(summary_rows):
    rows = []
    for row in summary_rows:
        classes = classify_group(row)
        occlusion = as_float(row, "occlusion_ratio")
        visible = as_float(row, "visible_ratio")
        floating = as_float(row, "floating_ratio")
        baseline = as_float(row, "mean_baseline_m")
        priority = occlusion * 3.0 + floating + min(baseline, 4.0) * 0.15 - visible * 0.5
        rows.append(
            {
                **row,
                "run": RUN_NAME,
                "group_classes": "|".join(classes),
                "priority_score": float(priority),
                "use_for_oarh_v2": int(any(c in classes for c in ["occlusion_core", "occlusion_borderline", "wrong_depth_hard_negative"])),
                "use_for_rsdh_v2": int(any(c in classes for c in ["wrong_depth_hard_negative", "low_overlap_far", "occlusion_core"])),
                "is_final_eval_candidate": int(row.get("split") in {"val", "test"} and any(c in classes for c in ["occlusion_core", "occlusion_borderline", "low_overlap_far"])),
            }
        )
    return sorted(rows, key=lambda r: (r["split"], -as_float(r, "priority_score"), r["scene"], int(r["num_views"])))


def row_bucket(row, group_lookup, target):
    group = group_lookup.get(row.get("group_key"), {})
    classes = group.get("group_classes", "general_support").split("|")
    split = row.get("split", "train")
    keep = as_float(row, "keep_label")
    support = as_float(row, "support_label")
    occluded = as_float(row, "occlusion_label")
    floating = as_float(row, "floating_label")
    match = as_float(row, "match_label")
    in_bounds = as_float(row, "target_in_bounds")

    if target == "oarh":
        if keep > 0.5 and support > 0.5:
            label = "keep_visible_positive"
        elif keep > 0.5 and occluded > 0.5:
            label = "keep_occluded_positive"
        elif floating > 0.5 or keep < 0.5:
            label = "reject_wrong_depth_negative"
        else:
            label = "unknown_or_out_of_view"
        if label == "unknown_or_out_of_view":
            return None
        hard_class = "occlusion" if any(c.startswith("occlusion") for c in classes) else "general"
        return (split, label, hard_class)

    if target == "rsdh":
        if match > 0.5:
            label = "match_positive"
        elif in_bounds > 0.5 and (floating > 0.5 or keep < 0.5 or occluded > 0.5):
            label = "hard_match_negative"
        else:
            return None
        hard_class = "low_overlap_far" if "low_overlap_far" in classes else "geometry_hard"
        return (split, label, hard_class)

    raise ValueError(target)


def reservoir_add(reservoir, counts, key, row, cap, rng):
    counts[key] += 1
    bucket = reservoir[key]
    if len(bucket) < cap:
        bucket.append(row)
        return
    j = rng.randrange(counts[key])
    if j < cap:
        bucket[j] = row


def slim_label_row(row, group_lookup, bucket):
    group = group_lookup.get(row.get("group_key"), {})
    keys = [
        "split",
        "scene",
        "num_views",
        "view_policy",
        "group_key",
        "source_image",
        "target_image",
        "candidate_type",
        "keep_label",
        "visibility_label",
        "support_label",
        "occlusion_label",
        "floating_label",
        "match_label",
        "src_x_norm",
        "src_y_norm",
        "src_depth_m",
        "candidate_depth_m",
        "candidate_depth_delta_m",
        "target_x_norm",
        "target_y_norm",
        "projected_target_depth_m",
        "observed_target_depth_m",
        "depth_residual_m",
        "target_in_bounds",
        "baseline_m",
    ]
    out = {k: row.get(k, "") for k in keys}
    out.update(
        {
            "run": RUN_NAME,
            "sample_bucket": "|".join(map(str, bucket)),
            "group_classes": group.get("group_classes", "general_support"),
            "group_priority_score": group.get("priority_score", ""),
        }
    )
    return out


def sample_label_subsets(label_cache_path, group_manifest):
    group_lookup = {row["group_key"]: row for row in group_manifest}
    rng = random.Random(SEED)
    oarh_reservoir = defaultdict(list)
    rsdh_reservoir = defaultdict(list)
    oarh_counts = Counter()
    rsdh_counts = Counter()
    total = 0

    with label_cache_path.open(newline="") as f:
        reader = csv.DictReader(f)
        for row in reader:
            total += 1
            if total % 500000 == 0:
                print("Run 20 streamed labels:", total)
            oarh_bucket = row_bucket(row, group_lookup, "oarh")
            if oarh_bucket is not None:
                reservoir_add(
                    oarh_reservoir,
                    oarh_counts,
                    oarh_bucket,
                    slim_label_row(row, group_lookup, oarh_bucket),
                    MAX_OARH_ROWS_PER_BUCKET,
                    rng,
                )
            rsdh_bucket = row_bucket(row, group_lookup, "rsdh")
            if rsdh_bucket is not None:
                reservoir_add(
                    rsdh_reservoir,
                    rsdh_counts,
                    rsdh_bucket,
                    slim_label_row(row, group_lookup, rsdh_bucket),
                    MAX_RSDH_ROWS_PER_BUCKET,
                    rng,
                )

    oarh_rows = [row for key in sorted(oarh_reservoir) for row in oarh_reservoir[key]]
    rsdh_rows = [row for key in sorted(rsdh_reservoir) for row in rsdh_reservoir[key]]
    count_rows = []
    for key, count in sorted(oarh_counts.items()):
        count_rows.append({"run": RUN_NAME, "target": "oarh_v2", "bucket": "|".join(map(str, key)), "available_rows": count, "sampled_rows": len(oarh_reservoir[key])})
    for key, count in sorted(rsdh_counts.items()):
        count_rows.append({"run": RUN_NAME, "target": "rsdh_v2", "bucket": "|".join(map(str, key)), "available_rows": count, "sampled_rows": len(rsdh_reservoir[key])})
    return oarh_rows, rsdh_rows, count_rows, total


def split_summary(rows):
    counts = Counter((row["split"], row["group_classes"].split("|")[0]) for row in rows)
    out = []
    for (split, cls), count in sorted(counts.items()):
        out.append({"run": RUN_NAME, "split": split, "primary_group_class": cls, "num_groups": count})
    return out


def main():
    started = time.time()
    out_dir = Path("/kaggle/working/outputs") / RUN_NAME
    out_dir.mkdir(parents=True, exist_ok=True)
    run19_dir = find_run19_dir()
    summary_rows = read_csv(run19_dir / "label_summary.csv")
    manifest_rows = read_csv(run19_dir / "view_group_manifest.csv") if (run19_dir / "view_group_manifest.csv").exists() else []
    scene_split_rows = read_csv(run19_dir / "scene_split.csv") if (run19_dir / "scene_split.csv").exists() else []

    group_manifest = build_group_manifest(summary_rows)
    final_eval_groups = [r for r in group_manifest if int(r["is_final_eval_candidate"]) == 1]
    oarh_groups = [r for r in group_manifest if int(r["use_for_oarh_v2"]) == 1]
    rsdh_groups = [r for r in group_manifest if int(r["use_for_rsdh_v2"]) == 1]
    oarh_rows, rsdh_rows, count_rows, total_labels = sample_label_subsets(run19_dir / "label_cache.csv", group_manifest)

    write_csv_union(out_dir / "subset_group_manifest.csv", group_manifest)
    write_csv_union(out_dir / "final_eval_group_manifest.csv", final_eval_groups)
    write_csv_union(out_dir / "oarh_v2_group_manifest.csv", oarh_groups)
    write_csv_union(out_dir / "rsdh_v2_group_manifest.csv", rsdh_groups)
    write_csv_union(out_dir / "oarh_v2_balanced_labels.csv", oarh_rows)
    write_csv_union(out_dir / "rsdh_v2_hard_negative_labels.csv", rsdh_rows)
    write_csv_union(out_dir / "sample_bucket_counts.csv", count_rows)
    write_csv_union(out_dir / "group_class_split_summary.csv", split_summary(group_manifest))
    write_csv_union(out_dir / "run19_view_group_manifest_copy.csv", manifest_rows)
    write_csv_union(out_dir / "run19_scene_split_copy.csv", scene_split_rows)

    config = {
        "run": RUN_NAME,
        "source_run19_dir": str(run19_dir),
        "total_label_rows_streamed": total_labels,
        "num_groups": len(group_manifest),
        "num_final_eval_groups": len(final_eval_groups),
        "num_oarh_groups": len(oarh_groups),
        "num_rsdh_groups": len(rsdh_groups),
        "num_oarh_sample_rows": len(oarh_rows),
        "num_rsdh_sample_rows": len(rsdh_rows),
        "thresholds": {
            "core_occlusion_ratio": CORE_OCCLUSION_RATIO,
            "borderline_occlusion_ratio": BORDERLINE_OCCLUSION_RATIO,
            "low_visible_ratio": LOW_VISIBLE_RATIO,
            "far_baseline_m": FAR_BASELINE_M,
        },
        "next_runs": {
            "run21": "Train OARH v2 from oarh_v2_balanced_labels.csv and evaluate on final_eval_group_manifest.csv.",
            "run24": "Train RSDH v2 from image-only MASt3R features using rsdh_v2_hard_negative_labels.csv as labels.",
        },
        "runtime_seconds": time.time() - started,
    }
    (out_dir / "run_config.json").write_text(json.dumps(config, indent=2))

    print("Run 20 config:")
    print(config)
    print("Run 20 top final eval groups:")
    for row in final_eval_groups[:24]:
        print(row)
    print("Run 20 sample buckets:")
    for row in count_rows[:36]:
        print(row)


if __name__ == "__main__":
    main()
