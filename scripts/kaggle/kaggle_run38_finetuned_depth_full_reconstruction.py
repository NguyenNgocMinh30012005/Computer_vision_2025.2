import json
import os
import random
import shutil
import subprocess
import sys
import time
import urllib.request
from collections import OrderedDict
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
r27 = r36.r27
base = r36.base
torch = r36.torch

RUN_NAME = "run_38_finetuned_depth_full_reconstruction"
SEED = int(os.environ.get("RUN38_SEED", "3838"))
MAX_SCENES_RAW = os.environ.get("RUN38_MAX_SCENES", "0")
MAX_EVAL_SCENES_RAW = os.environ.get("RUN38_MAX_EVAL_SCENES", "0")
MAX_EVAL_GROUPS_RAW = os.environ.get("RUN38_MAX_EVAL_GROUPS", "0")
MAX_TRAIN_GROUPS = int(os.environ.get("RUN38_MAX_TRAIN_GROUPS", "48"))
MAX_CANDIDATES_PER_GROUP = int(
    os.environ.get("RUN38_MAX_CANDIDATES_PER_GROUP", "3500")
)
CHECKPOINT_VARIANT = os.environ.get(
    "RUN38_DEPTH_CHECKPOINT",
    "controlled_best",
)
RETAIN_GROUP_OUTPUTS = int(
    os.environ.get("RUN38_RETAIN_GROUP_OUTPUTS", "6")
)
SAVE_EVERY_GROUPS = int(os.environ.get("RUN38_SAVE_EVERY_GROUPS", "25"))
GATE_MARGIN_F1 = float(os.environ.get("RUN38_GATE_MARGIN_F1", "0.005"))


def parse_optional_positive_limit(raw_value):
    if raw_value is None:
        return None
    text = str(raw_value).strip().lower()
    if text in {"", "none", "null", "all", "unlimited"}:
        return None
    value = int(text)
    return value if value > 0 else None


MAX_SCENES = parse_optional_positive_limit(MAX_SCENES_RAW)
MAX_EVAL_SCENES = parse_optional_positive_limit(MAX_EVAL_SCENES_RAW)
MAX_EVAL_GROUPS = parse_optional_positive_limit(MAX_EVAL_GROUPS_RAW)


def install_depth_dependencies():
    try:
        from PIL import Image
        from transformers import AutoImageProcessor, AutoModelForDepthEstimation

        return Image, AutoImageProcessor, AutoModelForDepthEstimation
    except Exception as exc:
        print("Installing Run 38 depth dependencies:", repr(exc))
        subprocess.check_call(
            [
                sys.executable,
                "-m",
                "pip",
                "install",
                "-q",
                "--no-cache-dir",
                "transformers>=4.45.0",
                "huggingface_hub>=0.24.0",
                "safetensors",
                "accelerate",
                "timm",
            ]
        )
        from PIL import Image
        from transformers import AutoImageProcessor, AutoModelForDepthEstimation

        return Image, AutoImageProcessor, AutoModelForDepthEstimation


def locate_run37_output():
    matches = sorted(
        Path("/kaggle/input").rglob(
            "run_37_depth_estimator_full_finetune/run_config.json"
        )
    )
    if not matches:
        raise FileNotFoundError(
            "Run 37 output is not mounted. Attach "
            "nguynnminh/mv-dust3r-run-37-depth-full-fine-tune "
            "as a Kaggle kernel source."
        )
    return matches[0].parent


def read_scene_splits(run37_dir):
    rows = r36.read_csv(Path(run37_dir) / "scene_split.csv")
    splits = {row["scene"]: row["split"] for row in rows}
    if not splits:
        raise RuntimeError("Run 37 scene_split.csv is empty.")
    return splits


class FineTunedDepthPredictor:
    def __init__(
        self,
        checkpoint_dir,
        posed_root,
        Image,
        AutoImageProcessor,
        AutoModelForDepthEstimation,
        cache_size=16,
    ):
        self.checkpoint_dir = Path(checkpoint_dir)
        self.posed_root = Path(posed_root)
        self.Image = Image
        self.device = torch.device("cuda")
        self.processor = AutoImageProcessor.from_pretrained(
            self.checkpoint_dir,
            local_files_only=True,
        )
        self.model = AutoModelForDepthEstimation.from_pretrained(
            self.checkpoint_dir,
            local_files_only=True,
        ).to(self.device)
        self.model.eval()
        self.cache_size = max(int(cache_size), 1)
        self.cache = OrderedDict()
        self.config = {
            "depth_model_name": (
                "depth-anything/Depth-Anything-V2-Metric-Indoor-Small-hf"
            ),
            "depth_checkpoint": str(self.checkpoint_dir),
        }

    def get(self, scene, frame):
        key = (scene, frame)
        if key in self.cache:
            self.cache.move_to_end(key)
            return self.cache[key]
        image_path = self.posed_root / scene / frame
        if not image_path.exists():
            raise FileNotFoundError(f"Missing RGB frame: {image_path}")
        image = self.Image.open(image_path).convert("RGB")
        inputs = self.processor(images=image, return_tensors="pt")
        pixel_values = inputs["pixel_values"].to(
            self.device,
            non_blocking=True,
        )
        with torch.inference_mode():
            predicted = self.model(
                pixel_values=pixel_values
            ).predicted_depth
        depth = predicted[0].float().cpu().numpy().astype(np.float32)
        self.cache[key] = depth
        self.cache.move_to_end(key)
        while len(self.cache) > self.cache_size:
            self.cache.popitem(last=False)
        return depth


def discover_all_scene_dirs(posed_root):
    scene_dirs = sorted(
        path
        for path in Path(posed_root).glob("scene*")
        if path.is_dir() and any(path.glob("*.jpg"))
    )
    if not scene_dirs:
        raise FileNotFoundError(
            f"No scene directories with JPG frames under {posed_root}"
        )
    return scene_dirs


def build_record(
    backbone,
    root,
    scene_lookup,
    group,
    group_dir,
    depth_predictor,
    retain_output,
):
    record = r36.build_record(
        backbone,
        root,
        scene_lookup,
        group,
        group_dir,
        depth_predictor,
        global_scale=1.0,
    )
    if not retain_output and Path(group_dir).exists():
        shutil.rmtree(group_dir)
    return record


def gate_decision(limit_rows, selected):
    selected_method = "predicted_depth_correction_raw"
    comparisons = {}
    for subset in [
        "overall",
        "occlusion_challenging",
        "ambiguity_challenging",
    ]:
        baselines = [
            row
            for row in limit_rows
            if row["split"] == "val"
            and row["limit_subset"] == subset
            and row["method_family"] == "rgb_only_baseline"
        ]
        baseline = max(
            baselines,
            key=lambda row: row["mean_fscore"],
        )
        predicted = r36.find_limit_row(
            limit_rows,
            "val",
            subset,
            selected_method,
        )
        comparisons[subset] = (baseline, predicted)

    overall_base, overall_pred = comparisons["overall"]
    occ_base, occ_pred = comparisons["occlusion_challenging"]
    amb_base, amb_pred = comparisons["ambiguity_challenging"]
    overall_delta = (
        overall_pred["mean_fscore"] - overall_base["mean_fscore"]
    )
    occ_delta = occ_pred["mean_fscore"] - occ_base["mean_fscore"]
    amb_delta = amb_pred["mean_fscore"] - amb_base["mean_fscore"]
    direct = r36.find_limit_row(
        limit_rows,
        "val",
        "overall",
        "predicted_depth_direct_backprojection",
    )
    passed = (
        overall_delta >= GATE_MARGIN_F1
        and occ_delta >= 0.0
        and amb_delta >= 0.0
    )
    return [
        {
            "run": RUN_NAME,
            "selected_method": selected_method,
            "selected_depth_scale_mode": "raw",
            "selected_tau_pred": selected["tau_pred"],
            "selected_alpha": selected["alpha"],
            "best_rgb_only_baseline_method": overall_base["method"],
            "validation_rgb_only_baseline_fscore": overall_base[
                "mean_fscore"
            ],
            "validation_predicted_correction_fscore": overall_pred[
                "mean_fscore"
            ],
            "delta_vs_rgb_only": overall_delta,
            "occlusion_delta_vs_rgb_only": occ_delta,
            "ambiguity_delta_vs_rgb_only": amb_delta,
            "validation_direct_predicted_depth_fscore": direct[
                "mean_fscore"
            ],
            "overall_pass": int(overall_delta >= GATE_MARGIN_F1),
            "occlusion_non_regression_pass": int(occ_delta >= 0.0),
            "ambiguity_non_regression_pass": int(amb_delta >= 0.0),
            "pass_all_limits": int(passed),
            "gate_margin_f1": GATE_MARGIN_F1,
            "final_project_claim_changed": int(passed),
        }
    ]


def evaluate_group_run38(record, selected, selected_by_mode):
    """Evaluate only scale modes generated by Run 38.

    Run 36 reports both raw and globally scale-aligned predicted depth. Run 38
    intentionally keeps only raw metric depth from the fine-tuned estimator to
    avoid validation/test scale fitting, so the inherited evaluator would ask
    for missing global_scale tensors.
    """
    group = record["group"]
    conf_mask, conf_percent, conf_threshold = r36.confidence_mask(
        record["conf"],
        int(group["num_views"]),
    )
    rows = [
        r36.score_points(
            record["points"],
            record,
            "mvdust3r_rgb_only_all_candidates",
            "rgb_only_baseline",
            {"selected_ratio": 1.0},
        ),
        r36.score_points(
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

    method_names = {
        "raw": "predicted_depth_correction_raw",
        "global_scale": "predicted_depth_correction_scale_aligned",
    }
    for mode in record["method_inputs"]:
        policy = selected_by_mode[mode]
        corrected, mask, residual = r36.corrected_points(
            record,
            mode,
            policy["tau_pred"],
            policy["alpha"],
        )
        valid = record["method_inputs"][mode]["valid"]
        rows.append(
            r36.score_points(
                corrected,
                record,
                method_names.get(mode, f"predicted_depth_correction_{mode}"),
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

    direct_mode = selected["depth_scale_mode"]
    rows.append(
        r36.score_points(
            record["method_inputs"][direct_mode]["direct_cloud"],
            record,
            "predicted_depth_direct_backprojection",
            "predicted_depth_direct",
            {
                "depth_scale_mode": direct_mode,
                "tau_pred": "",
                "alpha": "",
                "correction_ratio": "",
                "valid_predicted_depth_ratio": float(
                    record["method_inputs"][direct_mode]["valid"].mean()
                ),
            },
        )
    )
    for row in rows:
        print("Run 38 metric row:", row)
    return rows


def main():
    random.seed(SEED)
    np.random.seed(SEED)
    torch.manual_seed(SEED)
    started = time.time()
    out_dir = Path("/kaggle/working/outputs") / RUN_NAME
    out_dir.mkdir(parents=True, exist_ok=True)

    if CHECKPOINT_VARIANT not in {
        "controlled_best",
        "full_dataset_deployment",
    }:
        raise ValueError(
            "RUN38_DEPTH_CHECKPOINT must be controlled_best or "
            "full_dataset_deployment."
        )

    r36.RUN_NAME = RUN_NAME
    r36.MAX_CANDIDATES_PER_GROUP = MAX_CANDIDATES_PER_GROUP
    r36.SCALE_MODES = ["raw"]
    r36.GATE_MARGIN_F1 = GATE_MARGIN_F1
    r27.MAX_SCENES = sys.maxsize
    r27.validate_static_configuration()
    base.require_t4x2()

    run37_dir = locate_run37_output()
    checkpoint_dir = run37_dir / "checkpoints" / CHECKPOINT_VARIANT
    if not (checkpoint_dir / "model.safetensors").exists():
        raise FileNotFoundError(
            f"Run 37 checkpoint is incomplete: {checkpoint_dir}"
        )

    root = base.clone_repo()
    base.install_deps(root)
    Image, AutoImageProcessor, AutoModelForDepthEstimation = (
        install_depth_dependencies()
    )
    posed_root = base.find_posed_images_root()
    all_scene_dirs = discover_all_scene_dirs(posed_root)
    scene_dirs = (
        all_scene_dirs
        if MAX_SCENES is None
        else all_scene_dirs[:MAX_SCENES]
    )
    scene_lookup = {path.name: path for path in scene_dirs}

    run37_splits = read_scene_splits(run37_dir)
    missing_splits = sorted(set(scene_lookup) - set(run37_splits))
    if missing_splits:
        raise RuntimeError(
            f"Run 37 split is missing {len(missing_splits)} scenes."
        )
    splits = {scene: run37_splits[scene] for scene in scene_lookup}
    manifest = r27.build_group_manifest(scene_dirs, splits)
    train_group_candidates = [
        row for row in manifest if row["split"] == "train"
    ]
    train_groups = r27.balanced_group_subset(
        train_group_candidates,
        MAX_TRAIN_GROUPS,
    )
    eval_group_candidates = [
        row for row in manifest if row["split"] in {"val", "test"}
    ]
    eval_scene_candidates = list(
        dict.fromkeys(row["scene"] for row in eval_group_candidates)
    )
    selected_eval_scenes = (
        eval_scene_candidates
        if MAX_EVAL_SCENES is None
        else eval_scene_candidates[:MAX_EVAL_SCENES]
    )
    selected_eval_scene_set = set(selected_eval_scenes)
    scene_capped_eval_groups = [
        row
        for row in eval_group_candidates
        if row["scene"] in selected_eval_scene_set
    ]
    eval_groups = (
        scene_capped_eval_groups
        if MAX_EVAL_GROUPS is None
        else r27.balanced_group_subset(
            scene_capped_eval_groups,
            MAX_EVAL_GROUPS,
        )
    )
    evaluated_scene_ids = sorted(
        {row["scene"] for row in eval_groups}
    )
    _fit_groups, internal_val_groups, internal_val_scenes = (
        r27.split_internal_train_groups(train_groups)
    )
    r36.write_csv_union(out_dir / "eval_group_manifest.csv", eval_groups)
    print(
        "Run 38 group counts:",
        {
            "discovered_scenes": len(all_scene_dirs),
            "active_scenes": len(scene_dirs),
            "train_candidates": len(train_group_candidates),
            "internal_val": len(internal_val_groups),
            "eval_before_cap": len(eval_group_candidates),
            "eval_after_scene_cap": len(scene_capped_eval_groups),
            "eval_after_cap": len(eval_groups),
            "evaluated_scenes": len(evaluated_scene_ids),
        },
    )

    checkpoint = base.download_checkpoint(root)
    backbone = base.load_model(root, checkpoint)
    depth_predictor = FineTunedDepthPredictor(
        checkpoint_dir,
        posed_root,
        Image,
        AutoImageProcessor,
        AutoModelForDepthEstimation,
    )

    internal_records = []
    for index, group in enumerate(internal_val_groups):
        group_dir = (
            out_dir
            / "internal_val_groups"
            / group["group_key"]
        )
        internal_records.append(
            build_record(
                backbone,
                root,
                scene_lookup,
                group,
                group_dir,
                depth_predictor,
                retain_output=False,
            )
        )
        print(
            f"Run 38 policy groups completed: "
            f"{index + 1}/{len(internal_val_groups)}"
        )
    selected, selected_by_mode, policy_rows = r36.select_policies(
        internal_records
    )
    print("Run 38 selected policy:", selected)
    del internal_records
    torch.cuda.empty_cache()

    metric_rows = []
    retained_rows = []
    for index, group in enumerate(eval_groups, start=1):
        retain_output = len(retained_rows) < RETAIN_GROUP_OUTPUTS
        group_dir = out_dir / "eval_groups" / group["group_key"]
        record = build_record(
            backbone,
            root,
            scene_lookup,
            group,
            group_dir,
            depth_predictor,
            retain_output=retain_output,
        )
        metric_rows.extend(
            evaluate_group_run38(record, selected, selected_by_mode)
        )
        if retain_output:
            retained_rows.append(
                {
                    "run": RUN_NAME,
                    "split": group["split"],
                    "scene": group["scene"],
                    "group_key": group["group_key"],
                    "relative_output_dir": str(
                        group_dir.relative_to(out_dir)
                    ),
                }
            )
        del record
        torch.cuda.empty_cache()
        if (
            index % max(SAVE_EVERY_GROUPS, 1) == 0
            or index == len(eval_groups)
        ):
            r36.write_csv_union(
                out_dir / "metrics_partial.csv",
                metric_rows,
            )
            print(
                f"Run 38 completed groups: {index}/{len(eval_groups)}"
            )

    summary_rows = r36.summarize(metric_rows)
    limit_rows = r36.limit_summary(metric_rows)
    gate_rows = gate_decision(limit_rows, selected)
    comparison_rows = r36.comparison_table(
        limit_rows,
        [],
        selected,
    )
    r36.write_csv_union(out_dir / "metrics.csv", metric_rows)
    r36.write_csv_union(out_dir / "summary.csv", summary_rows)
    r36.write_csv_union(out_dir / "limit_summary.csv", limit_rows)
    r36.write_csv_union(out_dir / "policy_selection.csv", policy_rows)
    r36.write_csv_union(out_dir / "gate_decision.csv", gate_rows)
    r36.write_csv_union(
        out_dir / "comparison_table.csv",
        comparison_rows,
    )
    r36.write_csv_union(
        out_dir / "retained_qualitative_groups.csv",
        retained_rows,
    )
    partial_path = out_dir / "metrics_partial.csv"
    if partial_path.exists():
        partial_path.unlink()

    config = {
        "run": RUN_NAME,
        "purpose": (
            "Evaluate MV-DUSt3R+ candidate reconstruction corrected by the "
            "Run 37 fine-tuned metric depth estimator."
        ),
        "source_run37_dir": str(run37_dir),
        "depth_checkpoint_variant": CHECKPOINT_VARIANT,
        "depth_checkpoint_path": str(checkpoint_dir),
        "checkpoint_has_held_out_evaluation": (
            CHECKPOINT_VARIANT == "controlled_best"
        ),
        "seed": SEED,
        "scene_limit_raw": MAX_SCENES_RAW,
        "scene_limit_resolved": MAX_SCENES,
        "max_eval_scenes_raw": MAX_EVAL_SCENES_RAW,
        "max_eval_scenes_resolved": MAX_EVAL_SCENES,
        "max_eval_groups_raw": MAX_EVAL_GROUPS_RAW,
        "max_eval_groups_resolved": MAX_EVAL_GROUPS,
        "num_discovered_scenes": len(all_scene_dirs),
        "num_active_scenes": len(scene_dirs),
        "num_total_groups": len(manifest),
        "num_train_groups_before_subset": len(
            train_group_candidates
        ),
        "num_policy_train_groups": len(train_groups),
        "num_internal_val_groups": len(internal_val_groups),
        "num_total_eval_groups_before_cap": len(
            eval_group_candidates
        ),
        "num_eval_groups_after_scene_cap": len(
            scene_capped_eval_groups
        ),
        "num_eval_groups_after_cap": len(eval_groups),
        "evaluated_scene_count": len(evaluated_scene_ids),
        "evaluated_scene_ids": evaluated_scene_ids,
        "selected_policy": selected,
        "selected_policy_by_scale_mode": selected_by_mode,
        "max_candidates_per_group": MAX_CANDIDATES_PER_GROUP,
        "retained_group_outputs": len(retained_rows),
        "uses_true_source_depth_for_inference": False,
        "uses_true_source_depth_for_correction": False,
        "uses_true_source_depth_for_evaluation": True,
        "uses_finetuned_predicted_depth_for_inference": True,
        "uses_known_pose": True,
        "uses_known_intrinsics": True,
        "input_contract": (
            "Sparse RGB images + Run 37 estimated metric depth + known "
            "camera poses/intrinsics."
        ),
        "evaluation_contract": (
            "Scene discovery and policy selection remain uncapped. The "
            "default evaluation uses all sparse-view groups from all "
            "validation/test scenes. Set RUN38_MAX_EVAL_SCENES to a positive "
            "integer only for a smaller pilot. True depth is used only by the "
            "project proxy evaluator and hard-subset diagnostics."
        ),
        "claim_contract": (
            "This is not an official full ScanNet or ScanNet++ benchmark "
            "and is not a fully pose-free method."
        ),
        "gate_margin_f1": GATE_MARGIN_F1,
        "runtime_seconds": time.time() - started,
    }
    (out_dir / "run_config.json").write_text(
        json.dumps(config, indent=2),
        encoding="utf-8",
    )
    print("Run 38 summary:")
    for row in summary_rows:
        print(row)
    print("Run 38 gate decision:", gate_rows[0])
    print("Run 38 config:", config)
    print("Run 38 output dir:", out_dir)


if __name__ == "__main__":
    main()
