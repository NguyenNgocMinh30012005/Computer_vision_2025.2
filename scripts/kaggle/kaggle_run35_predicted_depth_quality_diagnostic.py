import json
import math
import os
import random
import subprocess
import sys
import time
import urllib.request
from pathlib import Path

import numpy as np
import torch
import torch.nn.functional as F
from PIL import Image

try:
    from kaggle_predicted_depth_utils import (
        depth_error_metrics,
        read_metric_depth,
        summarize_depth_rows,
    )
except ModuleNotFoundError:
    def read_metric_depth(path):
        depth = np.asarray(Image.open(path))
        if depth.ndim == 3:
            depth = depth[..., 0]
        depth = depth.astype(np.float32)
        if depth.size and float(np.nanmax(depth)) > 100.0:
            depth /= 1000.0
        depth[~np.isfinite(depth)] = 0.0
        return depth

    def depth_error_metrics(predicted_depth, source_depth, min_depth_m=0.10):
        predicted = np.asarray(predicted_depth, dtype=np.float32)
        source = np.asarray(source_depth, dtype=np.float32)
        valid = (
            np.isfinite(source)
            & np.isfinite(predicted)
            & (source > float(min_depth_m))
            & (predicted > 1e-6)
        )
        if int(valid.sum()) < 16:
            raise ValueError(
                f"Insufficient valid depth pixels: {int(valid.sum())}"
            )
        pred = predicted[valid].astype(np.float64)
        src = source[valid].astype(np.float64)
        error = pred - src
        ratio = np.maximum(
            pred / np.maximum(src, 1e-8),
            src / np.maximum(pred, 1e-8),
        )
        scales = src / np.maximum(pred, 1e-8)
        median_scale = float(np.median(scales))
        least_squares_scale = float(
            np.sum(pred * src) / max(np.sum(pred * pred), 1e-12)
        )
        aligned_error = pred * median_scale - src
        return {
            "num_pixels": int(source.size),
            "num_valid_pixels": int(valid.sum()),
            "valid_pixel_ratio": float(valid.mean()),
            "abs_rel": float(
                np.mean(np.abs(error) / np.maximum(src, 1e-8))
            ),
            "rmse": float(np.sqrt(np.mean(error * error))),
            "mae": float(np.mean(np.abs(error))),
            "delta1": float(np.mean(ratio < 1.25)),
            "delta2": float(np.mean(ratio < 1.25**2)),
            "delta3": float(np.mean(ratio < 1.25**3)),
            "scale_aligned_rmse": float(
                np.sqrt(np.mean(aligned_error * aligned_error))
            ),
            "scale_aligned_mae": float(
                np.mean(np.abs(aligned_error))
            ),
            "median_scale_ratio": median_scale,
            "least_squares_scale_ratio": least_squares_scale,
            "source_depth_mean_m": float(np.mean(src)),
            "predicted_depth_mean_m": float(np.mean(pred)),
        }

    def summarize_depth_rows(rows, split):
        selected = (
            rows
            if split == "all"
            else [row for row in rows if row["split"] == split]
        )
        if not selected:
            return None
        metric_names = [
            "valid_pixel_ratio",
            "abs_rel",
            "rmse",
            "mae",
            "delta1",
            "delta2",
            "delta3",
            "scale_aligned_rmse",
            "scale_aligned_mae",
            "median_scale_ratio",
            "least_squares_scale_ratio",
        ]
        output = {
            "split": split,
            "num_frames": len(selected),
            "num_scenes": len({row["scene"] for row in selected}),
        }
        for name in metric_names:
            values = [
                float(row[name])
                for row in selected
                if math.isfinite(float(row[name]))
            ]
            output[f"mean_{name}"] = float(np.mean(values))
            output[f"median_{name}"] = float(np.median(values))
        return output


RAW_RUN27 = "https://raw.githubusercontent.com/NguyenNgocMinh30012005/Computer_vision_2025.2/main/scripts/kaggle/kaggle_run27_joint_candidate_acceptance.py"


def ensure_helper_module(module_name, raw_url):
    try:
        return __import__(module_name)
    except ModuleNotFoundError:
        helper_path = Path(f"{module_name}.py")
        print(f"Downloading helper module {module_name} from GitHub raw")
        urllib.request.urlretrieve(raw_url, helper_path)
        sys.path.insert(0, str(Path.cwd()))
        return __import__(module_name)


r27 = ensure_helper_module("kaggle_run27_joint_candidate_acceptance", RAW_RUN27)
base = r27.base

RUN_NAME = "run_35_predicted_depth_quality_diagnostic"
SEED = 3535
MAX_SCENES = int(os.environ.get("RUN35_MAX_SCENES", "30"))
MAX_TRAIN_GROUPS = int(os.environ.get("RUN35_MAX_TRAIN_GROUPS", "48"))
MAX_EVAL_GROUPS = int(os.environ.get("RUN35_MAX_EVAL_GROUPS", "36"))
DEPTH_MODEL_NAME = os.environ.get(
    "RUN35_DEPTH_MODEL",
    "depth-anything/Depth-Anything-V2-Metric-Indoor-Small-hf",
)
DEPTH_MODEL_REVISION = os.environ.get("RUN35_DEPTH_MODEL_REVISION", "main")
MIN_DEPTH_M = float(os.environ.get("RUN35_MIN_DEPTH_M", "0.10"))


def ensure_depth_dependencies():
    try:
        from huggingface_hub import model_info
        from transformers import AutoImageProcessor, AutoModelForDepthEstimation

        return AutoImageProcessor, AutoModelForDepthEstimation, model_info
    except Exception as exc:
        print("Installing Run 35 depth dependencies:", repr(exc))
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
        from huggingface_hub import model_info
        from transformers import AutoImageProcessor, AutoModelForDepthEstimation

        return AutoImageProcessor, AutoModelForDepthEstimation, model_info


class MetricDepthPredictor:
    def __init__(self, model_name, revision):
        AutoImageProcessor, AutoModelForDepthEstimation, model_info = (
            ensure_depth_dependencies()
        )
        self.model_name = model_name
        self.requested_revision = revision
        try:
            self.resolved_revision = model_info(
                model_name,
                revision=revision,
            ).sha
        except Exception as exc:
            print("Unable to resolve model revision SHA:", repr(exc))
            self.resolved_revision = revision
        self.device = "cuda" if torch.cuda.is_available() else "cpu"
        self.processor = AutoImageProcessor.from_pretrained(
            model_name,
            revision=revision,
        )
        model_kwargs = {
            "revision": revision,
            "low_cpu_mem_usage": True,
        }
        if self.device == "cuda":
            model_kwargs["torch_dtype"] = torch.float16
        try:
            self.model = AutoModelForDepthEstimation.from_pretrained(
                model_name,
                **model_kwargs,
            )
        except Exception:
            model_kwargs.pop("torch_dtype", None)
            self.model = AutoModelForDepthEstimation.from_pretrained(
                model_name,
                **model_kwargs,
            )
        self.model.to(self.device).eval()

    def predict(self, image, output_shape):
        inputs = self.processor(images=image, return_tensors="pt")
        inputs = {
            key: value.to(self.device)
            for key, value in inputs.items()
        }
        with torch.inference_mode():
            outputs = self.model(**inputs)
            predicted = outputs.predicted_depth
            if predicted.ndim == 3:
                predicted = predicted.unsqueeze(1)
            predicted = F.interpolate(
                predicted,
                size=tuple(output_shape),
                mode="bicubic",
                align_corners=False,
            )
        depth = predicted.squeeze().detach().float().cpu().numpy()
        depth = np.asarray(depth, dtype=np.float32)
        depth[~np.isfinite(depth)] = 0.0
        depth[depth <= 0.0] = 1e-6
        return depth


def build_selected_groups(scene_dirs, splits):
    manifest = r27.build_group_manifest(scene_dirs, splits)
    train_groups = r27.balanced_group_subset(
        [row for row in manifest if row["split"] == "train"],
        MAX_TRAIN_GROUPS,
    )
    eval_groups = r27.balanced_group_subset(
        [row for row in manifest if row["split"] in {"val", "test"}],
        MAX_EVAL_GROUPS,
    )
    fit_groups, internal_val_groups, internal_val_scenes = (
        r27.split_internal_train_groups(train_groups)
    )
    stage_by_scene = {
        row["scene"]: "fit"
        for row in fit_groups
    }
    stage_by_scene.update(
        {
            row["scene"]: "internal_val"
            for row in internal_val_groups
        }
    )
    return train_groups, eval_groups, internal_val_scenes, stage_by_scene


def collect_unique_frames(scene_lookup, groups, stage_by_scene):
    frame_records = {}
    group_rows = []
    for group in groups:
        view_files = r27.choose_group_views(scene_lookup, group)
        group_rows.append(
            {
                **group,
                "selected_images": "|".join(path.name for path in view_files),
            }
        )
        for path in view_files:
            key = f"{group['scene']}/{path.name}"
            record = frame_records.setdefault(
                key,
                {
                    "split": group["split"],
                    "stage": (
                        stage_by_scene.get(group["scene"], "external_eval")
                        if group["split"] == "train"
                        else "external_eval"
                    ),
                    "scene": group["scene"],
                    "frame": path.name,
                    "image_path": path,
                    "group_keys": [],
                },
            )
            record["group_keys"].append(group["group_key"])
    return list(frame_records.values()), group_rows


def cache_prediction(cache_root, scene, frame, depth):
    relative = Path("predicted_depth_cache") / scene / f"{Path(frame).stem}.npz"
    path = cache_root / relative
    path.parent.mkdir(parents=True, exist_ok=True)
    np.savez_compressed(path, depth=depth.astype(np.float16))
    return relative.as_posix(), path.stat().st_size


def main():
    random.seed(SEED)
    np.random.seed(SEED)
    torch.manual_seed(SEED)
    started = time.time()
    out_dir = Path("/kaggle/working/outputs") / RUN_NAME
    out_dir.mkdir(parents=True, exist_ok=True)

    base.require_t4x2()
    posed_root = base.find_posed_images_root()
    scene_dirs = r27.discover_scene_dirs(posed_root)[:MAX_SCENES]
    scene_lookup = {path.name: path for path in scene_dirs}
    splits = r27.scene_splits(scene_dirs)
    train_groups, eval_groups, internal_val_scenes, stage_by_scene = (
        build_selected_groups(scene_dirs, splits)
    )
    frame_records, selected_group_rows = collect_unique_frames(
        scene_lookup,
        train_groups + eval_groups,
        stage_by_scene,
    )
    print(
        "Run 35 selection:",
        {
            "num_scenes": len(scene_dirs),
            "num_train_groups": len(train_groups),
            "num_eval_groups": len(eval_groups),
            "num_unique_frames": len(frame_records),
            "internal_val_scenes": internal_val_scenes,
        },
    )

    predictor = MetricDepthPredictor(
        DEPTH_MODEL_NAME,
        DEPTH_MODEL_REVISION,
    )
    metric_rows = []
    cache_rows = []
    for index, frame_record in enumerate(frame_records, start=1):
        image_path = frame_record["image_path"]
        source_depth = read_metric_depth(
            str(image_path).replace(".jpg", ".png")
        )
        image = Image.open(image_path).convert("RGB")
        predicted_depth = predictor.predict(image, source_depth.shape)
        metrics = depth_error_metrics(
            predicted_depth,
            source_depth,
            min_depth_m=MIN_DEPTH_M,
        )
        cache_relpath, cache_bytes = cache_prediction(
            out_dir,
            frame_record["scene"],
            frame_record["frame"],
            predicted_depth,
        )
        common = {
            "run": RUN_NAME,
            "split": frame_record["split"],
            "stage": frame_record["stage"],
            "scene": frame_record["scene"],
            "frame": frame_record["frame"],
            "num_reusing_groups": len(frame_record["group_keys"]),
            "group_keys": "|".join(sorted(frame_record["group_keys"])),
            "depth_model_name": DEPTH_MODEL_NAME,
            "depth_checkpoint": predictor.resolved_revision,
        }
        metric_row = {**common, **metrics}
        cache_row = {
            **common,
            "cache_relpath": cache_relpath,
            "height": int(predicted_depth.shape[0]),
            "width": int(predicted_depth.shape[1]),
            "cache_bytes": int(cache_bytes),
        }
        metric_rows.append(metric_row)
        cache_rows.append(cache_row)
        if index == 1 or index % 25 == 0 or index == len(frame_records):
            print(
                "Run 35 progress:",
                {
                    "frames": index,
                    "total": len(frame_records),
                    "latest_abs_rel": metrics["abs_rel"],
                    "latest_scale_aligned_rmse": metrics[
                        "scale_aligned_rmse"
                    ],
                },
            )

    summary_rows = [
        row
        for split in ["fit", "internal_val", "train", "val", "test", "all"]
        if (
            row := (
                summarize_depth_rows(
                    [
                        {
                            **metric,
                            "split": (
                                metric["stage"]
                                if split in {"fit", "internal_val"}
                                else metric["split"]
                            ),
                        }
                        for metric in metric_rows
                    ],
                    split,
                )
            )
        )
        is not None
    ]
    fit_rows = [row for row in metric_rows if row["stage"] == "fit"]
    if not fit_rows:
        raise RuntimeError("Run 35 has no fit frames for scale calibration.")
    global_scale = float(
        np.median([row["median_scale_ratio"] for row in fit_rows])
    )
    least_squares_global_scale = float(
        np.median(
            [row["least_squares_scale_ratio"] for row in fit_rows]
        )
    )
    scale_rows = [
        {
            "run": RUN_NAME,
            "calibration_source": "train_fit_scenes_only",
            "num_frames": len(fit_rows),
            "global_median_scale_ratio": global_scale,
            "global_median_least_squares_scale_ratio": (
                least_squares_global_scale
            ),
            "internal_val_scenes": "|".join(internal_val_scenes),
        }
    ]

    r27.write_csv_union(out_dir / "depth_metrics.csv", metric_rows)
    r27.write_csv_union(out_dir / "depth_summary.csv", summary_rows)
    r27.write_csv_union(
        out_dir / "predicted_depth_cache_manifest.csv",
        cache_rows,
    )
    r27.write_csv_union(
        out_dir / "selected_group_manifest.csv",
        selected_group_rows,
    )
    r27.write_csv_union(out_dir / "scale_calibration.csv", scale_rows)
    config = {
        "run": RUN_NAME,
        "purpose": (
            "Measure monocular metric-depth quality before predicted-depth "
            "candidate correction."
        ),
        "depth_model_name": DEPTH_MODEL_NAME,
        "depth_checkpoint": predictor.resolved_revision,
        "requested_depth_revision": DEPTH_MODEL_REVISION,
        "model_family": "Depth Anything V2 metric indoor small",
        "num_scenes": len(scene_dirs),
        "scene_splits": splits,
        "num_train_groups": len(train_groups),
        "num_eval_groups": len(eval_groups),
        "num_unique_frames": len(frame_records),
        "internal_val_scenes": internal_val_scenes,
        "min_valid_source_depth_m": MIN_DEPTH_M,
        "global_scale_fit_median": global_scale,
        "global_scale_fit_least_squares_median": (
            least_squares_global_scale
        ),
        "uses_true_source_depth_for_inference": False,
        "uses_true_source_depth_for_evaluation": True,
        "uses_predicted_depth_for_inference": True,
        "input_contract": "RGB images only for depth prediction.",
        "evaluation_contract": (
            "True source RGB-D depth is used only to score predicted depth "
            "and calibrate a train-fit global scale."
        ),
        "runtime_seconds": time.time() - started,
    }
    (out_dir / "run_config.json").write_text(
        json.dumps(config, indent=2),
        encoding="utf-8",
    )
    print("Run 35 scale calibration:", scale_rows[0])
    print("Run 35 summary:")
    for row in summary_rows:
        print(row)
    print("Run 35 config:", config)
    print("Run 35 output dir:", out_dir)


if __name__ == "__main__":
    main()
