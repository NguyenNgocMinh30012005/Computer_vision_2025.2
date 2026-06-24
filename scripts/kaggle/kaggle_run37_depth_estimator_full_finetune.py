import json
import os
import random
import subprocess
import sys
import time
from pathlib import Path

import numpy as np
from PIL import Image, ImageDraw, ImageEnhance

try:
    from kaggle_run37_depth_finetune_utils import (
        assign_scene_splits,
        depth_error_metrics,
        discover_complete_rgbd_pose_frames,
        read_metric_depth,
        resize_depth,
        split_summary,
        summarize_depth_rows,
        write_csv,
    )
except ModuleNotFoundError:
    from scripts.kaggle.kaggle_run37_depth_finetune_utils import (
        assign_scene_splits,
        depth_error_metrics,
        discover_complete_rgbd_pose_frames,
        read_metric_depth,
        resize_depth,
        split_summary,
        summarize_depth_rows,
        write_csv,
    )


RUN_NAME = "run_37_depth_estimator_full_finetune"
SEED = int(os.environ.get("RUN37_SEED", "3737"))
MODEL_NAME = os.environ.get(
    "RUN37_DEPTH_MODEL",
    "depth-anything/Depth-Anything-V2-Metric-Indoor-Small-hf",
)
MODEL_REVISION = os.environ.get("RUN37_DEPTH_MODEL_REVISION", "main")
IMAGE_SIZE = int(os.environ.get("RUN37_IMAGE_SIZE", "392"))
BATCH_SIZE = int(os.environ.get("RUN37_BATCH_SIZE", "2"))
GRAD_ACCUM_STEPS = int(os.environ.get("RUN37_GRAD_ACCUM_STEPS", "8"))
CONTROLLED_EPOCHS = int(os.environ.get("RUN37_CONTROLLED_EPOCHS", "1"))
DEPLOYMENT_EPOCHS = int(os.environ.get("RUN37_DEPLOYMENT_EPOCHS", "1"))
MAX_TRAIN_STEPS = int(os.environ.get("RUN37_MAX_TRAIN_STEPS", "0"))
MAX_DEPLOYMENT_STEPS = int(os.environ.get("RUN37_MAX_DEPLOYMENT_STEPS", "0"))
MAX_EVAL_FRAMES_PER_SPLIT = int(
    os.environ.get("RUN37_MAX_EVAL_FRAMES_PER_SPLIT", "1200")
)
NUM_WORKERS = int(os.environ.get("RUN37_NUM_WORKERS", "2"))
MIN_DEPTH_M = float(os.environ.get("RUN37_MIN_DEPTH_M", "0.10"))
MAX_DEPTH_M = float(os.environ.get("RUN37_MAX_DEPTH_M", "10.0"))
BACKBONE_LR = float(os.environ.get("RUN37_BACKBONE_LR", "2e-6"))
HEAD_LR = float(os.environ.get("RUN37_HEAD_LR", "2e-5"))
WEIGHT_DECAY = float(os.environ.get("RUN37_WEIGHT_DECAY", "0.01"))
GATE_ABSREL_MARGIN = float(os.environ.get("RUN37_GATE_ABSREL_MARGIN", "0.02"))
GATE_DELTA1_MARGIN = float(os.environ.get("RUN37_GATE_DELTA1_MARGIN", "0.02"))
TRAIN_FULL_DEPLOYMENT = os.environ.get("RUN37_TRAIN_FULL_DEPLOYMENT", "1") == "1"
TORCH_REEXEC_FLAG = "RUN37_TORCH_REEXECED_AFTER_COMPAT_INSTALL"


def run(cmd):
    print("+", " ".join(map(str, cmd)))
    subprocess.run(cmd, check=True)


def ensure_training_dependencies():
    try:
        import torch
        from huggingface_hub import model_info
        from transformers import AutoImageProcessor, AutoModelForDepthEstimation

        return torch, AutoImageProcessor, AutoModelForDepthEstimation, model_info
    except Exception as exc:
        print("Installing Run 37 depth fine-tuning dependencies:", repr(exc))
        run(
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
        import torch
        from huggingface_hub import model_info
        from transformers import AutoImageProcessor, AutoModelForDepthEstimation

        return torch, AutoImageProcessor, AutoModelForDepthEstimation, model_info


def verify_cuda_usable(torch):
    try:
        x = torch.ones(1, device="cuda")
        y = (x + 1).detach().cpu().item()
        return y == 2.0, None
    except Exception as exc:
        return False, repr(exc)


def install_p100_compatible_torch_and_reexec(torch, names):
    if os.environ.get(TORCH_REEXEC_FLAG) == "1":
        raise RuntimeError(
            "CUDA is still unusable after one Torch compatibility reinstall. "
            f"GPU names: {names}; torch={torch.__version__}"
        )
    if not any("P100" in name for name in names):
        raise RuntimeError(
            "CUDA is unusable and this script only auto-reinstalls Torch for P100. "
            f"GPU names: {names}; torch={torch.__version__}"
        )
    print("Detected P100 with an incompatible Torch/CUDA build.")
    print("Installing Torch 2.5.1 cu121 and restarting Run 37 once.")
    run(
        [
            sys.executable,
            "-m",
            "pip",
            "install",
            "-q",
            "--no-cache-dir",
            "--index-url",
            "https://download.pytorch.org/whl/cu121",
            "torch==2.5.1",
            "torchvision==0.20.1",
            "torchaudio==2.5.1",
        ]
    )
    os.environ[TORCH_REEXEC_FLAG] = "1"
    os.execv(sys.executable, [sys.executable] + sys.argv)


def require_cuda(torch):
    print("CUDA available:", torch.cuda.is_available())
    print("CUDA device count:", torch.cuda.device_count())
    names = []
    if torch.cuda.is_available():
        for index in range(torch.cuda.device_count()):
            name = torch.cuda.get_device_name(index)
            names.append(name)
            print(f"GPU {index}:", name)
    print("Torch:", torch.__version__)
    if not torch.cuda.is_available() or torch.cuda.device_count() < 1:
        raise RuntimeError("Run 37 requires at least one CUDA GPU.")
    ok, err = verify_cuda_usable(torch)
    if not ok:
        print("CUDA smoke test failed:", err)
        install_p100_compatible_torch_and_reexec(torch, names)
    if not (torch.cuda.device_count() >= 2 and all("T4" in name for name in names)):
        print(f"Warning: expected T4 x2, got {names}. Continuing.")


def find_posed_images_root():
    candidates = [
        Path("/kaggle/input/scannet-data/scannet/posed_images"),
        Path(
            "/kaggle/input/datasets/tiantiansyrinx1102/scannet-data/scannet/posed_images"
        ),
    ]
    for path in candidates:
        if path.exists():
            return path
    for path in Path("/kaggle/input").rglob("posed_images"):
        if path.is_dir() and any(child.is_dir() for child in path.iterdir()):
            return path
    raise FileNotFoundError("Could not locate scannet/posed_images under /kaggle/input")


def seed_everything(torch):
    random.seed(SEED)
    np.random.seed(SEED)
    torch.manual_seed(SEED)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(SEED)


def select_eval_rows(rows, split):
    selected = [row for row in rows if row["split"] == split]
    if MAX_EVAL_FRAMES_PER_SPLIT <= 0 or len(selected) <= MAX_EVAL_FRAMES_PER_SPLIT:
        return selected
    rng = random.Random(SEED + len(split))
    return sorted(
        rng.sample(selected, MAX_EVAL_FRAMES_PER_SPLIT),
        key=lambda row: (row["scene"], row["frame"]),
    )


class DepthFrameDataset:
    def __init__(self, rows, train=False):
        self.rows = list(rows)
        self.train = bool(train)

    def __len__(self):
        return len(self.rows)

    def __getitem__(self, index):
        row = self.rows[index]
        image = Image.open(row["rgb_path"]).convert("RGB")
        depth = read_metric_depth(row["depth_path"])
        if self.train and random.random() < 0.5:
            image = image.transpose(Image.Transpose.FLIP_LEFT_RIGHT)
            depth = np.fliplr(depth).copy()
        if self.train:
            if random.random() < 0.35:
                image = ImageEnhance.Brightness(image).enhance(
                    random.uniform(0.85, 1.15)
                )
            if random.random() < 0.35:
                image = ImageEnhance.Contrast(image).enhance(
                    random.uniform(0.85, 1.15)
                )
        return {"row": row, "image": image, "depth": depth}


def make_collate_fn(processor):
    def collate(samples):
        images = [
            sample["image"].resize(
                (IMAGE_SIZE, IMAGE_SIZE),
                resample=Image.Resampling.BICUBIC,
            )
            for sample in samples
        ]
        inputs = processor(images=images, return_tensors="pt")
        _, _, height, width = inputs["pixel_values"].shape
        depths = [
            resize_depth(sample["depth"], width=width, height=height)
            for sample in samples
        ]
        depth_tensor = np.stack(depths, axis=0)[:, None, :, :]
        return {
            "pixel_values": inputs["pixel_values"],
            "depth": depth_tensor.astype(np.float32),
            "rows": [sample["row"] for sample in samples],
        }

    return collate


def depth_finetune_loss(torch, F, predicted, target):
    if predicted.ndim == 3:
        predicted = predicted.unsqueeze(1)
    predicted = F.interpolate(
        predicted,
        size=target.shape[-2:],
        mode="bicubic",
        align_corners=False,
    )
    predicted = torch.clamp(predicted.float(), min=1e-4, max=MAX_DEPTH_M)
    target = target.float()
    valid = torch.isfinite(target) & (target > MIN_DEPTH_M) & (target < MAX_DEPTH_M)
    if int(valid.sum().item()) < 16:
        return predicted.sum() * 0.0, {
            "loss_log_l1": 0.0,
            "loss_abs_rel": 0.0,
            "loss_grad": 0.0,
        }

    log_pred = torch.log(predicted)
    log_target = torch.log(torch.clamp(target, min=1e-4))
    log_error = log_pred - log_target
    log_l1 = torch.abs(log_error[valid]).mean()
    abs_rel = (
        torch.abs(predicted[valid] - target[valid])
        / torch.clamp(target[valid], min=1e-4)
    ).mean()

    valid_x = valid[:, :, :, 1:] & valid[:, :, :, :-1]
    valid_y = valid[:, :, 1:, :] & valid[:, :, :-1, :]
    grad_terms = []
    if bool(valid_x.any()):
        grad_terms.append(torch.abs((log_error[:, :, :, 1:] - log_error[:, :, :, :-1])[valid_x]).mean())
    if bool(valid_y.any()):
        grad_terms.append(torch.abs((log_error[:, :, 1:, :] - log_error[:, :, :-1, :])[valid_y]).mean())
    grad_loss = torch.stack(grad_terms).mean() if grad_terms else log_l1 * 0.0
    loss = 0.50 * log_l1 + 0.30 * abs_rel + 0.20 * grad_loss
    return loss, {
        "loss_log_l1": float(log_l1.detach().cpu()),
        "loss_abs_rel": float(abs_rel.detach().cpu()),
        "loss_grad": float(grad_loss.detach().cpu()),
    }


def optimizer_for(torch, model):
    head_keywords = ("head", "neck", "decoder", "decode", "output", "fusion")
    backbone_params = []
    head_params = []
    for name, param in model.named_parameters():
        if not param.requires_grad:
            continue
        if any(keyword in name.lower() for keyword in head_keywords):
            head_params.append(param)
        else:
            backbone_params.append(param)
    return torch.optim.AdamW(
        [
            {"params": backbone_params, "lr": BACKBONE_LR},
            {"params": head_params, "lr": HEAD_LR},
        ],
        weight_decay=WEIGHT_DECAY,
    )


def make_loader(torch, rows, processor, train):
    dataset = DepthFrameDataset(rows, train=train)
    return torch.utils.data.DataLoader(
        dataset,
        batch_size=BATCH_SIZE,
        shuffle=train,
        num_workers=NUM_WORKERS,
        pin_memory=True,
        collate_fn=make_collate_fn(processor),
    )


def train_epoch(torch, F, model, processor, rows, optimizer, scaler, device, epoch, max_steps):
    loader = make_loader(torch, rows, processor, train=True)
    model.train()
    optimizer.zero_grad(set_to_none=True)
    rows_seen = 0
    total_loss = 0.0
    steps = 0
    start = time.time()
    for step, batch in enumerate(loader, start=1):
        pixel_values = batch["pixel_values"].to(device, non_blocking=True)
        target = torch.as_tensor(batch["depth"], device=device)
        with torch.cuda.amp.autocast(enabled=device.type == "cuda"):
            output = model(pixel_values=pixel_values)
            loss, loss_parts = depth_finetune_loss(
                torch,
                F,
                output.predicted_depth,
                target,
            )
            loss = loss / max(GRAD_ACCUM_STEPS, 1)
        scaler.scale(loss).backward()
        if step % GRAD_ACCUM_STEPS == 0:
            scaler.step(optimizer)
            scaler.update()
            optimizer.zero_grad(set_to_none=True)
        rows_seen += len(batch["rows"])
        total_loss += float(loss.detach().cpu()) * max(GRAD_ACCUM_STEPS, 1)
        steps += 1
        if step % 50 == 0:
            print(
                {
                    "epoch": epoch,
                    "step": step,
                    "rows_seen": rows_seen,
                    "loss": total_loss / max(steps, 1),
                    **loss_parts,
                }
            )
        if max_steps > 0 and step >= max_steps:
            break
    if steps % GRAD_ACCUM_STEPS != 0:
        scaler.step(optimizer)
        scaler.update()
        optimizer.zero_grad(set_to_none=True)
    return {
        "epoch": epoch,
        "train_rows_seen": rows_seen,
        "train_steps": steps,
        "train_loss": total_loss / max(steps, 1),
        "runtime_seconds": time.time() - start,
    }


def evaluate(torch, F, model, processor, rows, device, split, tag):
    loader = make_loader(torch, rows, processor, train=False)
    model.eval()
    metric_rows = []
    with torch.inference_mode():
        for batch in loader:
            pixel_values = batch["pixel_values"].to(device, non_blocking=True)
            target = torch.as_tensor(batch["depth"], device=device)
            output = model(pixel_values=pixel_values)
            predicted = output.predicted_depth
            if predicted.ndim == 3:
                predicted = predicted.unsqueeze(1)
            predicted = F.interpolate(
                predicted,
                size=target.shape[-2:],
                mode="bicubic",
                align_corners=False,
            )
            predicted = predicted.squeeze(1).float().cpu().numpy()
            target_np = target.squeeze(1).float().cpu().numpy()
            for index, row in enumerate(batch["rows"]):
                try:
                    metrics = depth_error_metrics(
                        predicted[index],
                        target_np[index],
                        min_depth_m=MIN_DEPTH_M,
                    )
                except ValueError as exc:
                    print("Skipping eval frame:", row["scene"], row["frame"], repr(exc))
                    continue
                metric_rows.append(
                    {
                        "run": RUN_NAME,
                        "tag": tag,
                        "split": split,
                        "scene": row["scene"],
                        "frame": row["frame"],
                        **metrics,
                    }
                )
    return metric_rows


def save_sample_pngs(torch, F, model, processor, rows, device, out_dir, max_images=8):
    out_dir.mkdir(parents=True, exist_ok=True)
    selected = rows[:max_images]
    if not selected:
        return []
    model.eval()
    manifest = []
    for row in selected:
        image = Image.open(row["rgb_path"]).convert("RGB")
        depth = read_metric_depth(row["depth_path"])
        input_image = image.resize((IMAGE_SIZE, IMAGE_SIZE), Image.Resampling.BICUBIC)
        inputs = processor(images=[input_image], return_tensors="pt")
        target = resize_depth(
            depth,
            width=inputs["pixel_values"].shape[-1],
            height=inputs["pixel_values"].shape[-2],
        )
        with torch.inference_mode():
            pred = model(
                pixel_values=inputs["pixel_values"].to(device),
            ).predicted_depth
            if pred.ndim == 3:
                pred = pred.unsqueeze(1)
            pred = F.interpolate(
                pred,
                size=target.shape,
                mode="bicubic",
                align_corners=False,
            )[0, 0].float().cpu().numpy()
        canvas = make_depth_comparison_image(input_image, target, pred)
        name = f"{row['scene']}_{row['frame']}.png"
        path = out_dir / name
        canvas.save(path)
        manifest.append(
            {
                "scene": row["scene"],
                "frame": row["frame"],
                "path": str(path),
            }
        )
    return manifest


def colorize_depth(depth):
    depth = np.asarray(depth, dtype=np.float32)
    valid = np.isfinite(depth) & (depth > MIN_DEPTH_M)
    if not valid.any():
        scaled = np.zeros_like(depth, dtype=np.uint8)
    else:
        lo, hi = np.percentile(depth[valid], [2, 98])
        scaled = np.clip((depth - lo) / max(hi - lo, 1e-6), 0.0, 1.0)
        scaled = (scaled * 255.0).astype(np.uint8)
    return Image.fromarray(scaled, mode="L").convert("RGB")


def make_depth_comparison_image(image, target, predicted):
    image = image.resize((IMAGE_SIZE, IMAGE_SIZE), Image.Resampling.BICUBIC)
    target_img = colorize_depth(target).resize(image.size)
    pred_img = colorize_depth(predicted).resize(image.size)
    width, height = image.size
    canvas = Image.new("RGB", (width * 3, height + 24), "white")
    canvas.paste(image, (0, 24))
    canvas.paste(target_img, (width, 24))
    canvas.paste(pred_img, (width * 2, 24))
    draw = ImageDraw.Draw(canvas)
    draw.text((6, 6), "RGB", fill=(0, 0, 0))
    draw.text((width + 6, 6), "source depth", fill=(0, 0, 0))
    draw.text((width * 2 + 6, 6), "prediction", fill=(0, 0, 0))
    return canvas


def load_model(torch, AutoImageProcessor, AutoModelForDepthEstimation, model_info):
    try:
        resolved_revision = model_info(MODEL_NAME, revision=MODEL_REVISION).sha
    except Exception as exc:
        print("Unable to resolve model revision SHA:", repr(exc))
        resolved_revision = MODEL_REVISION
    processor = AutoImageProcessor.from_pretrained(
        MODEL_NAME,
        revision=MODEL_REVISION,
    )
    model = AutoModelForDepthEstimation.from_pretrained(
        MODEL_NAME,
        revision=MODEL_REVISION,
    )
    return processor, model, resolved_revision


def write_json(path, data):
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(data, indent=2), encoding="utf-8")


def main():
    torch, AutoImageProcessor, AutoModelForDepthEstimation, model_info = (
        ensure_training_dependencies()
    )
    import torch.nn.functional as F

    seed_everything(torch)
    require_cuda(torch)
    device = torch.device("cuda")

    posed_images = find_posed_images_root()
    out_dir = Path("/kaggle/working/outputs") / RUN_NAME
    out_dir.mkdir(parents=True, exist_ok=True)

    scene_dirs = sorted(path.name for path in posed_images.iterdir() if path.is_dir())
    scene_splits = assign_scene_splits(scene_dirs)
    frame_rows = discover_complete_rgbd_pose_frames(
        posed_images,
        scene_splits=scene_splits,
    )
    if not frame_rows:
        raise RuntimeError(f"No complete RGB/depth/pose frames found in {posed_images}")

    scene_split_rows = [
        {"scene": scene, "split": split}
        for scene, split in sorted(scene_splits.items())
    ]
    write_csv(out_dir / "scene_split.csv", scene_split_rows)
    write_csv(out_dir / "frame_manifest.csv", frame_rows)
    write_csv(out_dir / "split_summary.csv", split_summary(frame_rows))

    train_rows = [row for row in frame_rows if row["split"] == "train"]
    all_rows = list(frame_rows)
    val_eval_rows = select_eval_rows(frame_rows, "val")
    test_eval_rows = select_eval_rows(frame_rows, "test")
    if not train_rows or not val_eval_rows or not test_eval_rows:
        raise RuntimeError("Need non-empty train/val/test frame splits for Run 37.")

    processor, model, resolved_revision = load_model(
        torch,
        AutoImageProcessor,
        AutoModelForDepthEstimation,
        model_info,
    )
    try:
        model.gradient_checkpointing_enable()
        print("Enabled gradient checkpointing.")
    except Exception as exc:
        print("Gradient checkpointing unavailable:", repr(exc))
    model.to(device)

    config = {
        "run": RUN_NAME,
        "posed_images": str(posed_images),
        "model_name": MODEL_NAME,
        "requested_revision": MODEL_REVISION,
        "resolved_revision": resolved_revision,
        "image_size": IMAGE_SIZE,
        "batch_size": BATCH_SIZE,
        "grad_accum_steps": GRAD_ACCUM_STEPS,
        "controlled_epochs": CONTROLLED_EPOCHS,
        "deployment_epochs": DEPLOYMENT_EPOCHS,
        "max_train_steps": MAX_TRAIN_STEPS,
        "max_deployment_steps": MAX_DEPLOYMENT_STEPS,
        "max_eval_frames_per_split": MAX_EVAL_FRAMES_PER_SPLIT,
        "min_depth_m": MIN_DEPTH_M,
        "max_depth_m": MAX_DEPTH_M,
        "backbone_lr": BACKBONE_LR,
        "head_lr": HEAD_LR,
        "weight_decay": WEIGHT_DECAY,
        "gate_absrel_margin": GATE_ABSREL_MARGIN,
        "gate_delta1_margin": GATE_DELTA1_MARGIN,
        "num_scenes": len(scene_dirs),
        "num_frames": len(frame_rows),
        "num_train_frames": len(train_rows),
        "num_val_eval_frames": len(val_eval_rows),
        "num_test_eval_frames": len(test_eval_rows),
        "note": (
            "Run 37 fine-tunes Depth Anything V2 Metric Indoor Small on the full "
            "Kaggle ScanNet-style RGB-D frame pool discovered under posed_images. "
            "Controlled_best trains only on train scenes and selects by validation "
            "depth quality. Full_dataset_deployment continues training on 100% "
            "discovered frames and is for deployment, not unbiased reporting."
        ),
    }
    write_json(out_dir / "run_config.json", config)
    print("Run 37 config:")
    print(json.dumps(config, indent=2))

    baseline_rows = []
    baseline_rows.extend(
        evaluate(torch, F, model, processor, val_eval_rows, device, "val", "baseline")
    )
    baseline_rows.extend(
        evaluate(torch, F, model, processor, test_eval_rows, device, "test", "baseline")
    )
    write_csv(out_dir / "baseline_depth_metrics.csv", baseline_rows)
    baseline_summary = [
        summarize_depth_rows(baseline_rows, "val"),
        summarize_depth_rows(baseline_rows, "test"),
        summarize_depth_rows(baseline_rows, "all"),
    ]
    baseline_summary = [row for row in baseline_summary if row is not None]
    write_csv(out_dir / "baseline_depth_summary.csv", baseline_summary)
    print("Baseline summary:")
    for row in baseline_summary:
        print(row)

    optimizer = optimizer_for(torch, model)
    scaler = torch.cuda.amp.GradScaler(enabled=device.type == "cuda")
    history = []
    best_val_absrel = float("inf")
    best_epoch = -1
    best_dir = out_dir / "checkpoints" / "controlled_best"

    for epoch in range(1, CONTROLLED_EPOCHS + 1):
        train_row = train_epoch(
            torch,
            F,
            model,
            processor,
            train_rows,
            optimizer,
            scaler,
            device,
            epoch,
            MAX_TRAIN_STEPS,
        )
        val_rows_epoch = evaluate(
            torch,
            F,
            model,
            processor,
            val_eval_rows,
            device,
            "val",
            f"controlled_epoch_{epoch}",
        )
        val_summary = summarize_depth_rows(val_rows_epoch, "val")
        train_row.update(
            {
                "tag": f"controlled_epoch_{epoch}",
                "val_mean_abs_rel": val_summary["mean_abs_rel"],
                "val_mean_delta1": val_summary["mean_delta1"],
                "val_mean_mae": val_summary["mean_mae"],
                "val_num_frames": val_summary["num_frames"],
            }
        )
        history.append(train_row)
        print("Run 37 train row:", train_row)
        if val_summary["mean_abs_rel"] < best_val_absrel:
            best_val_absrel = val_summary["mean_abs_rel"]
            best_epoch = epoch
            best_dir.mkdir(parents=True, exist_ok=True)
            model.save_pretrained(best_dir)
            processor.save_pretrained(best_dir)
            write_csv(out_dir / "controlled_best_val_metrics.csv", val_rows_epoch)
            write_json(
                out_dir / "controlled_best_summary.json",
                {
                    "best_epoch": best_epoch,
                    "best_val_mean_abs_rel": best_val_absrel,
                    **val_summary,
                },
            )

    if best_epoch >= 0:
        model = AutoModelForDepthEstimation.from_pretrained(best_dir).to(device)

    controlled_rows = []
    controlled_rows.extend(
        evaluate(
            torch,
            F,
            model,
            processor,
            val_eval_rows,
            device,
            "val",
            "controlled_best",
        )
    )
    controlled_rows.extend(
        evaluate(
            torch,
            F,
            model,
            processor,
            test_eval_rows,
            device,
            "test",
            "controlled_best",
        )
    )
    write_csv(out_dir / "controlled_best_depth_metrics.csv", controlled_rows)
    controlled_summary = [
        summarize_depth_rows(controlled_rows, "val"),
        summarize_depth_rows(controlled_rows, "test"),
        summarize_depth_rows(controlled_rows, "all"),
    ]
    controlled_summary = [row for row in controlled_summary if row is not None]
    write_csv(out_dir / "controlled_best_depth_summary.csv", controlled_summary)
    write_csv(out_dir / "training_history.csv", history)

    sample_manifest = save_sample_pngs(
        torch,
        F,
        model,
        processor,
        test_eval_rows,
        device,
        out_dir / "sample_predictions" / "controlled_best",
    )
    write_csv(out_dir / "sample_prediction_manifest.csv", sample_manifest)

    baseline_val = next(row for row in baseline_summary if row["split"] == "val")
    controlled_val = next(row for row in controlled_summary if row["split"] == "val")
    baseline_test = next(row for row in baseline_summary if row["split"] == "test")
    controlled_test = next(row for row in controlled_summary if row["split"] == "test")
    gate_row = {
        "run": RUN_NAME,
        "selected_checkpoint": "controlled_best",
        "best_epoch": best_epoch,
        "baseline_val_abs_rel": baseline_val["mean_abs_rel"],
        "controlled_val_abs_rel": controlled_val["mean_abs_rel"],
        "val_abs_rel_delta": controlled_val["mean_abs_rel"] - baseline_val["mean_abs_rel"],
        "baseline_val_delta1": baseline_val["mean_delta1"],
        "controlled_val_delta1": controlled_val["mean_delta1"],
        "val_delta1_delta": controlled_val["mean_delta1"] - baseline_val["mean_delta1"],
        "baseline_test_abs_rel": baseline_test["mean_abs_rel"],
        "controlled_test_abs_rel": controlled_test["mean_abs_rel"],
        "test_abs_rel_delta": controlled_test["mean_abs_rel"] - baseline_test["mean_abs_rel"],
        "baseline_test_delta1": baseline_test["mean_delta1"],
        "controlled_test_delta1": controlled_test["mean_delta1"],
        "test_delta1_delta": controlled_test["mean_delta1"] - baseline_test["mean_delta1"],
    }
    gate_row["passes_depth_gate"] = int(
        gate_row["val_abs_rel_delta"] <= -GATE_ABSREL_MARGIN
        and gate_row["val_delta1_delta"] >= GATE_DELTA1_MARGIN
    )
    write_csv(out_dir / "gate_decision.csv", [gate_row])
    print("Run 37 gate decision:")
    print(gate_row)

    if TRAIN_FULL_DEPLOYMENT:
        deployment_optimizer = optimizer_for(torch, model)
        deployment_scaler = torch.cuda.amp.GradScaler(enabled=device.type == "cuda")
        deployment_history = []
        for epoch in range(1, DEPLOYMENT_EPOCHS + 1):
            row = train_epoch(
                torch,
                F,
                model,
                processor,
                all_rows,
                deployment_optimizer,
                deployment_scaler,
                device,
                epoch,
                MAX_DEPLOYMENT_STEPS,
            )
            row["tag"] = f"full_dataset_deployment_epoch_{epoch}"
            deployment_history.append(row)
            print("Run 37 deployment train row:", row)
        deployment_dir = out_dir / "checkpoints" / "full_dataset_deployment"
        deployment_dir.mkdir(parents=True, exist_ok=True)
        model.save_pretrained(deployment_dir)
        processor.save_pretrained(deployment_dir)
        write_csv(out_dir / "deployment_training_history.csv", deployment_history)
        write_json(
            out_dir / "full_dataset_deployment_summary.json",
            {
                "checkpoint": str(deployment_dir),
                "num_training_frames_available": len(all_rows),
                "num_training_scenes_available": len(scene_dirs),
                "deployment_epochs": DEPLOYMENT_EPOCHS,
                "max_deployment_steps": MAX_DEPLOYMENT_STEPS,
                "reporting_note": (
                    "This checkpoint is trained on all discovered scenes/frames. "
                    "Use controlled_best, not this deployment checkpoint, for "
                    "unbiased val/test claims."
                ),
            },
        )

    print(f"Run 37 complete. Outputs: {out_dir}")


if __name__ == "__main__":
    main()
