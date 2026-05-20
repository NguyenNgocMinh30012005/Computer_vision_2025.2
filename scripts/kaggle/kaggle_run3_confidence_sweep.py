import builtins
import csv
import json
import os
import random
import shutil
import subprocess
import sys
import time
import types
from pathlib import Path

import numpy as np
import torch
from PIL import Image
from scipy.spatial import cKDTree


REPO_URL = "https://github.com/facebookresearch/mvdust3r.git"
DATASET_SLUG = "tiantiansyrinx1102/scannet-data"
SCENE = "scene0000_00"
SEED = 777
VIEW_COUNTS = [2, 3, 4, 5]
CONF_PERCENT = 3.0
CONF_SWEEP = [0.1, 0.2, 0.3, 0.5, 0.7]
IMAGE_SIZE = 224
MAX_POINTS = 50000
F_SCORE_THRESHOLD_M = 0.05


def run(cmd, **kwargs):
    print("+", " ".join(map(str, cmd)))
    subprocess.run(cmd, check=True, **kwargs)


def require_t4x2():
    print("CUDA available:", torch.cuda.is_available())
    print("CUDA device count:", torch.cuda.device_count())
    names = []
    if torch.cuda.is_available():
        for i in range(torch.cuda.device_count()):
            name = torch.cuda.get_device_name(i)
            names.append(name)
            print(f"GPU {i}:", name)
    assert torch.cuda.device_count() >= 2 and all("T4" in n for n in names), f"Expected T4 x2, got: {names}"
    print("Torch:", torch.__version__)


def clone_repo():
    root = Path("/kaggle/temp/mvdust3r")
    if root.exists():
        shutil.rmtree(root)
    run(["git", "clone", "--depth", "1", REPO_URL, str(root)])
    return root


def install_deps(root):
    req_src = root / "requirements.txt"
    req_kaggle = Path("/kaggle/working/requirements_kaggle.txt")
    skip_prefixes = ("torch==", "torchvision==", "tensorflow", "gsplat==", "numpy==", "scipy", "opencv-python")
    lines = []
    for line in req_src.read_text().splitlines():
        s = line.strip()
        if not s or s.startswith("#"):
            continue
        if s.startswith(skip_prefixes):
            print("Skip Kaggle-managed/heavy package:", s)
            continue
        lines.append(s)
    req_kaggle.write_text("\n".join(lines) + "\n")
    run([sys.executable, "-m", "pip", "install", "-q", "-r", str(req_kaggle)])


def find_posed_images_root():
    root = Path("/kaggle/input")
    candidates = [
        root / "scannet-data/scannet/posed_images",
        root / "scannet-data/posed_images",
        root / "datasets/tiantiansyrinx1102/scannet-data/scannet/posed_images",
        root / "datasets/tiantiansyrinx1102/scannet-data/posed_images",
    ]
    for p in candidates:
        if p.exists():
            return p
    for p in root.rglob("posed_images"):
        if p.exists():
            return p
    print("Input tree preview:")
    for i, p in enumerate(sorted(root.rglob("*"))):
        if i > 120:
            print("  ...")
            break
        print(" ", p.relative_to(root), "/" if p.is_dir() else "")
    raise FileNotFoundError("Cannot locate posed_images in /kaggle/input")


def install_pytorch3d_shim():
    try:
        import pytorch3d  # noqa: F401
        return
    except ModuleNotFoundError:
        pass

    pytorch3d = types.ModuleType("pytorch3d")
    ops = types.ModuleType("pytorch3d.ops")
    transforms = types.ModuleType("pytorch3d.transforms")

    def knn_points(x, y, K=1, **kwargs):
        d = torch.cdist(x, y)
        vals, idx = torch.topk(d, k=K, dim=-1, largest=False)
        return vals, idx, None

    def so3_relative_angle(rot_gt, rot_pred, eps=1e-4, **kwargs):
        r = rot_gt @ rot_pred.transpose(-1, -2)
        trace = r[..., 0, 0] + r[..., 1, 1] + r[..., 2, 2]
        cos = ((trace - 1.0) / 2.0).clamp(-1 + eps, 1 - eps)
        return torch.acos(cos)

    ops.knn_points = knn_points
    transforms.so3_relative_angle = so3_relative_angle
    pytorch3d.ops = ops
    pytorch3d.transforms = transforms
    sys.modules["pytorch3d"] = pytorch3d
    sys.modules["pytorch3d.ops"] = ops
    sys.modules["pytorch3d.transforms"] = transforms


def patch_torch_load():
    torch_load = torch.load

    def torch_load_compat(*args, **kwargs):
        kwargs.setdefault("weights_only", False)
        return torch_load(*args, **kwargs)

    torch.load = torch_load_compat


def download_checkpoint(root):
    from huggingface_hub import hf_hub_download

    (root / "checkpoints").mkdir(parents=True, exist_ok=True)
    return hf_hub_download(
        repo_id="Zhenggang/MV-DUSt3R",
        filename="checkpoints/MVDp_s2.pth",
        local_dir=str(root),
        local_dir_use_symlinks=False,
    )


def load_model(root, ckpt_path):
    sys.path.insert(0, str(root))
    os.chdir(root)
    builtins.input = lambda prompt="": ""
    install_pytorch3d_shim()
    patch_torch_load()

    from demo import AsymmetricCroCo3DStereoMultiView

    inf = np.inf
    device = "cuda"
    model = AsymmetricCroCo3DStereoMultiView(
        pos_embed="RoPE100",
        img_size=(224, 224),
        head_type="linear",
        output_mode="pts3d",
        depth_mode=("exp", -inf, inf),
        conf_mode=("exp", 1, 1e9),
        enc_embed_dim=1024,
        enc_depth=24,
        enc_num_heads=16,
        dec_embed_dim=768,
        dec_depth=12,
        dec_num_heads=12,
        GS=True,
        sh_degree=0,
        pts_head_config={"skip": True},
        m_ref_flag=True,
        n_ref=4,
    ).to(device)
    loaded = AsymmetricCroCo3DStereoMultiView.from_pretrained(str(ckpt_path)).to(device)
    model.load_state_dict(loaded.state_dict(), strict=True)
    model.eval()
    del loaded
    torch.cuda.empty_cache()
    return model


def choose_random_views(scene_dir, view_count):
    jpgs = sorted(scene_dir.glob("*.jpg"))[:80]
    rng = random.Random(SEED + view_count)
    chosen = sorted(rng.sample(jpgs, view_count))
    return chosen


def parse_pose(path):
    values = [float(x) for x in Path(path).read_text().split()]
    if len(values) == 16:
        return np.array(values, dtype=np.float32).reshape(4, 4)
    raise ValueError(f"Unexpected pose format in {path}: {len(values)} values")


def depth_to_points(depth_path, pose_path, first_pose, stride=4):
    depth = np.array(Image.open(depth_path))
    if depth.ndim == 3:
        depth = depth[..., 0]
    depth = depth.astype(np.float32)
    if depth.max() > 100:
        depth = depth / 1000.0

    h, w = depth.shape
    fx = 577.870605 * (w / 640.0)
    fy = 577.870605 * (h / 480.0)
    cx = 319.5 * (w / 640.0)
    cy = 239.5 * (h / 480.0)

    ys, xs = np.mgrid[0:h:stride, 0:w:stride]
    z = depth[ys, xs]
    valid = z > 0
    xs = xs[valid].astype(np.float32)
    ys = ys[valid].astype(np.float32)
    z = z[valid]
    x = (xs - cx) / fx * z
    y = (ys - cy) / fy * z
    pts_cam = np.stack([x, y, z, np.ones_like(z)], axis=1)

    pose = parse_pose(pose_path)
    pts_world = (pose @ pts_cam.T).T[:, :3]
    pts_first = (np.linalg.inv(first_pose) @ np.c_[pts_world, np.ones(len(pts_world))].T).T[:, :3]
    return pts_first.astype(np.float32)


def build_gt_cloud(view_files):
    first_pose = parse_pose(str(view_files[0]).replace(".jpg", ".txt"))
    clouds = []
    stats = []
    for jpg in view_files:
        depth_path = str(jpg).replace(".jpg", ".png")
        pose_path = str(jpg).replace(".jpg", ".txt")
        depth = np.array(Image.open(depth_path))
        stats.append(
            {
                "frame": jpg.name,
                "depth_shape": list(depth.shape),
                "depth_dtype": str(depth.dtype),
                "depth_min": float(depth.min()),
                "depth_max": float(depth.max()),
            }
        )
        clouds.append(depth_to_points(depth_path, pose_path, first_pose))
    gt = np.concatenate(clouds, axis=0)
    return downsample(gt, MAX_POINTS), stats


def output_to_pred_cloud(output, conf_percent=CONF_PERCENT):
    pts = [output["pred1"]["pts3d"][0].detach().cpu()]
    pts += [x["pts3d_in_other_view"][0].detach().cpu() for x in output["pred2s"]]
    conf = [output["pred1"]["conf"][0].detach().cpu()]
    conf += [x["conf"][0].detach().cpu() for x in output["pred2s"]]

    pts = torch.stack(pts, dim=0).numpy()
    conf = torch.stack(conf, dim=0).numpy()
    threshold = np.quantile(conf.reshape(-1), conf_percent / 100.0)
    mask = conf >= threshold
    pred = pts[mask]
    pred = pred[np.isfinite(pred).all(axis=1)]
    return downsample(pred.astype(np.float32), MAX_POINTS), float(threshold)


def downsample(points, max_points):
    if len(points) <= max_points:
        return points
    rng = np.random.default_rng(SEED)
    idx = rng.choice(len(points), max_points, replace=False)
    return points[idx]


def center_scale_align(pred, gt):
    pred_c = pred - pred.mean(axis=0, keepdims=True)
    gt_c = gt - gt.mean(axis=0, keepdims=True)
    pred_scale = np.sqrt((pred_c**2).sum(axis=1).mean()) + 1e-8
    gt_scale = np.sqrt((gt_c**2).sum(axis=1).mean()) + 1e-8
    return pred_c / pred_scale * gt_scale + gt.mean(axis=0, keepdims=True)


def compute_metrics(pred, gt, threshold=F_SCORE_THRESHOLD_M):
    pred = center_scale_align(pred, gt)
    tree_gt = cKDTree(gt)
    tree_pred = cKDTree(pred)
    d_pred_to_gt, _ = tree_gt.query(pred, k=1, workers=-1)
    d_gt_to_pred, _ = tree_pred.query(gt, k=1, workers=-1)

    precision = float((d_pred_to_gt < threshold).mean())
    recall = float((d_gt_to_pred < threshold).mean())
    fscore = 0.0 if precision + recall == 0 else 2 * precision * recall / (precision + recall)
    return {
        "accuracy": float(d_pred_to_gt.mean()),
        "completeness": float(d_gt_to_pred.mean()),
        "precision": precision,
        "recall": recall,
        "fscore": float(fscore),
        "chamfer": float(d_pred_to_gt.mean() + d_gt_to_pred.mean()),
        "num_pred_points": int(len(pred)),
        "num_gt_points": int(len(gt)),
        "threshold_m": float(threshold),
    }


def run_inference(model, root, view_files, out_dir):
    from demo import get_reconstructed_scene

    out_dir.mkdir(parents=True, exist_ok=True)
    torch.cuda.empty_cache()
    original_n_ref = getattr(model, "n_ref", None)
    model.n_ref = max(2, min(4, len(view_files)))
    print(f"Running inference with num_views={len(view_files)} model.n_ref={model.n_ref}")
    start = time.time()
    try:
        with torch.no_grad():
            output, glb_file, _ = get_reconstructed_scene(
                str(out_dir),
                model,
                "cuda",
                True,
                IMAGE_SIZE,
                [str(p) for p in view_files],
                CONF_PERCENT,
                True,
                False,
                0.05,
                10,
            )
        runtime = time.time() - start
        return output, glb_file, runtime
    finally:
        model.n_ref = original_n_ref


def write_csv(path, rows):
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", newline="") as f:
        writer = csv.DictWriter(f, fieldnames=list(rows[0].keys()))
        writer.writeheader()
        writer.writerows(rows)


def main():
    require_t4x2()
    root = clone_repo()
    install_deps(root)
    posed_root = find_posed_images_root()
    scene_dir = posed_root / SCENE
    print("POSED_IMAGES:", posed_root)
    print("SCENE_DIR:", scene_dir)

    ckpt_path = download_checkpoint(root)
    print("Checkpoint:", ckpt_path)
    model = load_model(root, ckpt_path)

    outputs_root = Path("/kaggle/working/outputs")

    # Run 3: Baseline B1 confidence threshold sweep.
    # We run inference once per view count, then sweep confidence percentiles
    # on the same model output. This isolates thresholding from inference noise.
    run3_dir = outputs_root / "run_03_baseline_b1_confidence"
    rows = []
    selected_views = {}
    for vc in VIEW_COUNTS:
        vc_dir = run3_dir / f"{vc}_views"
        view_files = choose_random_views(scene_dir, vc)
        selected_views[str(vc)] = [str(p) for p in view_files]
        output, glb_file, runtime = run_inference(model, root, view_files, vc_dir)
        gt, _ = build_gt_cloud(view_files)
        best_row = None
        for conf_percent in CONF_SWEEP:
            pred, conf_threshold = output_to_pred_cloud(output, conf_percent=conf_percent)
            metrics = compute_metrics(pred, gt)
            row = {
                "run": "run_03_baseline_b1_confidence",
                "method": "MV-DUSt3R+ random views + confidence sweep",
                "scene": SCENE,
                "num_views": vc,
                "runtime_seconds": runtime,
                "conf_percent": conf_percent,
                "conf_threshold": conf_threshold,
                "output_glb": str(glb_file),
                **metrics,
            }
            rows.append(row)
            if best_row is None or row["fscore"] > best_row["fscore"]:
                best_row = row
                np.savez_compressed(vc_dir / "best_point_clouds.npz", pred=pred, gt=gt)
        print("Best confidence row for", vc, "views:", best_row)
        del output
        torch.cuda.empty_cache()

    write_csv(run3_dir / "metrics.csv", rows)
    best_rows = []
    for vc in VIEW_COUNTS:
        vc_rows = [r for r in rows if r["num_views"] == vc]
        best_rows.append(max(vc_rows, key=lambda r: r["fscore"]))
    write_csv(run3_dir / "best_by_view_count.csv", best_rows)
    (run3_dir / "selected_views.json").write_text(json.dumps(selected_views, indent=2))
    (run3_dir / "run_config.json").write_text(
        json.dumps(
            {
                "run": "run_03_baseline_b1_confidence",
                "dataset_slug": DATASET_SLUG,
                "scene": SCENE,
                "view_counts": VIEW_COUNTS,
                "confidence_sweep": CONF_SWEEP,
                "view_policy": "random",
                "seed": SEED,
                "inference_conf_percent_for_glb_export": CONF_PERCENT,
                "gt_source": "depth PNG + pose TXT from scannet-data",
                "alignment": "center + RMS scale alignment before nearest-neighbor metrics",
                "metric_threshold_m": F_SCORE_THRESHOLD_M,
            },
            indent=2,
        )
    )
    print("Run 3 confidence sweep metrics:")
    for row in rows:
        print(row)
    print("Run 3 best by view count:")
    for row in best_rows:
        print(row)


if __name__ == "__main__":
    main()
