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
import matplotlib.pyplot as plt
from PIL import Image
from scipy.spatial import cKDTree


REPO_URL = "https://github.com/facebookresearch/mvdust3r.git"
DATASET_SLUG = "tiantiansyrinx1102/scannet-data"
SCENE = "scene0000_00"
SEED = 777
TEST_SEEDS = [777, 778, 779]
VIEW_COUNTS = [2, 3, 4, 5]
CONF_PERCENT = 3.0
FIXED_FINAL_CONF_BY_VIEW_COUNT = {2: 3.0, 3: 0.2, 4: 3.0, 5: 0.5}
BEST_POLICY_BY_VIEW_COUNT = {2: "hybrid", 3: "diversity_aware", 4: "hybrid", 5: "diversity_aware"}
CASE_POLICIES = {
    "normal": "best",
    "far_reference": "diversity_aware",
    "occlusion_heavy": "overlap_aware",
}
QUAL_CASES = ["normal", "far_reference", "occlusion_heavy"]
MAX_SCENES = 2
IMAGE_SIZE = 224
MAX_POINTS = 50000
F_SCORE_THRESHOLD_M = 0.05
TORCH_REEXEC_FLAG = "RUN11_TORCH_REEXECED_AFTER_COMPAT_INSTALL"


def run(cmd, **kwargs):
    print("+", " ".join(map(str, cmd)))
    subprocess.run(cmd, check=True, **kwargs)


def verify_cuda_usable():
    try:
        x = torch.ones(1, device="cuda")
        y = (x + 1).detach().cpu().item()
        return y == 2.0, None
    except Exception as exc:
        return False, repr(exc)


def install_p100_compatible_torch_and_reexec(names):
    if os.environ.get(TORCH_REEXEC_FLAG) == "1":
        raise RuntimeError(
            "CUDA is still unusable after one Torch compatibility reinstall. "
            f"GPU names: {names}; torch={torch.__version__}"
        )
    if not any("P100" in name for name in names):
        raise RuntimeError(
            "CUDA is unusable and this script only auto-reinstalls Torch for P100 fallback. "
            f"GPU names: {names}; torch={torch.__version__}"
        )
    print("Detected P100 with an incompatible Torch/CUDA build.")
    print("Installing Torch 2.5.1 cu121 and restarting Run 11 once.")
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


def require_t4x2():
    print("CUDA available:", torch.cuda.is_available())
    print("CUDA device count:", torch.cuda.device_count())
    names = []
    if torch.cuda.is_available():
        for i in range(torch.cuda.device_count()):
            name = torch.cuda.get_device_name(i)
            names.append(name)
            print(f"GPU {i}:", name)
    print("Torch:", torch.__version__)
    if not torch.cuda.is_available() or torch.cuda.device_count() < 1:
        raise RuntimeError("Run 11 requires at least one usable CUDA GPU.")
    ok, err = verify_cuda_usable()
    if not ok:
        print("CUDA smoke test failed:", err)
        install_p100_compatible_torch_and_reexec(names)
    if not (torch.cuda.device_count() >= 2 and all("T4" in n for n in names)):
        print(f"Warning: expected T4 x2, got {names}. Continuing with available compatible GPU(s).")


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


def configure_hf_token():
    if os.environ.get("HF_TOKEN") or os.environ.get("HUGGINGFACE_HUB_TOKEN"):
        return
    try:
        from kaggle_secrets import UserSecretsClient

        secrets = UserSecretsClient()
        for name in ("HF_TOKEN", "HUGGINGFACE_TOKEN", "HUGGINGFACE_HUB_TOKEN"):
            try:
                token = secrets.get_secret(name)
            except Exception:
                token = None
            if token:
                os.environ["HF_TOKEN"] = token
                os.environ["HUGGINGFACE_HUB_TOKEN"] = token
                print(f"Using Hugging Face token from Kaggle secret: {name}")
                return
    except Exception as exc:
        print("HF token secret lookup skipped:", type(exc).__name__)


def download_checkpoint(root):
    from huggingface_hub import hf_hub_download

    configure_hf_token()
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


def choose_random_views(scene_dir, view_count, seed=SEED):
    jpgs = sorted(scene_dir.glob("*.jpg"))[:80]
    rng = random.Random(seed + view_count)
    chosen = sorted(rng.sample(jpgs, view_count))
    return chosen


def camera_center(jpg_path):
    pose = parse_pose(str(jpg_path).replace(".jpg", ".txt"))
    return pose[:3, 3]


def choose_diversity_views(scene_dir, view_count):
    jpgs = sorted(scene_dir.glob("*.jpg"))[:80]
    centers = np.stack([camera_center(p) for p in jpgs], axis=0)
    chosen = [0]
    while len(chosen) < view_count:
        dists = []
        for i in range(len(jpgs)):
            if i in chosen:
                dists.append(-1)
                continue
            min_dist = np.linalg.norm(centers[i] - centers[chosen], axis=1).min()
            dists.append(min_dist)
        chosen.append(int(np.argmax(dists)))
    return sorted([jpgs[i] for i in chosen])


def choose_overlap_views(scene_dir, view_count):
    # A conservative overlap-aware proxy: choose a short temporal window.
    # Nearby ScanNet posed frames usually preserve overlap and reduce far-reference failures.
    jpgs = sorted(scene_dir.glob("*.jpg"))[:80]
    if view_count == 1:
        return [jpgs[0]]
    start = min(10, max(0, len(jpgs) - view_count))
    step = 2
    ids = [min(start + i * step, len(jpgs) - 1) for i in range(view_count)]
    return [jpgs[i] for i in ids]


def choose_hybrid_views(scene_dir, view_count):
    # Hybrid = keep local overlap but cover a little more trajectory than pure overlap.
    jpgs = sorted(scene_dir.glob("*.jpg"))[:80]
    if view_count == 2:
        return [jpgs[10], jpgs[18]]
    if view_count == 3:
        ids = [8, 16, 28]
    elif view_count == 4:
        ids = [8, 14, 24, 36]
    else:
        ids = [8, 14, 22, 32, 44]
    return [jpgs[min(i, len(jpgs) - 1)] for i in ids[:view_count]]


def choose_views(scene_dir, view_count, policy, seed=SEED):
    if policy == "best":
        policy = BEST_POLICY_BY_VIEW_COUNT[view_count]
    if policy == "random":
        return choose_random_views(scene_dir, view_count, seed=seed)
    if policy == "diversity_aware":
        return choose_diversity_views(scene_dir, view_count)
    if policy == "overlap_aware":
        return choose_overlap_views(scene_dir, view_count)
    if policy == "hybrid":
        return choose_hybrid_views(scene_dir, view_count)
    raise ValueError(f"Unknown view policy: {policy}")


def discover_scene_dirs(posed_root):
    scenes = sorted([p for p in posed_root.glob("scene*") if p.is_dir() and list(p.glob("*.jpg"))])
    if not scenes:
        scene_dir = posed_root / SCENE
        if scene_dir.exists():
            scenes = [scene_dir]
    return scenes[:MAX_SCENES]


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


def summarize_final_vs_b0(rows):
    summary = []
    for vc in VIEW_COUNTS:
        b0_rows = [r for r in rows if r["method_id"] == "B0" and r["num_views"] == vc]
        final_rows = [r for r in rows if r["method_id"] == "Final" and r["num_views"] == vc]
        if not b0_rows or not final_rows:
            continue
        b0_f = np.array([r["fscore"] for r in b0_rows], dtype=np.float32)
        final_f = np.array([r["fscore"] for r in final_rows], dtype=np.float32)
        summary.append(
            {
                "run": "run_11_final_validation_3seeds",
                "num_views": vc,
                "num_b0_trials": len(b0_rows),
                "num_final_trials": len(final_rows),
                "b0_mean_fscore": float(b0_f.mean()),
                "b0_std_fscore": float(b0_f.std()),
                "final_mean_fscore": float(final_f.mean()),
                "final_std_fscore": float(final_f.std()),
                "delta_mean_fscore": float(final_f.mean() - b0_f.mean()),
                "b0_mean_runtime_seconds": float(np.mean([r["runtime_seconds"] for r in b0_rows])),
                "final_mean_runtime_seconds": float(np.mean([r["runtime_seconds"] for r in final_rows])),
            }
        )
    return summary


def scatter_cloud(ax, points, title, max_points=7000):
    pts = points[np.isfinite(points).all(axis=1)]
    if len(pts) > max_points:
        rng = np.random.default_rng(SEED)
        pts = pts[rng.choice(len(pts), max_points, replace=False)]
    ax.scatter(pts[:, 0], pts[:, 2], c=pts[:, 1], s=0.18, cmap="viridis", linewidths=0)
    ax.set_title(title, fontsize=9)
    ax.set_xticks([])
    ax.set_yticks([])
    ax.set_aspect("equal", adjustable="box")


def make_qualitative_figure(cases, out_path):
    fig, axes = plt.subplots(len(cases), 2, figsize=(7.2, 9.0), dpi=170)
    for i, item in enumerate(cases):
        scatter_cloud(axes[i, 0], item["b0_pred"], f"{item['case_type']} - B0")
        scatter_cloud(axes[i, 1], item["final_pred"], f"{item['case_type']} - Final")
    fig.suptitle("Qualitative point-cloud comparison, B0 vs Final", fontsize=12)
    fig.tight_layout(rect=[0, 0, 1, 0.97])
    fig.savefig(out_path)
    plt.close(fig)


def main():
    require_t4x2()
    root = clone_repo()
    install_deps(root)
    posed_root = find_posed_images_root()
    scene_dirs = discover_scene_dirs(posed_root)
    print("POSED_IMAGES:", posed_root)
    print("SCENE_DIRS:", [str(p) for p in scene_dirs])

    ckpt_path = download_checkpoint(root)
    print("Checkpoint:", ckpt_path)
    model = load_model(root, ckpt_path)

    outputs_root = Path("/kaggle/working/outputs")

    # Run 11: final validation with fixed thresholds and three B0 seeds.
    run11_dir = outputs_root / "run_11_final_validation_3seeds"
    rows = []
    selected_views = {}
    qualitative_cases = []

    for scene_dir in scene_dirs:
        for seed in TEST_SEEDS:
            for vc in VIEW_COUNTS:
                b0_dir = run11_dir / "b0" / scene_dir.name / f"seed_{seed}" / f"{vc}_views"
                b0_views = choose_views(scene_dir, vc, "random", seed=seed)
                selected_views[f"B0_{scene_dir.name}_seed_{seed}_{vc}"] = [str(p) for p in b0_views]
                b0_output, b0_glb, b0_runtime = run_inference(model, root, b0_views, b0_dir)
                b0_gt, _ = build_gt_cloud(b0_views)
                b0_pred, b0_conf_threshold = output_to_pred_cloud(b0_output, conf_percent=CONF_PERCENT)
                b0_metrics = compute_metrics(b0_pred, b0_gt)
                np.savez_compressed(b0_dir / "point_clouds.npz", pred=b0_pred, gt=b0_gt)
                b0_row = {
                    "run": "run_11_final_validation_3seeds",
                    "method_id": "B0",
                    "method": "MV-DUSt3R+ default + random views",
                    "scene": scene_dir.name,
                    "seed": seed,
                    "num_views": vc,
                    "view_policy": "random",
                    "fusion": "default",
                    "occlusion": "disabled",
                    "ambiguity": "disabled",
                    "runtime_seconds": b0_runtime,
                    "conf_percent": CONF_PERCENT,
                    "conf_threshold": b0_conf_threshold,
                    "output_glb": str(b0_glb),
                    **b0_metrics,
                }
                rows.append(b0_row)
                print("Run 11 row:", b0_row)
                del b0_output
                torch.cuda.empty_cache()

                final_policy = BEST_POLICY_BY_VIEW_COUNT[vc]
                final_dir = run11_dir / "final" / scene_dir.name / f"seed_{seed}" / f"{vc}_views"
                final_views = choose_views(scene_dir, vc, final_policy, seed=seed)
                selected_views[f"Final_{scene_dir.name}_seed_{seed}_{vc}"] = [str(p) for p in final_views]
                final_output, final_glb, final_runtime = run_inference(model, root, final_views, final_dir)
                final_gt, _ = build_gt_cloud(final_views)
                final_conf_percent = FIXED_FINAL_CONF_BY_VIEW_COUNT[vc]
                final_pred, final_conf_threshold = output_to_pred_cloud(final_output, conf_percent=final_conf_percent)
                final_metrics = compute_metrics(final_pred, final_gt)
                np.savez_compressed(final_dir / "point_clouds.npz", pred=final_pred, gt=final_gt)
                row = {
                    "run": "run_11_final_validation_3seeds",
                    "method_id": "Final",
                    "method": "Final fixed-threshold pipeline",
                    "scene": scene_dir.name,
                    "seed": seed,
                    "num_views": vc,
                    "view_policy": final_policy,
                    "fusion": "F0_baseline",
                    "occlusion": "disabled_after_run6",
                    "ambiguity": "disabled_after_run7",
                    "runtime_seconds": final_runtime,
                    "conf_percent": final_conf_percent,
                    "conf_threshold": final_conf_threshold,
                    "output_glb": str(final_glb),
                    **final_metrics,
                }
                rows.append(row)
                print("Run 11 row:", row)
                del final_output
                torch.cuda.empty_cache()

    # Qualitative B0 vs Final on three case policies at 4 views, seed 777.
    qual_scene = scene_dirs[0]
    qual_seed = TEST_SEEDS[0]
    qual_view_count = 4
    for case_type in QUAL_CASES:
        case_policy = CASE_POLICIES[case_type]
        final_policy = BEST_POLICY_BY_VIEW_COUNT[qual_view_count] if case_policy == "best" else case_policy
        b0_views = choose_views(qual_scene, qual_view_count, "random", seed=qual_seed)
        final_views = choose_views(qual_scene, qual_view_count, final_policy, seed=qual_seed)

        b0_out, _, _ = run_inference(model, root, b0_views, run11_dir / "qualitative" / case_type / "b0")
        final_out, _, _ = run_inference(model, root, final_views, run11_dir / "qualitative" / case_type / "final")
        b0_pred, _ = output_to_pred_cloud(b0_out, conf_percent=CONF_PERCENT)
        final_pred, _ = output_to_pred_cloud(final_out, conf_percent=FIXED_FINAL_CONF_BY_VIEW_COUNT[qual_view_count])
        qualitative_cases.append({"case_type": case_type, "b0_pred": b0_pred, "final_pred": final_pred})
        del b0_out, final_out
        torch.cuda.empty_cache()

    make_qualitative_figure(qualitative_cases, run11_dir / "fig_qualitative_b0_vs_final.png")
    write_csv(run11_dir / "metrics.csv", rows)
    summary_rows = summarize_final_vs_b0(rows)
    write_csv(run11_dir / "summary_final_vs_b0_3seeds.csv", summary_rows)
    (run11_dir / "selected_views.json").write_text(json.dumps(selected_views, indent=2))
    (run11_dir / "run_config.json").write_text(
        json.dumps(
            {
                "run": "run_11_final_validation_3seeds",
                "dataset_slug": DATASET_SLUG,
                "scenes": [p.name for p in scene_dirs],
                "seeds": TEST_SEEDS,
                "view_counts": VIEW_COUNTS,
                "best_policy_by_view_count": BEST_POLICY_BY_VIEW_COUNT,
                "fixed_final_conf_by_view_count": FIXED_FINAL_CONF_BY_VIEW_COUNT,
                "b0_conf_percent": CONF_PERCENT,
                "seed": SEED,
                "run5_takeaway": "Custom F1/F2/F3 fusion variants underperformed F0 baseline.",
                "run6_takeaway": "Front-depth occlusion filtering underperformed O0 baseline.",
                "run7_takeaway": "Ambiguity filtering variants underperformed A0 no-ambiguity baseline.",
                "run10_takeaway": "Final thresholds are fixed from validation sensitivity and are not retuned in this run.",
                "full_pipeline_definition": "Best view policy + fixed confidence threshold + F0 baseline; occlusion and ambiguity disabled because ablations were negative.",
                "gt_source": "depth PNG + pose TXT from scannet-data",
                "alignment": "center + RMS scale alignment before nearest-neighbor metrics",
                "metric_threshold_m": F_SCORE_THRESHOLD_M,
                "max_scenes": MAX_SCENES,
            },
            indent=2,
        )
    )
    print("Run 11 metrics:")
    for row in rows:
        print(row)
    print("Run 11 summary final vs B0 3 seeds:")
    for row in summary_rows:
        print(row)
    print("Run 11 qualitative figure:")
    print(str(run11_dir / "fig_qualitative_b0_vs_final.png"))


if __name__ == "__main__":
    main()
