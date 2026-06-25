"""
Demo Inference Script — Sparse-View RGB 3D Reconstruction
=========================================================

Produces four visual outputs from user-supplied ScanNet RGB images:

  1. mvdust3r_only.glb        — Raw MV-DUSt3R+ candidate point cloud
  2. depth_backprojection.glb — Direct backprojection from fine-tuned depth
  3. corrected_final.glb      — MV-DUSt3R+ corrected by estimated depth (Ours)
  4. depth_maps/*.png         — Colorized predicted depth for each input view

Requirements
------------
- Kaggle T4 x2 GPU environment
- ScanNet posed images dataset mounted
- Run 37 fine-tuned depth checkpoint mounted

Environment variables
---------------------
  DEMO_SCENE        Scene directory name (default: scene0000_00)
  DEMO_NUM_VIEWS    Number of views to select (default: 5)
  DEMO_FRAMES       Comma-separated frame basenames to use instead of
                    automatic selection (e.g. "00000.jpg,00050.jpg,00100.jpg")
  DEMO_TAU          Correction threshold in meters (default: 0.30)
  DEMO_ALPHA        Correction blending weight (default: 1.0)
  DEMO_MAX_POINTS   Max points per output cloud (default: 50000)
"""

import json
import os
import random
import subprocess
import sys
import time
from collections import OrderedDict
from pathlib import Path

import numpy as np

# ---------------------------------------------------------------------------
# Configuration
# ---------------------------------------------------------------------------

DEMO_SCENE = os.environ.get("DEMO_SCENE", "scene0000_00")
DEMO_NUM_VIEWS = int(os.environ.get("DEMO_NUM_VIEWS", "5"))
DEMO_FRAMES = os.environ.get("DEMO_FRAMES", "")
DEMO_TAU = float(os.environ.get("DEMO_TAU", "0.30"))
DEMO_ALPHA = float(os.environ.get("DEMO_ALPHA", "1.0"))
DEMO_MAX_POINTS = int(os.environ.get("DEMO_MAX_POINTS", "50000"))
SEED = 4242
IMAGE_SIZE = 224

# ---------------------------------------------------------------------------
# Dependency helpers
# ---------------------------------------------------------------------------


def run_cmd(cmd, **kwargs):
    print("+", " ".join(map(str, cmd)))
    subprocess.run(cmd, check=True, **kwargs)


def install_trimesh():
    try:
        import trimesh
        return trimesh
    except ImportError:
        run_cmd([sys.executable, "-m", "pip", "install", "-q", "trimesh"])
        import trimesh
        return trimesh


def install_depth_deps():
    try:
        from PIL import Image
        from transformers import AutoImageProcessor, AutoModelForDepthEstimation
        return Image, AutoImageProcessor, AutoModelForDepthEstimation
    except Exception:
        run_cmd([
            sys.executable, "-m", "pip", "install", "-q", "--no-cache-dir",
            "transformers>=4.45.0", "huggingface_hub>=0.24.0",
            "safetensors", "accelerate", "timm",
        ])
        from PIL import Image
        from transformers import AutoImageProcessor, AutoModelForDepthEstimation
        return Image, AutoImageProcessor, AutoModelForDepthEstimation


# ---------------------------------------------------------------------------
# MV-DUSt3R+ backbone (reuses kaggle_run1_run2_eval_baseline.py patterns)
# ---------------------------------------------------------------------------

import builtins
import types

import torch
from PIL import Image
from scipy.spatial import cKDTree

REPO_URL = "https://github.com/facebookresearch/mvdust3r.git"


def clone_repo():
    import shutil
    root = Path("/kaggle/temp/mvdust3r")
    if root.exists():
        shutil.rmtree(root)
    run_cmd(["git", "clone", "--depth", "1", REPO_URL, str(root)])
    return root


def install_deps(root):
    req_src = root / "requirements.txt"
    req_kaggle = Path("/kaggle/working/requirements_kaggle.txt")
    skip_prefixes = (
        "torch==", "torchvision==", "tensorflow", "gsplat==",
        "numpy==", "scipy", "opencv-python",
    )
    lines = []
    for line in req_src.read_text().splitlines():
        s = line.strip()
        if not s or s.startswith("#"):
            continue
        if s.startswith(skip_prefixes):
            continue
        lines.append(s)
    req_kaggle.write_text("\n".join(lines) + "\n")
    run_cmd([sys.executable, "-m", "pip", "install", "-q", "-r", str(req_kaggle)])


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


def load_mvdust3r(root, ckpt_path):
    sys.path.insert(0, str(root))
    os.chdir(root)
    builtins.input = lambda prompt="": ""
    install_pytorch3d_shim()
    patch_torch_load()
    from demo import AsymmetricCroCo3DStereoMultiView
    inf = np.inf
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
    ).to("cuda")
    loaded = AsymmetricCroCo3DStereoMultiView.from_pretrained(str(ckpt_path)).to("cuda")
    model.load_state_dict(loaded.state_dict(), strict=True)
    model.eval()
    del loaded
    torch.cuda.empty_cache()
    return model


def run_mvdust3r_inference(model, root, view_files, out_dir):
    """Run MV-DUSt3R+ and return the raw output dict, a GLB path, and the
    candidate arrays (points, colors, conf, pixel coords, view ids)."""
    from demo import get_reconstructed_scene

    out_dir.mkdir(parents=True, exist_ok=True)
    torch.cuda.empty_cache()
    original_n_ref = getattr(model, "n_ref", None)
    model.n_ref = max(2, min(4, len(view_files)))
    print(f"MV-DUSt3R+ inference: {len(view_files)} views, n_ref={model.n_ref}")
    start = time.time()
    try:
        with torch.no_grad():
            output, glb_file, _ = get_reconstructed_scene(
                str(out_dir), model, "cuda", True, IMAGE_SIZE,
                [str(p) for p in view_files], 3.0, True, False, 0.05, 10,
            )
        runtime = time.time() - start
        print(f"MV-DUSt3R+ inference done in {runtime:.1f}s")
        return output, Path(glb_file), runtime
    finally:
        model.n_ref = original_n_ref


# ---------------------------------------------------------------------------
# Candidate extraction from MV-DUSt3R+ output
# ---------------------------------------------------------------------------


def extract_candidates(output):
    """Extract all candidate points, colors, and confidence from the output."""
    pts_list = [output["pred1"]["pts3d"][0].detach().cpu()]
    pts_list += [x["pts3d_in_other_view"][0].detach().cpu() for x in output["pred2s"]]
    conf_list = [output["pred1"]["conf"][0].detach().cpu()]
    conf_list += [x["conf"][0].detach().cpu() for x in output["pred2s"]]

    pts = torch.stack(pts_list, dim=0).numpy()   # [V, H, W, 3]
    conf = torch.stack(conf_list, dim=0).numpy()  # [V, H, W]

    num_views, height, width, _ = pts.shape
    all_points = []
    all_colors = []
    all_xs = []
    all_ys = []
    all_view_ids = []

    for v in range(num_views):
        pts_v = pts[v].reshape(-1, 3)     # [H*W, 3]
        conf_v = conf[v].reshape(-1)       # [H*W]
        ys, xs = np.mgrid[0:height, 0:width]
        xs_flat = xs.reshape(-1).astype(np.float32)
        ys_flat = ys.reshape(-1).astype(np.float32)

        # Filter out non-finite points
        finite = np.isfinite(pts_v).all(axis=1) & np.isfinite(conf_v)
        all_points.append(pts_v[finite])
        all_xs.append(xs_flat[finite])
        all_ys.append(ys_flat[finite])
        all_view_ids.append(np.full(int(finite.sum()), v, dtype=np.int32))

    points = np.concatenate(all_points, axis=0).astype(np.float32)
    xs = np.concatenate(all_xs, axis=0)
    ys = np.concatenate(all_ys, axis=0)
    view_ids = np.concatenate(all_view_ids, axis=0)

    return points, xs, ys, view_ids, height, width


# ---------------------------------------------------------------------------
# Camera geometry helpers
# ---------------------------------------------------------------------------


def parse_pose(path):
    values = [float(x) for x in Path(path).read_text().split()]
    if len(values) == 16:
        return np.array(values, dtype=np.float32).reshape(4, 4)
    raise ValueError(f"Unexpected pose format in {path}: {len(values)} values")


def intrinsics(depth_shape):
    height, width = depth_shape
    return (
        577.870605 * (width / 640.0),
        577.870605 * (height / 480.0),
        319.5 * (width / 640.0),
        239.5 * (height / 480.0),
    )


# ---------------------------------------------------------------------------
# Depth estimation
# ---------------------------------------------------------------------------


class DepthPredictor:
    """Loads the Run 37 fine-tuned Depth Anything V2 checkpoint."""

    def __init__(self, checkpoint_dir, PILImage, AutoImageProcessor,
                 AutoModelForDepthEstimation):
        self.checkpoint_dir = Path(checkpoint_dir)
        self.PILImage = PILImage
        self.device = torch.device("cuda")
        self.processor = AutoImageProcessor.from_pretrained(
            self.checkpoint_dir, local_files_only=True,
        )
        self.model = AutoModelForDepthEstimation.from_pretrained(
            self.checkpoint_dir, local_files_only=True,
        ).to(self.device)
        self.model.eval()

    def predict(self, rgb_path):
        """Predict metric depth from a single RGB image. Returns (H, W) float32."""
        image = self.PILImage.open(rgb_path).convert("RGB")
        inputs = self.processor(images=image, return_tensors="pt")
        pixel_values = inputs["pixel_values"].to(self.device, non_blocking=True)
        with torch.inference_mode():
            predicted = self.model(pixel_values=pixel_values).predicted_depth
        depth = predicted[0].float().cpu().numpy().astype(np.float32)
        return depth

    def predict_and_colorize(self, rgb_path, output_path):
        """Predict depth, save colorized PNG, and return raw depth array."""
        depth = self.predict(rgb_path)
        colorized = colorize_depth_turbo(depth)
        output_path.parent.mkdir(parents=True, exist_ok=True)
        colorized.save(output_path)
        return depth


def colorize_depth_turbo(depth, min_depth=0.10):
    """Colorize a depth map using a turbo-like gradient for visual appeal."""
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.cm as cm

    depth = np.asarray(depth, dtype=np.float32)
    valid = np.isfinite(depth) & (depth > min_depth)
    if not valid.any():
        return Image.fromarray(np.zeros((*depth.shape, 3), dtype=np.uint8))

    lo, hi = np.percentile(depth[valid], [2, 98])
    normalized = np.clip((depth - lo) / max(hi - lo, 1e-6), 0.0, 1.0)
    # Invert so closer = warmer colors
    normalized = 1.0 - normalized
    colored = cm.turbo(normalized)[:, :, :3]  # drop alpha
    colored = (colored * 255).astype(np.uint8)
    # Mark invalid pixels as black
    colored[~valid] = 0
    return Image.fromarray(colored)


# ---------------------------------------------------------------------------
# Depth backprojection (direct depth-only point cloud)
# ---------------------------------------------------------------------------


def backproject_depth_cloud(view_files, depth_maps, stride=4, min_depth_m=0.10):
    """Backproject estimated depth maps into 3D using known poses."""
    poses = [parse_pose(str(p).replace(".jpg", ".txt")) for p in view_files]
    first_pose_inv = np.linalg.inv(poses[0]).astype(np.float32)
    clouds = []
    colors_list = []

    for view_index, (view_file, depth) in enumerate(zip(view_files, depth_maps)):
        depth = np.asarray(depth, dtype=np.float32)
        height, width = depth.shape
        yy, xx = np.mgrid[0:height:stride, 0:width:stride]
        z = depth[yy, xx]
        valid = np.isfinite(z) & (z > min_depth_m)
        if not valid.any():
            continue
        x_px = xx[valid].astype(np.float32)
        y_px = yy[valid].astype(np.float32)
        z_val = z[valid]
        fx, fy, cx, cy = intrinsics(depth.shape)
        camera = np.column_stack([
            (x_px - cx) / fx * z_val,
            (y_px - cy) / fy * z_val,
            z_val,
            np.ones(len(z_val), dtype=np.float32),
        ])
        world = (poses[view_index] @ camera.T).T
        first_camera = (first_pose_inv @ world.T).T[:, :3]
        finite = np.isfinite(first_camera).all(axis=1)
        clouds.append(first_camera[finite].astype(np.float32))

        # Sample colors from the RGB image
        rgb = np.asarray(Image.open(view_file).convert("RGB").resize(
            (width, height), Image.Resampling.BILINEAR
        ), dtype=np.uint8)
        frame_colors = rgb[yy[valid], xx[valid]]
        colors_list.append(frame_colors[finite])

    if not clouds:
        return np.zeros((0, 3), dtype=np.float32), np.zeros((0, 3), dtype=np.uint8)
    return np.concatenate(clouds, axis=0), np.concatenate(colors_list, axis=0)


# ---------------------------------------------------------------------------
# Source-ray correction with estimated depth
# ---------------------------------------------------------------------------


def sample_depth_targets(view_files, depth_maps, xs, ys, view_ids,
                         candidate_height, candidate_width, min_depth_m=0.10):
    """For each MV-DUSt3R+ candidate, compute the corresponding estimated-depth
    3D point by back-projecting the estimated depth along the source ray."""
    poses = [
        parse_pose(str(p).replace(".jpg", ".txt")).astype(np.float32)
        for p in view_files
    ]
    first_pose_inv = np.linalg.inv(poses[0]).astype(np.float32)
    targets = np.zeros((len(xs), 3), dtype=np.float32)
    valid = np.zeros(len(xs), dtype=bool)

    for view_index, depth in enumerate(depth_maps):
        indices = np.where(view_ids == view_index)[0]
        if not len(indices):
            continue
        depth = np.asarray(depth, dtype=np.float32)
        height, width = depth.shape
        depth_x = np.clip(
            np.rint(xs[indices] / max(candidate_width - 1, 1) * max(width - 1, 1)).astype(np.int32),
            0, width - 1,
        )
        depth_y = np.clip(
            np.rint(ys[indices] / max(candidate_height - 1, 1) * max(height - 1, 1)).astype(np.int32),
            0, height - 1,
        )
        z = depth[depth_y, depth_x]
        usable = np.isfinite(z) & (z > min_depth_m)
        if not usable.any():
            continue
        usable_indices = indices[usable]
        z = z[usable]
        x = depth_x[usable].astype(np.float32)
        y = depth_y[usable].astype(np.float32)
        fx, fy, cx, cy = intrinsics(depth.shape)
        camera = np.column_stack([
            (x - cx) / fx * z,
            (y - cy) / fy * z,
            z,
            np.ones(len(z), dtype=np.float32),
        ])
        world = (poses[view_index] @ camera.T).T
        first_camera = (first_pose_inv @ world.T).T
        targets[usable_indices] = first_camera[:, :3]
        valid[usable_indices] = True
    return targets, valid


def apply_correction(points, targets, valid, tau, alpha):
    """Selectively correct candidate points where depth residual exceeds tau."""
    residual = np.linalg.norm(points - targets, axis=1).astype(np.float32)
    mask = valid & (residual >= tau)
    corrected = points.copy()
    corrected[mask] = (1.0 - alpha) * points[mask] + alpha * targets[mask]
    num_corrected = int(mask.sum())
    print(f"Correction: {num_corrected}/{len(points)} points corrected "
          f"({100.0 * num_corrected / max(len(points), 1):.1f}%)")
    return corrected


# ---------------------------------------------------------------------------
# GLB export
# ---------------------------------------------------------------------------


def export_glb(points, colors, output_path):
    """Export a colored point cloud as a GLB file using trimesh."""
    trimesh = install_trimesh()
    output_path = Path(output_path)
    output_path.parent.mkdir(parents=True, exist_ok=True)
    rgba = np.concatenate(
        [colors.astype(np.uint8),
         np.full((len(colors), 1), 255, dtype=np.uint8)],
        axis=1,
    )
    cloud = trimesh.points.PointCloud(vertices=points, colors=rgba)
    cloud.export(output_path)
    print(f"Exported GLB: {output_path} ({len(points)} points)")
    return str(output_path)


def downsample(points, colors, max_points):
    """Randomly downsample if too many points."""
    if len(points) <= max_points:
        return points, colors
    rng = np.random.default_rng(SEED)
    idx = np.sort(rng.choice(len(points), max_points, replace=False))
    return points[idx], colors[idx]


# ---------------------------------------------------------------------------
# View selection
# ---------------------------------------------------------------------------


def select_views_diverse(scene_dir, num_views):
    """Select views with angular diversity from the available frames."""
    jpgs = sorted(scene_dir.glob("*.jpg"))
    if len(jpgs) <= num_views:
        return jpgs

    # Load all poses and compute camera positions
    positions = []
    valid_jpgs = []
    for jpg in jpgs:
        pose_path = str(jpg).replace(".jpg", ".txt")
        if not Path(pose_path).exists():
            continue
        try:
            pose = parse_pose(pose_path)
            if not np.isfinite(pose).all():
                continue
            positions.append(pose[:3, 3])
            valid_jpgs.append(jpg)
        except Exception:
            continue

    if len(valid_jpgs) <= num_views:
        return valid_jpgs[:num_views]

    positions = np.array(positions, dtype=np.float32)

    # Greedy farthest-point sampling for spatial diversity
    selected = [0]  # start with the first frame
    for _ in range(num_views - 1):
        selected_positions = positions[selected]
        dists = np.min(
            np.linalg.norm(
                positions[:, None, :] - selected_positions[None, :, :],
                axis=2,
            ),
            axis=1,
        )
        dists[selected] = -1  # exclude already-selected
        selected.append(int(np.argmax(dists)))

    chosen = sorted([valid_jpgs[i] for i in selected], key=lambda p: p.name)
    print(f"Selected {num_views} diverse views: {[p.name for p in chosen]}")
    return chosen


# ---------------------------------------------------------------------------
# Locate data sources
# ---------------------------------------------------------------------------


def find_posed_images_root():
    candidates = [
        Path("/kaggle/input/scannet-data/scannet/posed_images"),
        Path("/kaggle/input/scannet-data/posed_images"),
        Path("/kaggle/input/datasets/tiantiansyrinx1102/scannet-data/scannet/posed_images"),
    ]
    for p in candidates:
        if p.exists():
            return p
    for p in Path("/kaggle/input").rglob("posed_images"):
        if p.is_dir():
            return p
    raise FileNotFoundError("Cannot locate posed_images in /kaggle/input")


def find_run37_checkpoint(variant="controlled_best"):
    """Find the Run 37 fine-tuned depth checkpoint."""
    matches = sorted(
        Path("/kaggle/input").rglob(
            "run_37_depth_estimator_full_finetune/run_config.json"
        )
    )
    if not matches:
        raise FileNotFoundError(
            "Run 37 output not found. Mount "
            "nguynnminh/mv-dust3r-run-37-depth-full-fine-tune "
            "as a Kaggle kernel source."
        )
    run37_dir = matches[0].parent
    checkpoint_dir = run37_dir / "checkpoints" / variant
    if not (checkpoint_dir / "model.safetensors").exists():
        raise FileNotFoundError(
            f"Run 37 checkpoint incomplete: {checkpoint_dir}"
        )
    return checkpoint_dir


# ---------------------------------------------------------------------------
# Assign dummy colors to MV-DUSt3R+ points (sample from input images)
# ---------------------------------------------------------------------------


def sample_colors_for_candidates(view_files, xs, ys, view_ids,
                                 candidate_height, candidate_width):
    """Sample RGB colors from input images for each candidate point."""
    colors = np.zeros((len(xs), 3), dtype=np.uint8)
    for view_index, view_file in enumerate(view_files):
        indices = np.where(view_ids == view_index)[0]
        if not len(indices):
            continue
        rgb = np.asarray(
            Image.open(view_file).convert("RGB").resize(
                (candidate_width, candidate_height),
                Image.Resampling.BILINEAR,
            ),
            dtype=np.uint8,
        )
        px = np.clip(xs[indices].astype(np.int32), 0, candidate_width - 1)
        py = np.clip(ys[indices].astype(np.int32), 0, candidate_height - 1)
        colors[indices] = rgb[py, px]
    return colors


# ---------------------------------------------------------------------------
# Main
# ---------------------------------------------------------------------------


def main():
    random.seed(SEED)
    np.random.seed(SEED)
    torch.manual_seed(SEED)
    started = time.time()

    out_dir = Path("/kaggle/working/outputs/demo_inference")
    out_dir.mkdir(parents=True, exist_ok=True)

    # --- Locate data ---
    posed_root = find_posed_images_root()
    scene_dir = posed_root / DEMO_SCENE
    if not scene_dir.exists():
        raise FileNotFoundError(f"Scene directory not found: {scene_dir}")

    # --- Select views ---
    if DEMO_FRAMES:
        frame_names = [f.strip() for f in DEMO_FRAMES.split(",") if f.strip()]
        view_files = [scene_dir / name for name in frame_names]
        for vf in view_files:
            if not vf.exists():
                raise FileNotFoundError(f"Specified frame not found: {vf}")
    else:
        view_files = select_views_diverse(scene_dir, DEMO_NUM_VIEWS)

    print(f"\n{'='*60}")
    print(f"DEMO INFERENCE — {DEMO_SCENE}")
    print(f"Views: {[p.name for p in view_files]}")
    print(f"Correction: tau={DEMO_TAU}, alpha={DEMO_ALPHA}")
    print(f"{'='*60}\n")

    # =====================================================================
    # STAGE 1: MV-DUSt3R+ Backbone
    # =====================================================================
    print("\n[Stage 1/4] Setting up MV-DUSt3R+ backbone...")
    root = clone_repo()
    install_deps(root)
    ckpt_path = download_checkpoint(root)
    backbone = load_mvdust3r(root, ckpt_path)

    print("[Stage 1/4] Running MV-DUSt3R+ inference...")
    output, mvdust3r_glb, mvdust3r_runtime = run_mvdust3r_inference(
        backbone, root, view_files, out_dir / "mvdust3r_workspace",
    )
    # The built-in demo GLB is saved; copy it to our output dir
    import shutil
    mvdust3r_output_glb = out_dir / "mvdust3r_only.glb"
    shutil.copy2(mvdust3r_glb, mvdust3r_output_glb)
    print(f"[Stage 1/4] MV-DUSt3R+ GLB saved: {mvdust3r_output_glb}")

    # Extract raw candidates for correction
    points, xs, ys, view_ids, cand_h, cand_w = extract_candidates(output)
    candidate_colors = sample_colors_for_candidates(
        view_files, xs, ys, view_ids, cand_h, cand_w,
    )
    print(f"[Stage 1/4] Extracted {len(points)} candidate points")

    del output
    torch.cuda.empty_cache()

    # =====================================================================
    # STAGE 2: Depth Estimation
    # =====================================================================
    print("\n[Stage 2/4] Loading fine-tuned depth estimator...")
    checkpoint_dir = find_run37_checkpoint()
    PILImage, AutoImageProcessor, AutoModelForDepthEstimation = install_depth_deps()
    depth_predictor = DepthPredictor(
        checkpoint_dir, PILImage, AutoImageProcessor, AutoModelForDepthEstimation,
    )

    depth_map_dir = out_dir / "depth_maps"
    depth_maps = []
    for view_file in view_files:
        depth_png = depth_map_dir / f"depth_{view_file.stem}.png"
        print(f"  Predicting depth for {view_file.name}...")
        depth = depth_predictor.predict_and_colorize(view_file, depth_png)
        depth_maps.append(depth)
    print(f"[Stage 2/4] Saved {len(depth_maps)} depth maps to {depth_map_dir}")

    # Also save a side-by-side comparison (RGB | Depth) for each view
    comparison_dir = out_dir / "comparisons"
    comparison_dir.mkdir(parents=True, exist_ok=True)
    for view_file, depth in zip(view_files, depth_maps):
        rgb_img = Image.open(view_file).convert("RGB")
        depth_colored = colorize_depth_turbo(depth)
        # Resize depth to match RGB dimensions
        depth_colored = depth_colored.resize(rgb_img.size, Image.Resampling.BILINEAR)
        # Create side-by-side
        w, h = rgb_img.size
        canvas = Image.new("RGB", (w * 2, h + 30), "white")
        canvas.paste(rgb_img, (0, 30))
        canvas.paste(depth_colored, (w, 30))
        from PIL import ImageDraw
        draw = ImageDraw.Draw(canvas)
        draw.text((10, 8), "RGB Input", fill=(0, 0, 0))
        draw.text((w + 10, 8), "Estimated Depth", fill=(0, 0, 0))
        canvas.save(comparison_dir / f"comparison_{view_file.stem}.png")
    print(f"[Stage 2/4] Saved RGB-Depth comparisons to {comparison_dir}")

    # =====================================================================
    # STAGE 3: Direct Depth Backprojection
    # =====================================================================
    print("\n[Stage 3/4] Building direct depth backprojection cloud...")
    direct_pts, direct_colors = backproject_depth_cloud(
        view_files, depth_maps, stride=2,
    )
    direct_pts, direct_colors = downsample(direct_pts, direct_colors, DEMO_MAX_POINTS)
    export_glb(direct_pts, direct_colors, out_dir / "depth_backprojection.glb")

    # =====================================================================
    # STAGE 4: Estimated-Depth Correction (Our Method)
    # =====================================================================
    print("\n[Stage 4/4] Applying estimated-depth source-ray correction...")
    targets, valid = sample_depth_targets(
        view_files, depth_maps, xs, ys, view_ids, cand_h, cand_w,
    )
    corrected = apply_correction(points, targets, valid, DEMO_TAU, DEMO_ALPHA)
    corrected_pts, corrected_colors = downsample(
        corrected, candidate_colors, DEMO_MAX_POINTS,
    )
    export_glb(corrected_pts, corrected_colors, out_dir / "corrected_final.glb")

    # =====================================================================
    # Summary
    # =====================================================================
    elapsed = time.time() - started
    summary = {
        "scene": DEMO_SCENE,
        "views": [p.name for p in view_files],
        "num_views": len(view_files),
        "correction_tau": DEMO_TAU,
        "correction_alpha": DEMO_ALPHA,
        "num_candidates": len(points),
        "num_corrected_points": len(corrected_pts),
        "num_direct_depth_points": len(direct_pts),
        "depth_checkpoint": str(checkpoint_dir),
        "outputs": {
            "mvdust3r_only": str(out_dir / "mvdust3r_only.glb"),
            "depth_backprojection": str(out_dir / "depth_backprojection.glb"),
            "corrected_final": str(out_dir / "corrected_final.glb"),
            "depth_maps": str(depth_map_dir),
            "comparisons": str(comparison_dir),
        },
        "total_runtime_seconds": elapsed,
    }
    (out_dir / "demo_config.json").write_text(
        json.dumps(summary, indent=2), encoding="utf-8",
    )

    print(f"\n{'='*60}")
    print("DEMO COMPLETE")
    print(f"{'='*60}")
    print(f"Total runtime: {elapsed:.1f}s")
    print(f"\nOutputs in {out_dir}:")
    print(f"  1. mvdust3r_only.glb        — Raw MV-DUSt3R+ point cloud")
    print(f"  2. depth_backprojection.glb  — Direct estimated-depth cloud")
    print(f"  3. corrected_final.glb       — Corrected final cloud (Ours)")
    print(f"  4. depth_maps/               — Colorized depth predictions")
    print(f"  5. comparisons/              — RGB vs Depth side-by-side")
    print(f"\nOpen GLB files in https://gltf-viewer.donmccurdy.com/ or MeshLab")


if __name__ == "__main__":
    main()
