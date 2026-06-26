"""
Demo Inference Script — Sparse-View RGB 3D Reconstruction (Pose-Free)
=====================================================================

Produces four visual outputs from user-supplied RGB images:

  1. mvdust3r_only.glb        — Raw MV-DUSt3R+ candidate point cloud
  2. depth_backprojection.glb — Direct backprojection from fine-tuned depth
  3. corrected_final.glb      — MV-DUSt3R+ corrected by estimated depth (Ours)
  4. depth_maps/*.png         — Colorized predicted depth for each input view

This version uses MV-DUSt3R+'s internally estimated camera poses and 
intrinsics, making it completely "pose-free" and capable of running on 
custom images without ground-truth camera data.

Requirements
------------
- Kaggle T4 x2 GPU environment
- Directory of JPG images mounted (default: scannet-data)
- Run 37 fine-tuned depth checkpoint mounted
"""

import json
import os
import random
import subprocess
import sys
import time
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
DEMO_DEPTH_CKPT = os.environ.get("DEMO_DEPTH_CKPT", "")
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
# MV-DUSt3R+ backbone
# ---------------------------------------------------------------------------

import builtins
import types

import torch
from PIL import Image

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
    """Run MV-DUSt3R+ and return the raw output dict, a GLB path, and runtime."""
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
# Pose and Intrinsics Extraction from MV-DUSt3R+ output
# ---------------------------------------------------------------------------

def extract_poses_and_intrinsics(output):
    """Extract Camera-to-World poses and intrinsics from MV-DUSt3R+ pointmaps."""
    from dust3r.losses import estimate_focal_knowing_depth, calibrate_camera_pnpransac
    
    with torch.no_grad():
        _, h, w = output['pred1']['rgb'].shape[0:3]
        pts3d = [output['pred1']['pts3d'][0]] + [x['pts3d_in_other_view'][0] for x in output['pred2s']]
        conf = torch.stack([output['pred1']['conf'][0]] + [x['conf'][0] for x in output['pred2s']], 0)
        
        # Estimate focal length from the first view
        conf_first = conf[0].reshape(-1)
        conf_sorted = conf_first.sort()[0]
        conf_thres = conf_sorted[int(conf_first.shape[0] * 0.03)]
        valid_first = (conf_first >= conf_thres).reshape(h, w)
        
        focals = estimate_focal_knowing_depth(pts3d[0][None].cuda(), valid_first[None].cuda()).cpu().item()
        
        intrinsics = torch.eye(3,)
        intrinsics[0, 0] = focals
        intrinsics[1, 1] = focals
        intrinsics[0, 2] = w / 2
        intrinsics[1, 2] = h / 2
        intrinsics = intrinsics.cuda()
        
        y_coords, x_coords = torch.meshgrid(torch.arange(h), torch.arange(w), indexing='ij')
        pixel_coords = torch.stack([x_coords, y_coords], dim=-1).float().cuda()
        
        # Calibrate poses using PnP
        c2ws = []
        conf_thres_global = conf.reshape(-1).sort()[0][int(conf.numel() * 0.03)]
        msk = conf >= conf_thres_global
        
        for (pr_pt, valid) in zip(pts3d, msk):
            c2ws_i = calibrate_camera_pnpransac(
                pr_pt.cuda().flatten(0,1)[None], 
                pixel_coords.flatten(0,1)[None], 
                valid.cuda().flatten(0,1)[None], 
                intrinsics[None]
            )
            c2ws.append(c2ws_i[0])
            
        cams2world = torch.stack(c2ws, dim=0).cpu().numpy()
        intrinsics_np = intrinsics.cpu().numpy()
        
    return cams2world, intrinsics_np, (h, w)


def scale_intrinsics(intrinsics_matrix, mvd_shape, target_shape):
    """Scale intrinsics matrix from MV-DUSt3R resolution to target resolution."""
    mvd_h, mvd_w = mvd_shape
    target_h, target_w = target_shape
    
    fx = intrinsics_matrix[0, 0] * (target_w / mvd_w)
    fy = intrinsics_matrix[1, 1] * (target_h / mvd_h)
    cx = intrinsics_matrix[0, 2] * (target_w / mvd_w)
    cy = intrinsics_matrix[1, 2] * (target_h / mvd_h)
    return fx, fy, cx, cy


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
# Scale Alignment (Arbitrary MV-DUSt3R+ space -> Metric Depth space)
# ---------------------------------------------------------------------------

def compute_scale_alignment(mvd_points, metric_depths, xs, ys, view_ids, 
                            cand_h, cand_w, poses, intrinsics_matrix, min_depth_m=0.10):
    """Compute scale scalar to convert MV-DUSt3R poses to metric scale."""
    scales = []
    
    for view_index, depth in enumerate(metric_depths):
        indices = np.where(view_ids == view_index)[0]
        if not len(indices):
            continue
            
        pts = mvd_points[indices]
        c2w = poses[view_index]
        w2c = np.linalg.inv(c2w)
        
        # Transform MVD points into the LOCAL camera system to get their local Z
        pts_hom = np.column_stack([pts, np.ones(len(pts))])
        pts_cam = (w2c @ pts_hom.T).T[:, :3]
        z_mvd = pts_cam[:, 2]
        
        # Get the corresponding metric depth for these pixels
        depth = np.asarray(depth, dtype=np.float32)
        height, width = depth.shape
        px = np.clip(np.rint(xs[indices] / max(cand_w - 1, 1) * max(width - 1, 1)).astype(np.int32), 0, width - 1)
        py = np.clip(np.rint(ys[indices] / max(cand_h - 1, 1) * max(height - 1, 1)).astype(np.int32), 0, height - 1)
        z_metric = depth[py, px]
        
        valid = (z_mvd > 0) & (z_metric > min_depth_m)
        if valid.sum() > 100:
            scales.extend((z_metric[valid] / z_mvd[valid]).tolist())
            
    if not scales:
        print("Warning: Could not compute scale alignment. Using 1.0.")
        return 1.0
        
    global_scale = float(np.median(scales))
    print(f"Computed Scale Factor (Metric / MVD): {global_scale:.4f}")
    return global_scale


# ---------------------------------------------------------------------------
# Depth estimation
# ---------------------------------------------------------------------------

class DepthPredictor:
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
        image = self.PILImage.open(rgb_path).convert("RGB")
        inputs = self.processor(images=image, return_tensors="pt")
        pixel_values = inputs["pixel_values"].to(self.device, non_blocking=True)
        with torch.inference_mode():
            predicted = self.model(pixel_values=pixel_values).predicted_depth
        depth = predicted[0].float().cpu().numpy().astype(np.float32)
        return depth

    def predict_and_colorize(self, rgb_path, output_path):
        depth = self.predict(rgb_path)
        colorized = colorize_depth_turbo(depth)
        output_path.parent.mkdir(parents=True, exist_ok=True)
        colorized.save(output_path)
        return depth


def colorize_depth_turbo(depth, min_depth=0.10):
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.cm as cm

    depth = np.asarray(depth, dtype=np.float32)
    valid = np.isfinite(depth) & (depth > min_depth)
    if not valid.any():
        return Image.fromarray(np.zeros((*depth.shape, 3), dtype=np.uint8))

    lo, hi = np.percentile(depth[valid], [2, 98])
    normalized = np.clip((depth - lo) / max(hi - lo, 1e-6), 0.0, 1.0)
    normalized = 1.0 - normalized
    colored = cm.turbo(normalized)[:, :, :3]
    colored = (colored * 255).astype(np.uint8)
    colored[~valid] = 0
    return Image.fromarray(colored)


# ---------------------------------------------------------------------------
# Depth backprojection
# ---------------------------------------------------------------------------

def backproject_depth_cloud(view_files, depth_maps, poses, intrinsics_matrix, 
                            mvd_shape, stride=4, min_depth_m=0.10):
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
        
        fx, fy, cx, cy = scale_intrinsics(intrinsics_matrix, mvd_shape, depth.shape)
        
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

        rgb = np.asarray(Image.open(view_file).convert("RGB").resize(
            (width, height), Image.Resampling.BILINEAR
        ), dtype=np.uint8)
        frame_colors = rgb[yy[valid], xx[valid]]
        colors_list.append(frame_colors[finite])

    if not clouds:
        return np.zeros((0, 3), dtype=np.float32), np.zeros((0, 3), dtype=np.uint8)
    return np.concatenate(clouds, axis=0), np.concatenate(colors_list, axis=0)


# ---------------------------------------------------------------------------
# Source-ray correction
# ---------------------------------------------------------------------------

def sample_depth_targets(depth_maps, xs, ys, view_ids, candidate_height, candidate_width, 
                         poses, intrinsics_matrix, min_depth_m=0.10):
    first_pose_inv = np.linalg.inv(poses[0]).astype(np.float32)
    targets = np.zeros((len(xs), 3), dtype=np.float32)
    valid = np.zeros(len(xs), dtype=bool)

    for view_index, depth in enumerate(depth_maps):
        indices = np.where(view_ids == view_index)[0]
        if not len(indices):
            continue
            
        depth = np.asarray(depth, dtype=np.float32)
        height, width = depth.shape
        depth_x = np.clip(np.rint(xs[indices] / max(candidate_width - 1, 1) * max(width - 1, 1)).astype(np.int32), 0, width - 1)
        depth_y = np.clip(np.rint(ys[indices] / max(candidate_height - 1, 1) * max(height - 1, 1)).astype(np.int32), 0, height - 1)
        z = depth[depth_y, depth_x]
        
        usable = np.isfinite(z) & (z > min_depth_m)
        if not usable.any():
            continue
            
        usable_indices = indices[usable]
        z = z[usable]
        x = depth_x[usable].astype(np.float32)
        y = depth_y[usable].astype(np.float32)
        
        fx, fy, cx, cy = scale_intrinsics(intrinsics_matrix, (candidate_height, candidate_width), depth.shape)
        
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
    if len(points) <= max_points:
        return points, colors
    rng = np.random.default_rng(SEED)
    idx = np.sort(rng.choice(len(points), max_points, replace=False))
    return points[idx], colors[idx]


# ---------------------------------------------------------------------------
# View selection
# ---------------------------------------------------------------------------

def select_views_evenly(scene_dir, num_views):
    """Select evenly spaced frames from the directory (no poses needed!)."""
    jpgs = sorted(scene_dir.glob("*.jpg"))
    if not jpgs:
        # Fallback to PNG if testing on custom datasets
        jpgs = sorted(scene_dir.glob("*.png"))
        
    if len(jpgs) <= num_views:
        return jpgs
        
    indices = np.linspace(0, len(jpgs) - 1, num_views, dtype=int)
    chosen = [jpgs[i] for i in indices]
    print(f"Selected {num_views} evenly spaced views: {[p.name for p in chosen]}")
    return chosen


# ---------------------------------------------------------------------------
# Locate data sources
# ---------------------------------------------------------------------------

def find_images_root():
    candidates = [
        Path("/kaggle/input/scannet-data/scannet/posed_images"),
        Path("/kaggle/input/scannet-data/posed_images"),
        Path("/kaggle/input/datasets/tiantiansyrinx1102/scannet-data/scannet/posed_images"),
    ]
    for p in candidates:
        if p.exists():
            return p
            
    # Shallow search to avoid hanging on millions of files
    input_dir = Path("/kaggle/input")
    if input_dir.exists():
        for dataset_dir in input_dir.iterdir():
            if dataset_dir.is_dir():
                # Check up to 2 levels deep
                for p in dataset_dir.glob("**/posed_images"):
                    if p.is_dir():
                        return p
                        
    print("Warning: scannet-data not found. Returning /kaggle/input as root.")
    return Path("/kaggle/input")


def find_run37_checkpoint(variant="controlled_best"):
    if not DEMO_DEPTH_CKPT:
        raise ValueError(
            "ERROR: You must provide the exact path to the depth model checkpoint! "
            "Please set os.environ['DEMO_DEPTH_CKPT'] = '/kaggle/input/your-dataset/checkpoints/controlled_best'"
        )
        
    ckpt_path = Path(DEMO_DEPTH_CKPT)
    if not ckpt_path.exists():
        raise FileNotFoundError(
            f"ERROR: The path you provided in DEMO_DEPTH_CKPT does not exist: {ckpt_path}\n"
            "Double-check your Kaggle /kaggle/input directory!"
        )
        
    return ckpt_path


def sample_colors_for_candidates(view_files, xs, ys, view_ids,
                                 candidate_height, candidate_width):
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

    images_root = find_images_root()
    scene_dir = images_root / DEMO_SCENE
    if not scene_dir.exists():
        # Maybe DEMO_SCENE is an absolute path?
        scene_dir = Path(DEMO_SCENE)
        if not scene_dir.exists():
            raise FileNotFoundError(f"Scene directory not found: {scene_dir}")

    if DEMO_FRAMES:
        frame_names = [f.strip() for f in DEMO_FRAMES.split(",") if f.strip()]
        view_files = [scene_dir / name for name in frame_names]
    else:
        view_files = select_views_evenly(scene_dir, DEMO_NUM_VIEWS)

    print(f"\n{'='*60}")
    print(f"POSE-FREE DEMO INFERENCE — {scene_dir.name}")
    print(f"Views: {[p.name for p in view_files]}")
    print(f"{'='*60}\n")

    # =====================================================================
    # STAGE 1: MV-DUSt3R+ Backbone & Pose Extraction
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
    
    import shutil
    mvdust3r_output_glb = out_dir / "mvdust3r_only.glb"
    shutil.copy2(mvdust3r_glb, mvdust3r_output_glb)
    
    # Extract candidate points and poses
    points, xs, ys, view_ids, cand_h, cand_w = extract_candidates(output)
    poses, intrinsics_matrix, mvd_shape = extract_poses_and_intrinsics(output)
    candidate_colors = sample_colors_for_candidates(view_files, xs, ys, view_ids, cand_h, cand_w)

    del output
    torch.cuda.empty_cache()

    # =====================================================================
    # STAGE 2: Depth Estimation
    # =====================================================================
    print("\n[Stage 2/4] Predicting Metric Depth Maps...")
    
    print("  -> Finding checkpoint...")
    checkpoint_dir = find_run37_checkpoint()
    print(f"  -> Found checkpoint at: {checkpoint_dir}")
    
    print("  -> Installing/Importing depth dependencies...")
    try:
        from PIL import Image
        from transformers import AutoImageProcessor, AutoModelForDepthEstimation
        PILImage = Image
        print("  -> Dependencies imported successfully.")
    except Exception as e:
        print(f"  -> Missing dependencies! Running pip install (WARNING: This will HANG FOREVER if Kaggle Internet is OFF!)...")
        run_cmd([
            sys.executable, "-m", "pip", "install", "-q", "--no-cache-dir",
            "transformers>=4.45.0", "huggingface_hub>=0.24.0",
            "safetensors", "accelerate", "timm",
        ])
        print("  -> Pip install finished.")
        from PIL import Image
        from transformers import AutoImageProcessor, AutoModelForDepthEstimation
        PILImage = Image
    
    print("  -> Initializing DepthPredictor (WARNING: If Kaggle Internet is OFF, timm might hang here!)...")
    depth_predictor = DepthPredictor(checkpoint_dir, PILImage, AutoImageProcessor, AutoModelForDepthEstimation)
    print("  -> DepthPredictor initialized successfully.")

    depth_maps = []
    for view_file in view_files:
        print(f"  -> Predicting depth for {view_file.name}...")
        depth = depth_predictor.predict(view_file)
        depth_maps.append(depth)

    # =====================================================================
    # STAGE 3: Scale Alignment & Direct Depth Backprojection
    # =====================================================================
    print("\n[Stage 3/4] Aligning Metric Scale with MV-DUSt3R+ Scale...")
    scale_factor = compute_scale_alignment(
        points, depth_maps, xs, ys, view_ids, cand_h, cand_w, poses, intrinsics_matrix
    )
    
    # Scale MV-DUSt3R+ point clouds and poses to match Metric space
    points_metric = points * scale_factor
    metric_poses = []
    for pose in poses:
        p = pose.copy()
        p[:3, 3] *= scale_factor
        metric_poses.append(p)
    metric_poses = np.array(metric_poses)

    # Export scaled MV-DUSt3R+ for fair comparison
    points_metric_ds, cand_colors_ds = downsample(points_metric, candidate_colors, DEMO_MAX_POINTS)
    export_glb(points_metric_ds, cand_colors_ds, out_dir / "mvdust3r_only_metric_scaled.glb")

    print("[Stage 3/4] Building direct depth backprojection cloud using predicted poses...")
    direct_pts, direct_colors = backproject_depth_cloud(
        view_files, depth_maps, metric_poses, intrinsics_matrix, mvd_shape, stride=2,
    )
    direct_pts, direct_colors = downsample(direct_pts, direct_colors, DEMO_MAX_POINTS)
    export_glb(direct_pts, direct_colors, out_dir / "depth_backprojection.glb")

    # =====================================================================
    # STAGE 4: Estimated-Depth Correction (Pose-Free)
    # =====================================================================
    print("\n[Stage 4/4] Applying estimated-depth source-ray correction...")
    targets, valid = sample_depth_targets(
        depth_maps, xs, ys, view_ids, cand_h, cand_w, metric_poses, intrinsics_matrix
    )
    corrected = apply_correction(points_metric, targets, valid, DEMO_TAU, DEMO_ALPHA)
    corrected_pts, corrected_colors = downsample(corrected, candidate_colors, DEMO_MAX_POINTS)
    export_glb(corrected_pts, corrected_colors, out_dir / "corrected_final.glb")

    # =====================================================================
    # Summary
    # =====================================================================
    elapsed = time.time() - started
    print(f"\n{'='*60}")
    print("POSE-FREE DEMO COMPLETE")
    print(f"Scale Factor applied: {scale_factor:.4f}")
    print(f"Outputs saved to: {out_dir}")
    print(f"{'='*60}\n")


if __name__ == "__main__":
    main()
