import csv
import json
import os
import random
import shutil
import subprocess
import sys
import time
import urllib.request
from pathlib import Path

import cv2
import numpy as np
from PIL import Image

RAW_BASE = "https://raw.githubusercontent.com/NguyenNgocMinh30012005/Computer_vision_2025.2/main/scripts/kaggle/kaggle_run11_final_validation_3seeds.py"


def ensure_helper_module(module_name, raw_url):
    try:
        return __import__(module_name)
    except ModuleNotFoundError:
        helper_path = Path(f"{module_name}.py")
        print(f"Downloading helper module {module_name} from GitHub raw")
        urllib.request.urlretrieve(raw_url, helper_path)
        sys.path.insert(0, str(Path.cwd()))
        return __import__(module_name)


base = ensure_helper_module("kaggle_run11_final_validation_3seeds", RAW_BASE)


RUN_NAME = "run_15_mast3r_reciprocal_features"
VIEW_COUNTS = [3, 4, 5]
SCENE_INDICES = [0, 1]
MAX_PAIRS_PER_CASE = 4
MAX_MATCHES_PER_PAIR = 2500
POS_DIST_M = 0.05
NEG_DIST_M = 0.30
SEED = 1515
MAST3R_MODEL = "naver/MASt3R_ViTLarge_BaseDecoder_512_catmlpdpt_metric"


def run(cmd, **kwargs):
    print("+", " ".join(map(str, cmd)))
    subprocess.run(cmd, check=True, **kwargs)


def install_filtered_requirements(req_path):
    if not req_path.exists():
        return
    skip_prefixes = (
        "torch",
        "torchvision",
        "torchaudio",
        "tensorflow",
        "opencv-python",
        "numpy",
        "scipy",
    )
    filtered = Path("/kaggle/working") / f"requirements_{req_path.parent.name}_{req_path.name}"
    lines = []
    for line in req_path.read_text().splitlines():
        s = line.strip()
        if not s or s.startswith("#"):
            continue
        if s.startswith(skip_prefixes):
            print("Skip Kaggle-managed/heavy package:", s)
            continue
        lines.append(s)
    if lines:
        filtered.write_text("\n".join(lines) + "\n")
        run([sys.executable, "-m", "pip", "install", "-q", "-r", str(filtered)])


def write_csv_union(path, rows):
    if not rows:
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


def read_depth(path):
    depth = np.array(Image.open(path))
    if depth.ndim == 3:
        depth = depth[..., 0]
    depth = depth.astype(np.float32)
    if depth.max() > 100:
        depth = depth / 1000.0
    return depth


def backproject_pixel(jpg_path, xy, first_pose):
    depth_path = str(jpg_path).replace(".jpg", ".png")
    pose_path = str(jpg_path).replace(".jpg", ".txt")
    depth = read_depth(depth_path)
    h, w = depth.shape
    x = int(np.clip(round(float(xy[0])), 0, w - 1))
    y = int(np.clip(round(float(xy[1])), 0, h - 1))
    z = float(depth[y, x])
    if z <= 0 or not np.isfinite(z):
        return None

    fx = 577.870605 * (w / 640.0)
    fy = 577.870605 * (h / 480.0)
    cx = 319.5 * (w / 640.0)
    cy = 239.5 * (h / 480.0)
    X = (x - cx) / fx * z
    Y = (y - cy) / fy * z
    pt_cam = np.array([X, Y, z, 1.0], dtype=np.float32)
    pose = base.parse_pose(pose_path)
    pt_world = pose @ pt_cam
    pt_first = np.linalg.inv(first_pose) @ pt_world
    return pt_first[:3].astype(np.float32)


def scale_match_xy(xy, match_shape, jpg_path):
    depth = read_depth(str(jpg_path).replace(".jpg", ".png"))
    dh, dw = depth.shape
    mh, mw = match_shape
    return np.array([float(xy[0]) / max(mw - 1, 1) * (dw - 1), float(xy[1]) / max(mh - 1, 1) * (dh - 1)])


def setup_mast3r_backend():
    root = Path("/kaggle/temp/mast3r")
    try:
        base.configure_hf_token()
        if root.exists():
            shutil.rmtree(root)
        run(["git", "clone", "--recursive", "--depth", "1", "https://github.com/naver/mast3r.git", str(root)])
        reqs = [root / "requirements.txt", root / "dust3r" / "requirements.txt"]
        for req in reqs:
            install_filtered_requirements(req)
        sys.path.insert(0, str(root))
        import mast3r.utils.path_to_dust3r  # noqa: F401
        from dust3r.inference import inference
        from dust3r.utils.image import load_images
        from mast3r.fast_nn import fast_reciprocal_NNs
        from mast3r.model import AsymmetricMASt3R

        import torch

        model = AsymmetricMASt3R.from_pretrained(MAST3R_MODEL).to("cuda")
        model.eval()
        print("MASt3R backend ready:", MAST3R_MODEL)
        return {
            "name": "mast3r",
            "model": model,
            "inference": inference,
            "load_images": load_images,
            "fast_reciprocal_NNs": fast_reciprocal_NNs,
            "torch": torch,
        }
    except Exception as exc:
        print("MASt3R backend unavailable; falling back to ORB reciprocal matches:", repr(exc))
        return {"name": "orb_fallback", "error": repr(exc)}


def extract_mast3r_pair(backend, img_a, img_b):
    torch = backend["torch"]
    images = backend["load_images"]([str(img_a), str(img_b)], size=512)
    with torch.no_grad():
        output = backend["inference"]([tuple(images)], backend["model"], "cuda", batch_size=1, verbose=False)
    view1, view2 = output["view1"], output["view2"]
    desc1 = output["pred1"]["desc"].squeeze(0).detach()
    desc2 = output["pred2"]["desc"].squeeze(0).detach()
    m0, m1 = backend["fast_reciprocal_NNs"](
        desc1,
        desc2,
        subsample_or_initxy1=8,
        device="cuda",
        dist="dot",
        block_size=2**13,
    )
    h0, w0 = [int(x) for x in view1["true_shape"][0]]
    h1, w1 = [int(x) for x in view2["true_shape"][0]]
    valid0 = (m0[:, 0] >= 3) & (m0[:, 0] < w0 - 3) & (m0[:, 1] >= 3) & (m0[:, 1] < h0 - 3)
    valid1 = (m1[:, 0] >= 3) & (m1[:, 0] < w1 - 3) & (m1[:, 1] >= 3) & (m1[:, 1] < h1 - 3)
    valid = valid0 & valid1
    m0, m1 = m0[valid], m1[valid]
    if len(m0) > MAX_MATCHES_PER_PAIR:
        idx = np.linspace(0, len(m0) - 1, MAX_MATCHES_PER_PAIR).astype(int)
        m0, m1 = m0[idx], m1[idx]
    d1 = desc1.detach().cpu().numpy()
    d2 = desc2.detach().cpu().numpy()
    rows = []
    for xy0, xy1 in zip(m0, m1):
        x0, y0 = int(round(xy0[0])), int(round(xy0[1]))
        x1, y1 = int(round(xy1[0])), int(round(xy1[1]))
        sim = float(np.dot(d1[y0, x0], d2[y1, x1]))
        rows.append(
            {
                "backend": "mast3r",
                "x0": float(xy0[0]),
                "y0": float(xy0[1]),
                "x1": float(xy1[0]),
                "y1": float(xy1[1]),
                "match_h0": h0,
                "match_w0": w0,
                "match_h1": h1,
                "match_w1": w1,
                "descriptor_similarity": sim,
                "descriptor_margin": 0.0,
                "reciprocal_flag": 1.0,
            }
        )
    return rows


def extract_orb_pair(img_a, img_b):
    im0 = cv2.imread(str(img_a), cv2.IMREAD_GRAYSCALE)
    im1 = cv2.imread(str(img_b), cv2.IMREAD_GRAYSCALE)
    orb = cv2.ORB_create(nfeatures=5000)
    kp0, des0 = orb.detectAndCompute(im0, None)
    kp1, des1 = orb.detectAndCompute(im1, None)
    if des0 is None or des1 is None or not kp0 or not kp1:
        return []
    matcher = cv2.BFMatcher(cv2.NORM_HAMMING, crossCheck=False)
    fwd = matcher.knnMatch(des0, des1, k=2)
    rev = matcher.knnMatch(des1, des0, k=2)
    rev_best = {m[0].queryIdx: m[0].trainIdx for m in rev if len(m) >= 1}
    rows = []
    for pair in fwd:
        if len(pair) < 2:
            continue
        m, n = pair
        if rev_best.get(m.trainIdx) != m.queryIdx:
            continue
        xy0 = kp0[m.queryIdx].pt
        xy1 = kp1[m.trainIdx].pt
        margin = float(n.distance - m.distance)
        rows.append(
            {
                "backend": "orb_fallback",
                "x0": float(xy0[0]),
                "y0": float(xy0[1]),
                "x1": float(xy1[0]),
                "y1": float(xy1[1]),
                "match_h0": int(im0.shape[0]),
                "match_w0": int(im0.shape[1]),
                "match_h1": int(im1.shape[0]),
                "match_w1": int(im1.shape[1]),
                "descriptor_similarity": float(1.0 - m.distance / 256.0),
                "descriptor_margin": margin,
                "reciprocal_flag": 1.0,
            }
        )
    if len(rows) > MAX_MATCHES_PER_PAIR:
        rows = rows[:MAX_MATCHES_PER_PAIR]
    return rows


def annotate_pair_rows(rows, img_a, img_b, first_pose):
    annotated = []
    for row in rows:
        xy0 = scale_match_xy((row["x0"], row["y0"]), (row["match_h0"], row["match_w0"]), img_a)
        xy1 = scale_match_xy((row["x1"], row["y1"]), (row["match_h1"], row["match_w1"]), img_b)
        p0 = backproject_pixel(img_a, xy0, first_pose)
        p1 = backproject_pixel(img_b, xy1, first_pose)
        valid_depth = p0 is not None and p1 is not None
        xyz_dist = float(np.linalg.norm(p0 - p1)) if valid_depth else 1e9
        row = dict(row)
        row.update(
            {
                "depth_valid": float(valid_depth),
                "xyz_distance_m": xyz_dist,
                "match_label": float(valid_depth and xyz_dist <= POS_DIST_M),
                "hard_negative": float(valid_depth and xyz_dist >= NEG_DIST_M),
                "pixel_distance_norm": float(
                    np.linalg.norm(
                        np.array([row["x0"] / max(row["match_w0"], 1), row["y0"] / max(row["match_h0"], 1)])
                        - np.array([row["x1"] / max(row["match_w1"], 1), row["y1"] / max(row["match_h1"], 1)])
                    )
                ),
            }
        )
        annotated.append(row)
    return annotated


def case_pairs(view_files):
    pairs = []
    for i in range(len(view_files)):
        for j in range(i + 1, len(view_files)):
            pairs.append((view_files[i], view_files[j]))
    return pairs[:MAX_PAIRS_PER_CASE]


def collect_rows(posed_root, backend):
    scenes = base.discover_scene_dirs(posed_root)
    rows = []
    summary = []
    for scene_index in SCENE_INDICES:
        scene_dir = scenes[scene_index]
        split = "train_proxy" if scene_index == 0 else "heldout"
        for vc in VIEW_COUNTS:
            policy = base.BEST_POLICY_BY_VIEW_COUNT[vc]
            view_files = base.choose_views(scene_dir, vc, policy, seed=SEED)
            first_pose = base.parse_pose(str(view_files[0]).replace(".jpg", ".txt"))
            before_case = len(rows)
            for pair_index, (img_a, img_b) in enumerate(case_pairs(view_files)):
                if backend["name"] == "mast3r":
                    pair_rows = extract_mast3r_pair(backend, img_a, img_b)
                else:
                    pair_rows = extract_orb_pair(img_a, img_b)
                pair_rows = annotate_pair_rows(pair_rows, img_a, img_b, first_pose)
                for k, row in enumerate(pair_rows):
                    row.update(
                        {
                            "run": RUN_NAME,
                            "split": split,
                            "scene": scene_dir.name,
                            "num_views": vc,
                            "view_policy": policy,
                            "pair_index": pair_index,
                            "frame0": img_a.name,
                            "frame1": img_b.name,
                            "match_index": k,
                        }
                    )
                rows.extend(pair_rows)
                print(
                    "Run 15 pair:",
                    {
                        "split": split,
                        "scene": scene_dir.name,
                        "num_views": vc,
                        "pair": [img_a.name, img_b.name],
                        "matches": len(pair_rows),
                        "backend": backend["name"],
                    },
                )
            case_rows = rows[before_case:]
            summary.append(
                {
                    "run": RUN_NAME,
                    "split": split,
                    "scene": scene_dir.name,
                    "num_views": vc,
                    "view_policy": policy,
                    "backend": backend["name"],
                    "num_matches": len(case_rows),
                    "positive_ratio": float(np.mean([r["match_label"] for r in case_rows])) if case_rows else 0.0,
                    "depth_valid_ratio": float(np.mean([r["depth_valid"] for r in case_rows])) if case_rows else 0.0,
                }
            )
    return rows, summary


def main():
    random.seed(SEED)
    np.random.seed(SEED)
    base.require_t4x2()
    posed_root = base.find_posed_images_root()
    print("POSED_IMAGES:", posed_root)
    started = time.time()
    out_dir = Path("/kaggle/working/outputs") / RUN_NAME
    out_dir.mkdir(parents=True, exist_ok=True)
    backend = setup_mast3r_backend()
    rows, summary = collect_rows(posed_root, backend)
    write_csv_union(out_dir / "match_features.csv", rows)
    write_csv_union(out_dir / "feature_summary.csv", summary)
    (out_dir / "run_config.json").write_text(
        json.dumps(
            {
                "run": RUN_NAME,
                "backend": backend["name"],
                "backend_error": backend.get("error"),
                "mast3r_model": MAST3R_MODEL,
                "view_counts": VIEW_COUNTS,
                "max_pairs_per_case": MAX_PAIRS_PER_CASE,
                "max_matches_per_pair": MAX_MATCHES_PER_PAIR,
                "positive_distance_m": POS_DIST_M,
                "negative_distance_m": NEG_DIST_M,
                "runtime_seconds": time.time() - started,
            },
            indent=2,
        )
    )
    print("Run 15 summary:")
    for row in summary:
        print(row)


if __name__ == "__main__":
    main()
