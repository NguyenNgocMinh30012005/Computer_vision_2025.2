# Kaggle Experiment Guide

File nay huong dan dung Kaggle de chay cac thi nghiem trong
`docs/experiments/experiment_run_order.md`.

## Final status

Run 30 is the final recommended Kaggle run:

```text
mv-dust3r-run-30-rgbd-source-depth-correction
```

It uses sparse posed RGB-D views, known intrinsics/extrinsics, MV-DUSt3R+
candidate reconstruction, and source-depth/source-ray correction. Runs 22-29
showed that RGB-only filtering/correction was insufficient; Run 30 changes the
inference contract to RGB-D and passes the overall, occlusion, and ambiguity
gates.

Run 31 is a coverage-only follow-up:

```text
mv-dust3r-run-31-rgbd-coverage-stress-test
```

It freezes the Run 30 policy and evaluates more sparse-view groups. It must not
be used to retune the method.

Run 32 is the direct source-depth diagnostic:

```text
mv-dust3r-run-32-direct-rgbd-backprojection
```

It does not use MV-DUSt3R+ and mounts Run 30 outputs for an exact comparison.
The completed primary voxel baseline is below Run 30, but the sampled direct
diagnostic exposes circularity in the controlled depth-derived proxy target.

Run 33 is the MV-DUSt3R+ only RGB baseline diagnostic:

```text
mv-dust3r-run-33-mvdust3r-only-rgb-baseline
```

It reuses Run 30 `all_candidates` and `confidence_fixed_final` rows instead of
rerunning MV-DUSt3R+ inference. Source-depth inference, correction, and direct
RGB-D backprojection are all disabled in the run config.

## 1. Trang thai setup local

Credential Kaggle da duoc copy vao:

```text
%USERPROFILE%\.kaggle\kaggle.json
```

Trong repo, `kaggle.json` da duoc ignore boi `.gitignore`. Khong commit file nay.

Lenh Kaggle CLI dung duoc trong PowerShell:

```powershell
python -m kaggle.cli --version
python -m kaggle.cli datasets list -s "scannet" --max-size 1000
```

Neu muon dung lenh ngan `kaggle ...`, can them thu muc Scripts cua Python vao `PATH`. Tuy nhien trong repo nay cu dung `python -m kaggle.cli ...` la on dinh.

## 2. Chon cach chay tren Kaggle

Khuyen dung Kaggle Notebook GPU cho cac run dau:

- Run 0: sanity check.
- Run 1: evaluation pipeline.
- Run 2: baseline B0 tren 1-2 scene truoc.

Sau khi pipeline on dinh moi chay cac run lon hon:

- Run 3: confidence threshold.
- Run 4: view selection.
- Run 5-7: fusion/occlusion/ambiguity ablation.
- Run 8-9: full pipeline va final stress test.

Khuyen dung accelerator `GPU T4 x2`. Kaggle CLI/API hien chi expose truong:

```json
"enable_gpu": true
```

Voi Kaggle CLI cu (`kaggle` package 1.x), truong nay khong pin duoc cu the T4x2. Can dung Kaggle CLI moi tren Python 3.11+ va push voi:

```powershell
python -m kaggle kernels push -p kaggle_run0_submission --accelerator NvidiaTeslaT4
```

Notebook `notebooks/kaggle_run0_mvdust3r_sanity.ipynb` co cell kiem tra som:

```text
CUDA device count >= 2 va ten GPU co T4
```

Neu Kaggle cap P100 hoac GPU don, notebook se dung som de tranh chay sai moi truong.

Run thanh cong da xac nhan:

```text
CUDA device count: 2
GPU 0: Tesla T4
GPU 1: Tesla T4
```

## 3. Cau truc input tren Kaggle

Voi Run 0, notebook co the clone source code truc tiep tu GitHub:

```text
https://github.com/facebookresearch/mvdust3r.git
```

Dataset trong anh cua ban:

```text
tiantiansyrinx1102/scannet-data
```

Khi Add Input trong Kaggle Notebook, dataset nay thuong duoc mount thanh:

```text
/kaggle/input/scannet-data/scannet/posed_images/
```

Dataset nay phu hop de chay sanity check/baseline anh dau vao. Tuy nhien, de lam Run 1 evaluation geometry dung nghia, van can ground-truth mesh/point cloud/depth tu split co geometry GT.

Neu ve sau can chay day du hon, nen tao hoac attach cac dataset rieng:

```text
/kaggle/input/
  mvdust3r-checkpoints/       # pretrained checkpoint
  scannet-data/                # dataset tiantiansyrinx1102/scannet-data
  scannetpp-subset/           # subset scenes co geometry GT
```

Output nen ghi vao:

```text
/kaggle/working/outputs/
```

Kaggle se cho download cac file trong `/kaggle/working` sau khi notebook chay xong.

## 4. Cell setup notebook de xuat

Notebook mau da co trong repo:

```text
notebooks/kaggle_run0_mvdust3r_sanity.ipynb
```

Notebook nay thuc hien:

- Clone `https://github.com/facebookresearch/mvdust3r.git`.
- Cai dependencies, giu lai PyTorch/TorchVision san co cua Kaggle.
- Khong downgrade `numpy`, `scipy`, `opencv`, vi co the lam hong binary wheels san co trong Kaggle image.
- Shim PyTorch3D import cho Run 0, vi Run 0 khong dung metric/loss PyTorch3D.
- Patch `torch.load(weights_only=False)` cho checkpoint chinh thuc MV-DUSt3R+ tren PyTorch 2.6+.
- Download checkpoint `MVDp_s2.pth` tu Hugging Face.
- Lay anh tu `tiantiansyrinx1102/scannet-data`.
- Xuat `/kaggle/working/outputs/run_00_sanity_check/scene.glb`.

Cell 1 - Kiem tra GPU:

```python
import os
import torch

print("CUDA available:", torch.cuda.is_available())
if torch.cuda.is_available():
    print("GPU:", torch.cuda.get_device_name(0))
```

Cell 2 - Copy code vao working directory:

```python
import shutil
from pathlib import Path

src = Path("/kaggle/input/mvdust3r-code/mvdust3r")
dst = Path("/kaggle/working/mvdust3r")

if dst.exists():
    shutil.rmtree(dst)
shutil.copytree(src, dst)

print("Code copied to", dst)
```

Cell 3 - Install dependencies:

```python
%cd /kaggle/working/mvdust3r
%pip install -r requirements.txt
```

Neu dependency nao loi do version CUDA/PyTorch, sua trong notebook truoc, sau do ghi lai vao log run.

Cell 4 - Khai bao path chuan:

```python
from pathlib import Path

ROOT = Path("/kaggle/working/mvdust3r")
DATA_ROOT = Path("/kaggle/input/scannetpp-subset")
CKPT_ROOT = Path("/kaggle/input/mvdust3r-checkpoints")
OUT_ROOT = Path("/kaggle/working/outputs")

OUT_ROOT.mkdir(parents=True, exist_ok=True)

print("DATA_ROOT:", DATA_ROOT)
print("CKPT_ROOT:", CKPT_ROOT)
print("OUT_ROOT:", OUT_ROOT)
```

## 5. Run 0 - Sanity check tren Kaggle

Muc tieu:

```text
input images -> MV-DUSt3R+ -> point cloud -> visualize duoc
```

Chay tren 1-2 scene nho. Chua can metric.

Output folder:

```text
/kaggle/working/outputs/run_00_sanity_check/
```

Can luu:

- input image list.
- checkpoint name.
- predicted point cloud.
- visualization screenshot neu co.
- runtime tham khao.

## 6. Run 1 - Evaluation pipeline tren Kaggle

Truoc khi them module, can co script do:

```text
pred_point_cloud vs gt_point_cloud
```

Metrics:

- accuracy.
- completeness / recall.
- precision.
- F-score.
- Chamfer distance neu kip.
- runtime per scene.
- number of retained points.

Khong dung `nvs_test` cua ScanNet++ de danh gia reconstruction geometry neu split do khong co 3D ground truth.

Output folder:

```text
/kaggle/working/outputs/run_01_evaluation_pipeline/
```

## 7. Run 2-33 mapping

Dung dung thu tu trong `docs/experiments/experiment_run_order.md`. Runs 2-11
create the strong RGB-only baseline; Runs 12-29 are diagnostic learned/RGB-only
experiments; Run 30 is the accepted final RGB-D result.

| Run | Ten | Output folder |
| --- | --- | --- |
| 0 | Sanity check | `run_00_sanity_check` |
| 1 | Evaluation pipeline | `run_01_evaluation_pipeline` |
| 2 | Baseline B0 | `run_02_baseline_b0` |
| 3 | Baseline B1 confidence | `run_03_baseline_b1_confidence` |
| 4 | View selection | `run_04_view_selection` |
| 5 | Basic fusion | `run_05_basic_fusion` |
| 6 | Occlusion-aware fusion | `run_06_occlusion_fusion` |
| 7 | Repeated-structure filtering | `run_07_repeated_structure` |
| 8 | Full pipeline | `run_08_full_pipeline` |
| 9 | Final stress test | `run_09_final_stress_test` |
| 10 | Sensitivity visualization | `run_10_sensitivity_visualization` |
| 11 | Final RGB-only baseline validation | `run_11_final_validation_3seeds` |
| 12-29 | RGB-only learned diagnostics | see `scripts/kaggle/README.md` |
| 30 | Final RGB-D source-depth correction | `run_30_rgbd_source_depth_correction` |
| 31 | Frozen-policy RGB-D coverage stress test | `run_31_rgbd_coverage_stress_test` |
| 32 | Direct RGB-D backprojection baseline | `run_32_direct_rgbd_backprojection_baseline` |
| 33 | MV-DUSt3R+ only RGB baseline | `run_33_mvdust3r_only_rgb_baseline` |

Moi run nen sinh it nhat:

```text
metrics.csv
run_config.yaml
runtime_log.txt
selected_views.json
```

Neu co visualization:

```text
figures/
point_clouds/
```

## 8. Download output tu Kaggle ve local

Neu output duoc luu thanh Kaggle dataset, tai ve bang:

```powershell
python -m kaggle.cli datasets download -d <owner>/<dataset-slug> -p downloads/kaggle_outputs --unzip
```

Neu output nam trong notebook session, download truc tiep tu tab Output cua Kaggle Notebook.

## 9. Final Run 30 Output

Run 30 should produce:

```text
correction_label_summary.csv
policy_selection.csv
metrics.csv
summary.csv
limit_summary.csv
gate_decision.csv
run_config.json
```

The current accepted method claim remains the Run 30 RGB-D/source-depth result.
Run 31 is the supported coverage-only follow-up and must not retune that method.

## 10. Run 31 Coverage Output

Run 31 should produce:

```text
group_manifest.csv
coverage_summary.csv
correction_label_summary.csv
metrics.csv
summary.csv
limit_summary.csv
paired_group_deltas.csv
stability_summary.csv
view_count_stability.csv
gate_decision.csv
run_config.json
```

The expected default coverage is 360 groups over 30 scenes. Review
`gate_decision.csv` and `stability_summary.csv` before making any stability
claim.

## 11. Run 32 Direct RGB-D Baseline Output

Run 32 should produce:

```text
metrics.csv
summary.csv
limit_summary.csv
gate_decision.csv
qualitative_manifest.csv
run_config.json
```

Review the circularity warning in `run_config.json`: current GT clouds and the
direct baseline use the same selected input depth maps. Do not convert a strong
Run 32 number into an official benchmark claim.

Completed Run 32 result:

| Split / subset | Direct voxel RGB-D | Run 30 selected | Run 30 - direct |
| --- | ---: | ---: | ---: |
| Val overall | 0.0874 | 0.1753 | +0.0879 |
| Val occlusion | 0.1162 | 0.2522 | +0.1360 |
| Val ambiguity | 0.0998 | 0.3000 | +0.2002 |
| Test overall | 0.1359 | 0.2764 | +0.1405 |
| Test occlusion | 0.1196 | 0.3146 | +0.1951 |
| Test ambiguity | 0.1837 | 0.2948 | +0.1111 |

The auxiliary `direct_rgbd_backprojection_sampled` diagnostic reaches `0.8500`
validation overall and `0.8666` test overall because it samples from nearly the
same depth distribution used to build the proxy target. Treat that as a warning
that independent mesh/laser-scan GT is required for a fair direct-depth
benchmark.

## 12. Run 33 MV-DUSt3R+ Only RGB Baseline Output

Run 33 should produce:

```text
metrics.csv
summary.csv
limit_summary.csv
gate_decision.csv
run_config.json
qualitative_manifest.csv
```

Completed Run 33 result:

| Split / subset | MV-DUSt3R+ RGB-only | Run 30 selected | Run 30 - RGB-only |
| --- | ---: | ---: | ---: |
| Val overall | 0.1194 | 0.1753 | +0.0559 |
| Val occlusion | 0.1064 | 0.2522 | +0.1458 |
| Val ambiguity | 0.1623 | 0.3000 | +0.1377 |
| Test overall | 0.1758 | 0.2764 | +0.1007 |
| Test occlusion | 0.2111 | 0.3146 | +0.1035 |
| Test ambiguity | 0.1621 | 0.2948 | +0.1327 |

Review `run_config.json` for the source-depth flags. They must remain false:
Run 33 is selected sparse RGB views only, with MV-DUSt3R+ confidence and no
source-depth correction. The gate outcome is
`run30_adds_value_over_mvdust3r_only`; the final claim remains Run 30 RGB-D,
not RGB-only.
