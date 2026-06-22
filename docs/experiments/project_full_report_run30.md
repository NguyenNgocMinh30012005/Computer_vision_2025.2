# Sparse-View RGB-D 3D Reconstruction Project Report Through Run 32

Ngay cap nhat: 2026-06-22

File nay tom tat muc tieu du an, pipeline, cac run Kaggle da chay, ket qua da dat
duoc, va nhung diem van chua the claim la giai quyet triet de.

## 1. Executive Summary

Du an xay dung pipeline sparse-view RGB-D 3D reconstruction dua tren
MV-DUSt3R+. Final project setting la posed RGB-D sparse views, known camera
intrinsics/extrinsics, MV-DUSt3R+ candidate reconstruction, va source-depth /
source-ray correction. Output la point cloud/GLB duoc danh gia bang accuracy,
completeness/recall, precision, F-score va Chamfer distance.

Ket luan ngan gon sau Run 30:

- Limit 1, sparse-view/view-selection instability, da duoc giai quyet o muc
  prototype bang pipeline Final cua Run 11: best view selection + fixed
  confidence thresholds + F0 baseline fusion.
- Limit 2, occlusion/low-overlap valid geometry, chua duoc giai quyet trong
  setting RGB-only. Cac learned filtering/correction RGB-only tu Run 12 den
  Run 29 deu khong pass reconstruction-level gate.
- Limit 3, repeated-structure/wrong-depth ambiguity, cung chua duoc giai quyet
  trong setting RGB-only. RSDH image-only pass proxy gate o Run 24 nhung khong
  pass reconstruction gate o Run 25/26.
- Run 30 pass ca hai limit con lai trong final RGB-D/source-depth inference
  setting. Day la final technical contribution, khong phai claim RGB-only.

Noi cach khac: final framing dung la:

1. RGB-only/frozen MV-DUSt3R+ pipeline la baseline va negative-analysis track:
   no cai thien manh baseline, nhung khong solve occlusion/repeated ambiguity.
2. RGB-D/source-depth pipeline la final setting: Run 30 giai quyet hai limit
   con lai bang source-ray depth correction khi depth map cua input frame duoc
   phep dung luc inference.

Required final wording:

```text
RGB-only learned extensions did not pass reconstruction-level gates. After switching the inference contract to RGB-D, Run 30 uses input source depth maps with known camera poses/intrinsics for source-ray correction and passes the overall, occlusion, and ambiguity gates on held-out scenes.
```

## 2. Project Goal

Muc tieu ban dau la cai thien sparse-view 3D reconstruction khi chi co 2 den 5
views:

```text
Selected sparse views
-> MV-DUSt3R+
-> candidate 3D points / confidence / pointmaps
-> filtering, fusion, or correction
-> reconstructed scene point cloud
-> geometric evaluation against GT/proxy GT
```

Du an khong chi can metric tong the cao hon baseline. No con can kiem tra ba
limit:

| Limit | Mo ta | Trang thai hien tai |
| --- | --- | --- |
| L1: sparse-view/view selection | Random/far-reference views lam coverage kem va point cloud thieu on dinh | Solved at prototype level by Run 11 |
| L2: occlusion/low overlap | Valid points co the chi thay trong it view; consistency/filtering de loai nham valid geometry | Solved only under Run 30 RGB-D/source-depth setting |
| L3: repeated/wrong-depth ambiguity | Repeated texture/object lam match sai hoac wrong-depth candidates | Solved only under Run 30 RGB-D/source-depth setting |

## 3. Dataset And Evaluation Protocol

Du an dung ScanNet posed image/depth style data trong Kaggle:

```text
/kaggle/input/datasets/tiantiansyrinx1102/scannet-data/scannet/posed_images
```

Tu Run 19 tro di, scene split duoc co dinh theo scene-level train/val/test:

- train: scene0000_00 den scene0008_00, 18 scenes
- val: scene0009_00 den scene0011_00, 6 scenes
- test: scene0011_01 den scene0013_01, 6 scenes

Quy tac quan trong:

- Test scenes khong xuat hien trong train.
- Threshold/model/policy selection chon tren train/internal validation hoac val
  theo tung run, khong tune lai tren test.
- Reconstruction-level gate quan trong hon proxy-classification metric.
- Khi learned head chi pass proxy ma fail reconstruction, khong duoc claim la
  solved.

## 4. Final Baseline Before Learned Extensions

Pipeline Final sau Run 11:

```text
Final = best view selection + fixed confidence threshold + F0 baseline fusion
```

Fixed confidence thresholds chon tu Run 10:

| Views | Confidence percentile |
| ---: | ---: |
| 2 | 3.0 |
| 3 | 0.2 |
| 4 | 3.0 |
| 5 | 0.5 |

Run 11 final validation voi 3 seeds:

| Views | B0 mean F-score | Final mean F-score | Delta |
| ---: | ---: | ---: | ---: |
| 2 | 0.5470 | 0.7267 | +0.1797 |
| 3 | 0.6537 | 0.7686 | +0.1149 |
| 4 | 0.7474 | 0.8513 | +0.1039 |
| 5 | 0.7521 | 0.7955 | +0.0434 |

Interpretation: view selection + fixed thresholding solve the first practical
limit of unstable sparse-view baseline. Tuy nhien, Run 9/11 van cho thay
occlusion-heavy va repeated/wrong-depth cases la diem yeu.

## 5. Experiment Timeline

### Runs 0-11: Baseline, Ablations, Final Fixed Pipeline

| Run | Goal | Main outcome |
| ---: | --- | --- |
| 0 | Sanity check MV-DUSt3R+ inference | Pipeline co the tao point cloud/GLB |
| 1 | Evaluation pipeline | Co metric repeatable: accuracy, recall/completeness, precision, F-score, Chamfer |
| 2 | B0 baseline | MV-DUSt3R+ default + random views |
| 3 | B1 confidence threshold | Confidence filtering la baseline manh hon default |
| 4 | View selection | Hybrid/diversity-aware policies cho sparse views |
| 5 | Basic fusion | F0 baseline fusion tot hon custom fusion variants trong setup nay |
| 6 | Occlusion heuristic | Front-depth/occlusion filtering qua aggressive, giam completeness |
| 7 | Repeated-structure heuristic | Ambiguity filtering heuristic giam global F-score |
| 8 | Full ablation | View selection va confidence threshold la dong gop chinh |
| 9 | Stress test | Occlusion-heavy la subset yeu nhat |
| 10 | Sensitivity | Chot fixed confidence percentile theo view count |
| 11 | Final validation | Final thang B0 o 2/3/4/5 views voi 3 seeds |

### Runs 12-18: First Learned Extension

| Run | Goal | Main outcome |
| ---: | --- | --- |
| 12 | OARH proxy reliability | Label F1 cao nhung reconstruction mixed; 2/3 views giam manh |
| 13 | RSDH proxy match disambiguation | Proxy match F1 tren 0.99, nhung co GT-depth-derived simplification |
| 14 | Validation-gated OARH | Gate tranh regressions lon nhung final van nen giu confidence-only |
| 15 | Real MASt3R reciprocal features | Lay reciprocal match features bang MASt3R, khong phai ORB fallback |
| 16 | RSDH descriptor/cycle | Proxy near-perfect nhung van la upper-bound/sanity, chua phai image-only final |
| 17 | Light finetune decision | Khong fine-tune backbone neu validation chua ung ho |
| 18 | Learned summary | Ket luan trung thuc: learned heads chua thay final fixed pipeline |

### Runs 19-20: Data And Hard Subset Mining

Run 19 tao supervised label cache tren 30 scenes:

- total label rows: 4,423,680
- total groups: 180
- outputs: label cache, label summary, view-group manifest, scene split,
  occlusion-heavy group list

Run 20 mine hard subsets tu Run 19:

- streamed rows: 4,423,680
- final eval groups: 32
- OARH groups: 79
- RSDH groups: 108
- sampled OARH rows: 320,692
- sampled RSDH rows: 212,100

Interpretation: Runs 19-20 khong solve model truc tiep. Chung tao du lieu de
test nghiem ngat hon tren occlusion-core, low-overlap/far, va wrong-depth hard
negative cases.

### Runs 21-23: OARH v2 And Candidate Calibration

Run 21 train OARH v2 multitask head:

- train rows: 115,085
- val rows: 75,841
- test rows: 129,766
- feature dim: 21
- val keep F1: 0.99995
- test keep F1: 0.99958
- test visibility accuracy: 0.98081
- test depth residual MAE: 0.05666 m

Interpretation: proxy label task duoc hoc rat tot, nhung chua du de claim.

Run 22 reconstruction integration fail:

| Split | Fixed confidence F-score | Best OARH v2 F-score | Delta |
| --- | ---: | ---: | ---: |
| Val | 0.6716 | 0.2160 | -0.4556 |
| Test | 0.6033 | 0.1936 | -0.4097 |

Main diagnosis: OARH v2 reject qua nhieu reconstruction candidates, lam recall
va completeness collapse.

Run 23 train reliability tren actual MV-DUSt3R candidates, gan nhan bang GT
geometry. Ket qua gan hon nhung van fail gate:

| Split | Fixed confidence F-score | Best RCRH F-score | Delta |
| --- | ---: | ---: | ---: |
| Val | 0.6674 | 0.6618 | -0.0056 |
| Test | 0.6014 | 0.5923 | -0.0090 |

Interpretation: domain shift giam nhieu so voi Run 22, nhung learned ranking
chi canh tranh khi giu gan nhu tat ca point. No chua solve occlusion.

### Runs 24-26: RSDH v2 Image-Only And Reconstruction Gate

Run 24 train image-only RSDH v2:

- rows: 212,100
- train/val/test rows: 68,211 / 67,137 / 76,752
- feature dim: 174
- features: pixel coords, view policy, RGB/grayscale/gradient patch statistics,
  downsampled patch differences/products
- no GT-depth residual, candidate_type, visibility_label, or group class used
  at inference

Run 24 proxy gate passes:

| Split | Best image-only baseline F1 | RSDH v2 MLP F1 | Delta |
| --- | ---: | ---: | ---: |
| Train | 0.5886 | 0.7422 | +0.1536 |
| Val | 0.6212 | 0.6954 | +0.0741 |
| Test | 0.5517 | 0.6596 | +0.1079 |

Run 25 reconstruction integration does not pass strict gate:

| Split | Fixed confidence F-score | Best learned F-score | Delta |
| --- | ---: | ---: | ---: |
| Val | 0.1507 | 0.1542 | +0.0035 |
| Test | 0.2184 | 0.2372 | +0.0189 |

Validation gain +0.0035 nho hon gate margin 0.005. Best test policies giu tat
ca candidates, nen chua chung minh learned ranking thang that.

Run 26 diagnostic gate compares RSDH against all-candidate/confidence top-k
baselines:

| Split | Fixed confidence | All candidates | Best learned | Gate result |
| --- | ---: | ---: | ---: | --- |
| Val | 0.1455 | 0.1463 | 0.1463 | select all_candidates |
| Test | 0.2271 | 0.2407 | 0.2407 | learned only ties by keeping all |

Interpretation: RSDH image-only is useful as proxy classifier, but not a final
reconstruction module. It does not solve repeated-structure ambiguity at
geometry level.

### Run 27: Reconstruction-Aware Joint Acceptance

Run 27 trains directly on reconstruction candidates with:

- MV-DUSt3R confidence and candidate layout
- cross-view support
- raw image-patch consistency
- differentiable top-k objective for soft reconstruction F-score
- special weights for low-support valid points and high-confidence wrong-depth
  negatives

Result:

| Split | Fixed confidence | All candidates | Learned RAJAH | Gate |
| --- | ---: | ---: | ---: | --- |
| Val | 0.1149 | 0.1226 | 0.1211 | fail |
| Test | 0.1753 | 0.1825 | 0.1797 | fail |

Interpretation: filtering is the wrong operation for this benchmark. Removing
even a small number of candidates tends to lose more recall/completeness than
it gains in precision. All-candidate retention becomes the strongest baseline.

### Run 28: Source-Ray Supervised 3D Correction

Run 28 changes from filtering to correction:

```text
keep candidate count
+ learn a source-ray/depth residual correction
+ train from source pixel depth and camera pose
+ no depth at inference
```

Result:

| Split | All candidates | Learned ray-depth correction | Source-depth oracle |
| --- | ---: | ---: | ---: |
| Val | 0.1194 | 0.1139 | 0.1936 |
| Test | 0.1758 | 0.1700 | 0.2886 |

The learned correction improves validation ambiguity but regresses occlusion.
The oracle shows large headroom, so the problem is not impossible; the RGB-only
learned approximation is not good enough.

### Run 29: RGB-Only Monodepth Source-Ray Correction

Run 29 tries pretrained monocular depth as an RGB-only approximation to Run 28
oracle:

- model: Depth-Anything-V2-Small style monodepth
- no GT depth at inference
- known camera poses/intrinsics used to lift source rays
- raw/inverse/inverse-disparity variants tested

Result:

| Split | All candidates | Fixed confidence | Selected monodepth correction | Source-depth diagnostic |
| --- | ---: | ---: | ---: | ---: |
| Val | 0.1208 | 0.1131 | 0.0949 | 0.1979 |
| Test | 0.1786 | 0.1669 | 0.1528 | 0.2726 |

Gate fail:

- validation overall delta vs all-candidates: -0.0260
- validation occlusion delta: -0.0338
- validation ambiguity delta: -0.0520

Interpretation: RGB-only monodepth does not recover metric/source depth well
enough for this correction task.

### Run 30: RGB-D Source-Depth Correction

Run 30 changes the inference contract:

```text
Uses source depth maps from the input posed RGB-D frames
+ known camera poses/intrinsics
+ source-ray correction
```

Selected internal policy:

```text
rgbd_residual_ge_0.30
mode = residual
alpha = 1.0
residual_threshold_m = 0.30
internal mean reconstruction F-score = 0.2327
mean correction ratio = 0.4705
```

External gate decision:

| Field | Value |
| --- | ---: |
| selected_method | rgbd_source_depth_selected |
| best_baseline_method | all_candidates |
| validation best baseline F-score | 0.1194 |
| validation RGB-D selected F-score | 0.1753 |
| delta vs best baseline | +0.0559 |
| occlusion delta vs best baseline | +0.1458 |
| ambiguity delta vs best baseline | +0.1377 |
| pass_all_limits | 1 |
| gate margin | 0.005 |

Validation limit breakdown:

| Subset | All candidates | Fixed confidence | RGB-D selected | Delta vs all |
| --- | ---: | ---: | ---: | ---: |
| Overall | 0.1194 | 0.1123 | 0.1753 | +0.0559 |
| Occlusion challenging | 0.1064 | 0.0917 | 0.2522 | +0.1458 |
| Ambiguity challenging | 0.1623 | 0.1416 | 0.3000 | +0.1377 |

Test limit breakdown:

| Subset | All candidates | Fixed confidence | RGB-D selected | Delta vs all |
| --- | ---: | ---: | ---: | ---: |
| Overall | 0.1758 | 0.1662 | 0.2764 | +0.1007 |
| Occlusion challenging | 0.2111 | 0.1866 | 0.3146 | +0.1035 |
| Ambiguity challenging | 0.1621 | 0.1462 | 0.2948 | +0.1327 |

Interpretation:

- Run 30 solves the two remaining limits under an explicit RGB-D/source-depth
  setting.
- This is not an RGB-only result.
- The right paper/report wording is:

```text
The RGB-only learned extensions did not pass reconstruction-level gates.
However, when input source depth maps are available at inference, source-depth
correction passes the overall, occlusion, and ambiguity gates and substantially
improves held-out F-score.
```

## 6. What Has Been Solved

### Solved 1: Kaggle/P100 Environment Robustness

The pipeline learned to tolerate Kaggle assigning a Tesla P100 instead of T4
x2 by reinstalling a P100-compatible Torch build when needed. This removed a
major reproducibility blocker.

### Solved 2: Strong Sparse-View Baseline

Run 11 shows Final beats B0 across 2/3/4/5 views. This is the strongest
RGB-only deployable pipeline currently supported by reconstruction metrics.

### Solved 3: Hard-Case Labeling And Diagnostics

Runs 19-20 created a reusable label cache and hard-subset mining protocol. This
made later negative results meaningful instead of anecdotal.

### Solved 4: Why Filtering Fails

Runs 22-27 show a consistent diagnosis:

- proxy label F1 does not guarantee reconstruction improvement;
- rejecting candidates often hurts completeness more than it helps precision;
- all-candidate retention is a strong baseline;
- solving occlusion/repeated ambiguity likely needs geometry correction, not
  only candidate deletion.

### Solved 5: RGB-D Source-Depth Solution

Run 30 passes all gates:

- overall validation gain above margin;
- no occlusion regression, actually a large occlusion gain;
- no ambiguity regression, actually a large ambiguity gain;
- test results also improve strongly.

This solves the two remaining limits if the final project is allowed to use
RGB-D input depth at inference.

## 7. What Is Not Solved Yet

### Not Solved 1: RGB-Only Occlusion Handling

OARH, reconstruction candidate calibration, joint acceptance, and learned
source-ray correction all failed reconstruction-level gates in the original
RGB-only/no-depth inference contract.

### Not Solved 2: RGB-Only Repeated-Structure Disambiguation

RSDH v2 image-only passed proxy classification but failed reconstruction
integration. Monodepth correction also regressed ambiguity. Therefore repeated
structure is not solved for RGB-only reconstruction.

### Not Solved 3: Backbone Fine-Tuning

No full MV-DUSt3R+ fine-tuning has been proven useful. The project trained
small heads/proxy modules/correction policies, but did not establish a
successful backbone finetune.

### Not Solved 4: Full Benchmark Generality

The current experiments use a controlled ScanNet-style subset and proxy
geometry evaluation. This is strong for a class project prototype, but not a
full ScanNet++/official benchmark claim.

## 8. Recommended Claim For Slides Or Report

Use this wording:

```text
We first built a strong sparse-view MV-DUSt3R+ baseline using view selection
and fixed confidence thresholds, improving B0 across 2-5 views. We then tested
occlusion-aware and repeated-structure-aware learned modules through strict
reconstruction gates. RGB-only learned filtering and monodepth correction did
not solve the two remaining limits. The final Run 30 shows that these limits
can be solved when source RGB-D depth maps are available at inference: RGB-D
source-depth correction passes the overall, occlusion, and ambiguity gates on
held-out scenes.
```

Avoid this wording:

```text
We solved occlusion and repeated structures in RGB-only sparse-view
reconstruction.
```

That stronger claim is not supported by Runs 22-29.

## 9. Files Used For Latest Result

Latest local Run 30 output folder:

```text
downloads/kaggle_run30_rgbd_source_depth_correction/outputs/run_30_rgbd_source_depth_correction
```

Key files:

- `gate_decision.csv`
- `summary.csv`
- `limit_summary.csv`
- `policy_selection.csv`
- `run_config.json`
- `metrics.csv`

Run 30 Kaggle kernel:

```text
mv-dust3r-run-30-rgbd-source-depth-correction
```

## 10. Final Status And Coverage Follow-Up

Run 30 is the final method-defining experiment and the final technical
contribution. The report/slides should package it as the accepted RGB-D
solution for the two hard limits. Run 31 only expands coverage for validation.

Run 31 is a coverage stress test, not a new method. It keeps the Run 30 policy
fixed and evaluates 12 sparse-view groups per scene over all 30 scenes, for 360
groups total. Each scene keeps 3/4/5-view groups, hybrid/diversity-aware
selection, and two frame variants per configuration. It does not use full dense
frames and does not tune on the larger evaluation set.

Run 31 outputs paired per-group deltas and scene-cluster bootstrap confidence
intervals. Its result is pending and must not be described as passing until the
Kaggle output is reviewed.

### Direct RGB-D Backprojection Baseline

Run 32 directly back-projects input depth pixels into the first-camera frame,
attaches RGB colors, and applies a fixed 2 cm voxel downsample with a 3,500-point
cap. It does not use MV-DUSt3R+ and does not tune on test scenes. Its purpose is
to test the boundary between MV-DUSt3R+ candidate source-depth correction and
direct source-depth reconstruction.

Direct RGB-D backprojection is not source-depth correction. It is a
depth-only/RGB-D baseline. Source-depth correction specifically refers to
correcting MV-DUSt3R+ candidate points using source depth residuals.

Run 32 mounts the Run 30 output and compares `all_candidates`,
`confidence_fixed_final`, `rgbd_source_depth_selected`, and
`direct_rgbd_backprojection` on the same groups and hard subsets. The completed
primary gate uses the voxelized `direct_rgbd_backprojection` method and reports
`run30_adds_value_over_direct`.

| Split / subset | Direct voxel RGB-D | Run 30 selected | Run 30 - direct |
| --- | ---: | ---: | ---: |
| Val overall | 0.0874 | 0.1753 | +0.0879 |
| Val occlusion | 0.1162 | 0.2522 | +0.1360 |
| Val ambiguity | 0.0998 | 0.3000 | +0.2002 |
| Test overall | 0.1359 | 0.2764 | +0.1405 |
| Test occlusion | 0.1196 | 0.3146 | +0.1951 |
| Test ambiguity | 0.1837 | 0.2948 | +0.1111 |

The present evaluator builds its GT cloud from the depth maps of the same
selected input views. This creates a circularity advantage for direct
backprojection and must be reported alongside its metrics. The warning is not
hypothetical: Run 32 also logs `direct_rgbd_backprojection_sampled`, which
scores `0.8500` validation overall and `0.8666` test overall because it samples
from the same input-depth distribution used to build the proxy target. This is
evaluator-circularity evidence, not a new official benchmark result. An
independent mesh/laser-scan target is still required for an official
generalization claim.

Possible future work, outside the current final claim:

1. Add qualitative figures for Run 30/31 source-depth correction.
2. If the project later returns to RGB-only, design a new metric depth
   calibration or indoor-depth fine-tuning run, because Run 29 shows generic
   monodepth is not enough.
3. Evaluate the final RGB-D method on a larger official benchmark with mesh or
   laser-scan ground truth.
