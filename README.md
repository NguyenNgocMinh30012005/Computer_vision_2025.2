# Sparse-View RGB + Estimated-Depth 3D Reconstruction

This repository contains the Computer Vision 2025.2 project on sparse-view
indoor 3D reconstruction. The validated reference result is Run 30:
MV-DUSt3R+ candidate reconstruction + input RGB-D source depth maps + known
camera poses/intrinsics + source-ray correction. The current target setting
moves that idea back toward RGB-only input by replacing true source depth with
depth estimated by the fine-tuned depth model from Run 37.

```text
sparse posed RGB views
+ known camera intrinsics/extrinsics
+ MV-DUSt3R+ candidate reconstruction
+ fine-tuned estimated source depth from RGB
+ predicted-depth / source-ray correction
-> corrected point cloud reconstruction under sparse views
```

Runs 0-37 are the full experiment history. RGB-only experiments are kept as a
strong baseline and negative-analysis track. Run 30 is the validated RGB-D
reference contribution. Run 31 is a coverage stress test of that frozen method.
Run 32 adds a direct RGB-D backprojection diagnostic to test the boundary
between source-depth correction and depth-only reconstruction.
Run 33 isolates the MV-DUSt3R+ RGB-only baseline by reusing clean Run 30
all-candidate and fixed-confidence outputs without source-depth correction.
Run 34 exports matched qualitative point clouds/dense scenes and reports two
additional normalized geometry metrics.
Runs 35-36 form an experimental predicted-depth generalization track. They
replace true source depth with monocular depth predicted from RGB while keeping
known poses/intrinsics. Run 36 did not pass the reconstruction gates with the
generic depth model.
Run 37 fine-tunes that predicted-depth estimator on the full Kaggle RGB-D frame
pool discovered under `posed_images`, not the earlier 30-scene controlled
sparse-view subset.
The next target setting is to use the Run 37 fine-tuned depth checkpoint as the
depth source for MV-DUSt3R+ candidate correction.

Coverage semantics are explicit in the final Kaggle scripts. When
`scene_limit` is `null`, scene discovery is uncapped over the scenes visible in
the mounted Kaggle `posed_images` dataset. When `RUN30_MAX_EVAL_GROUPS=0` or
`RUN32_MAX_EVAL_GROUPS=0`, metric evaluation uses all discovered validation and
test sparse-view groups instead of treating zero as an empty cap. This is
all-scene discovery and all-group evaluation within the project dataset mount;
it is still not an official full ScanNet or ScanNet++ benchmark claim.

## Project Summary

This is my main computer vision research project. The goal is to make
sparse-view indoor 3D reconstruction more reliable when only a few posed views
are available and scenes contain occlusion, repeated structures, or weak
overlap.

The project first builds a strong MV-DUSt3R+ RGB-only baseline with view
selection and fixed confidence thresholding. Run 11 is the strongest supported
RGB-only baseline, but it is not the final project method. Runs 12-29 test
learned heads, match disambiguation, candidate filtering, source-ray
correction, and monodepth as diagnostics. Those RGB-only extensions do not pass
the required reconstruction-level gates. Run 30 switches the inference
contract to RGB-D/source-depth and passes the overall, occlusion, and ambiguity
gates. Runs 35-37 then move toward the new target contract:

```text
RGB-only sparse input
-> MV-DUSt3R+ candidate point cloud
-> fine-tuned depth estimator predicts source depth from RGB
-> predicted-depth source-ray correction aligns MV candidates
-> corrected reconstruction
```

## Current Target Setting

The intended next setting is RGB-only at reconstruction input time, plus known
camera intrinsics/extrinsics:

- RGB sparse views go into MV-DUSt3R+ to produce candidate point maps.
- The fine-tuned depth estimator predicts a source depth map for each RGB view.
- The predicted depth is back-projected with known camera geometry.
- MV-DUSt3R+ candidates are corrected toward the predicted-depth source ray.

This is not the same as Run 30 because Run 30 uses true source depth at
inference. It is also not fully pose-free because known camera poses and
intrinsics are still used.

## Validated Reference Claim

Under the RGB-D/source-depth inference setting, Run 30 solves the remaining
occlusion and repeated/wrong-depth limits by using input source depth maps with
camera poses/intrinsics for source-ray correction. RGB-only learned extensions
did not pass reconstruction-level gates and are kept as diagnostics/baselines.

Use this exact framing for the validated reference:

```text
RGB-only learned extensions did not pass reconstruction-level gates. After switching the inference contract to RGB-D, Run 30 uses input source depth maps with known camera poses/intrinsics for source-ray correction and passes the overall, occlusion, and ambiguity gates on held-out scenes.
```

Use this exact framing for the new target setting:

```text
The target RGB-only reconstruction pipeline keeps MV-DUSt3R+ as the sparse-view candidate generator, estimates source depth from RGB using the Run 37 fine-tuned depth model, and applies predicted-depth source-ray correction with known camera poses/intrinsics. This setting must be evaluated separately before it can replace the Run 30 RGB-D reference claim.
```

## Final RGB-D Result

Run 30 selected:

- `selected_method`: `rgbd_source_depth_selected`
- `best_baseline_method`: `all_candidates`
- selected internal policy: `rgbd_residual_ge_0.30`
- mode: `residual`
- alpha: `1.0`
- residual threshold: `0.30 m`
- internal mean reconstruction F-score: `0.2327`
- mean correction ratio: `0.4705`
- gate margin: `0.005`
- `pass_all_limits`: `1`

Validation gate:

| Metric | Best baseline | RGB-D selected | Delta |
| --- | ---: | ---: | ---: |
| Overall F-score | 0.1194 | 0.1753 | +0.0559 |
| Occlusion challenging | 0.1064 | 0.2522 | +0.1458 |
| Ambiguity challenging | 0.1623 | 0.3000 | +0.1377 |

Held-out test:

| Metric | All candidates | RGB-D selected | Delta |
| --- | ---: | ---: | ---: |
| Overall F-score | 0.1758 | 0.2764 | +0.1007 |
| Occlusion challenging | 0.2111 | 0.3146 | +0.1035 |
| Ambiguity challenging | 0.1621 | 0.2948 | +0.1327 |

## Evaluation Metrics

The main metric-space comparison uses Accuracy, Completeness, Precision,
Recall, F-score at 5 cm, and Chamfer distance. Run 34 additionally reports two
scale- and translation-invariant metrics:

```text
normalized_distance
dac_at_0_2_normalized
```

Each cloud is independently zero-centered and divided by its RMS radius:

```text
X_hat = (X - mean(X)) / sqrt(mean(||X - mean(X)||^2))
```

`normalized_distance` is the symmetric nearest-neighbor distance:

```text
0.5 * (mean d(P_hat, G_hat) + mean d(G_hat, P_hat))
```

Lower is better. `dac_at_0_2_normalized` is the fraction of normalized
prediction points whose nearest GT point is within `0.2`; higher is better.
These are auxiliary shape-consistency metrics and still use the current
depth-derived proxy GT rather than an official independent ScanNet mesh.

Run 34 verified means over its three matched 3,500-point comparison groups:

| Method | Mean normalized distance | Mean DAc@0.2 normalized |
| --- | ---: | ---: |
| `rgbd_source_depth_selected` | **0.0515** | **0.9841** |
| `direct_rgbd_backprojection` | 0.0926 | 0.8617 |
| `mvdust3r_rgb_only_all_candidates` | 0.1308 | 0.7967 |

These three-group qualitative results do not replace the broader Run 30
validation/test evaluation.

## RGB-only Predicted-Depth Generalization Track

Run 35 evaluates
`depth-anything/Depth-Anything-V2-Metric-Indoor-Small-hf` against the controlled
source RGB-D depth maps. True depth is used only for diagnostic metrics and a
global scale calibrated on train-fit scenes.

Run 36 then evaluates:

- MV-DUSt3R+ RGB-only all candidates and fixed confidence;
- direct predicted-depth backprojection;
- raw predicted-depth correction;
- train-scale-aligned predicted-depth correction;
- Run 30 true RGB-D correction as a reference-only method.

The Run 36 method input contract is:

```text
RGB images + predicted monocular depth + known pose/intrinsics
```

True source depth is forbidden from correction, residual gating, and direct
predicted-depth backprojection.

Completed Run 35 diagnostic:

| Split | AbsRel ↓ | RMSE (m) ↓ | MAE (m) ↓ | delta1 ↑ | Scale-aligned RMSE (m) ↓ | Scale-aligned MAE (m) ↓ |
| --- | ---: | ---: | ---: | ---: | ---: | ---: |
| Validation | 0.2207 | 0.4773 | 0.4115 | 0.7175 | 0.2175 | 0.1088 |
| Test | 0.1861 | 0.4642 | 0.3525 | 0.7736 | 0.2786 | 0.1614 |

The train-fit global median scale is `0.9123`. Scale alignment improves depth
error substantially, so Run 36 evaluates both raw and globally aligned depth.

Completed Run 36 reconstruction result:

| Split | RGB-only all candidates | Predicted correction | Direct predicted depth | Run 30 true RGB-D |
| --- | ---: | ---: | ---: | ---: |
| Validation F-score | 0.1231 | 0.1012 | 0.1747 | 0.1753 |
| Test F-score | 0.1744 | 0.1465 | 0.1894 | 0.2764 |

The validation-selected correction uses global scale, `tau_pred = 0.5`, and
`alpha = 0.25`. It fails the overall, occlusion, and ambiguity gates:
validation deltas versus RGB-only are `-0.0218`, `-0.0421`, and `-0.0310`.
Direct predicted-depth backprojection is a useful diagnostic and beats the
RGB-only baseline overall, but it regresses on the test occlusion subset
(`0.1840` versus `0.2120`) and remains far below Run 30 on test
(`0.1894` versus `0.2764`). Therefore Run 30 remains the validated RGB-D
reference until the fine-tuned estimated-depth correction run passes the same
reconstruction gates.

## Run 37 Full-Dataset Depth Fine-Tuning And Next Correction Run

Run 37 is the next predicted-depth experiment. It fine-tunes
`depth-anything/Depth-Anything-V2-Metric-Indoor-Small-hf` on the full Kaggle
ScanNet-style RGB-D frame pool discovered under `scannet/posed_images`.

The run deliberately separates two checkpoints:

- `checkpoints/controlled_best`: trained only on a scene-level train split,
  selected by validation depth quality, and evaluated on held-out validation
  and test scenes.
- `checkpoints/full_dataset_deployment`: continues training on 100% of the
  discovered scenes/frames. This is useful for deployment, but it is not used
  as unbiased validation/test evidence.

Run 37 reports depth metrics only: AbsRel, RMSE, MAE, delta1/2/3, and
scale-aligned RMSE/MAE. The next reconstruction run should use the fine-tuned
depth checkpoint as the estimated source-depth input for correction:

```text
RGB sparse views
-> MV-DUSt3R+ candidate reconstruction
-> Run 37 fine-tuned depth estimator predicts per-view depth
-> back-project predicted depth with known pose/intrinsics
-> correct MV-DUSt3R+ candidates along source rays
-> evaluate against the same reconstruction gates
```

Until that run passes the gates, Run 30 remains the validated RGB-D reference
and the fine-tuned predicted-depth correction remains the target setting.

## Why We Switched From RGB-only To RGB-D

Runs 22-29 showed that RGB-only filtering/correction was insufficient. Run 30
changes the inference contract to RGB-D and passes the overall, occlusion, and
ambiguity gates.

Main evidence:

- OARH v2 learned the proxy labels well in Run 21, but Run 22 showed a large
  reconstruction recall/completeness collapse.
- Candidate calibration in Run 23 removed much of the domain shift, but still
  lost to fixed confidence on validation.
- RSDH v2 image-only passed its proxy match-validity gate in Run 24, but Runs
  25-26 showed that reconstruction gains were matched by non-learned
  all-candidate retention.
- Run 27 showed that candidate filtering is the wrong operation: learned
  filtering can beat fixed confidence but still loses to full retention.
- Run 28 showed source-depth correction has large oracle headroom.
- Run 29 showed generic RGB-only monodepth does not recover metric/source depth
  well enough.
- Run 30 makes source depth an explicit input and turns that headroom into a
  validated method.

## What Is Not Claimed

This repository does not claim:

- RGB-only reconstruction solved occlusion or repeated/wrong-depth ambiguity.
- OARH/RSDH/RAJAH are final reconstruction modules.
- generic RGB-only monodepth is enough for source-ray correction.
- full MV-DUSt3R+ backbone fine-tuning was proven useful.
- full ScanNet++ or official-benchmark generality from the controlled
  ScanNet-style subset.

Correct limitation framing:

- Limit 1, sparse-view/view-selection instability, was solved by the RGB-only
  baseline pipeline through Run 11.
- Limit 2, occlusion/low-overlap valid geometry, is solved under the Run 30
  RGB-D/source-depth setting.
- Limit 3, repeated/wrong-depth ambiguity, is solved under the Run 30
  RGB-D/source-depth setting.

## Repository Layout

```text
.
+-- docs/
|   +-- experiments/   # run order, Kaggle guide, final result summaries
|   +-- method/        # OARH/RSDH diagnostics and transition notes
|   +-- proposal/      # original proposal sources and team PDF
+-- notebooks/         # notebook-based sanity checks
+-- pdf/               # LaTeX Beamer source and generated main.pdf
+-- scripts/
|   +-- kaggle/        # reproducible staged Kaggle experiments
+-- tools/             # local maintenance helpers
```

Generated Kaggle submission folders, downloaded outputs, local credentials, and
the cloned upstream `mvdust3r/` repository are intentionally ignored. Local
Hugging Face token files such as `HF.json` and `.env` are also ignored; use
Kaggle Secrets (`HF_TOKEN`) for runs that download from the Hub.

## Experiment History

The staged Kaggle scripts live in `scripts/kaggle/`. Runs 0-37 are the complete
history:

- Runs 0-11: baseline, view selection, confidence thresholding, heuristic
  fusion/filtering ablations, final fixed-threshold baseline.
- Runs 12-18: first RGB-only OARH/RSDH diagnostic experiments; no final method
  was selected from this phase.
- Runs 19-20: supervised label cache and hard subset mining.
- Runs 21-23: OARH v2 and reconstruction-candidate calibration, both negative
  at reconstruction level.
- Runs 24-26: image-only RSDH v2, positive proxy but negative reconstruction
  gate.
- Run 27: reconstruction-aware joint acceptance, negative gate against
  all-candidate retention.
- Run 28: source-ray supervised correction, negative RGB-only learned result
  but strong source-depth oracle headroom.
- Run 29: RGB-only monodepth correction, negative gate.
- Run 30: RGB-D source-depth correction, validated reference contribution.
- Run 31: frozen-policy coverage stress test with 12 sparse-view groups per
  scene across all 30 scenes.
- Run 32: direct RGB-D source-depth backprojection without MV-DUSt3R+;
  completed diagnostic baseline and evaluator-circularity check.
- Run 33: MV-DUSt3R+ only RGB baseline extraction from Run 30 all-candidate
  and fixed-confidence rows; no source-depth inference, correction, or direct
  RGB-D backprojection.
- Run 34: matched 3D visualization exports plus normalized distance and
  DAc@0.2 auxiliary comparison metrics.
- Run 35: completed metric monocular-depth quality diagnostic and
  predicted-depth cache; test AbsRel `0.1861`, delta1 `0.7736`.
- Run 36: completed predicted-depth correction diagnostic. The selected
  correction fails all reconstruction gates; direct predicted-depth
  backprojection is promising but does not replace Run 30.
- Run 37: full Kaggle RGB-D frame-pool fine-tuning for the predicted-depth
  estimator, with separate controlled and deployment checkpoints.
- Run 38 target: use the Run 37 fine-tuned depth checkpoint to estimate source
  depth from RGB, then apply predicted-depth correction to MV-DUSt3R+
  candidates. This is the desired RGB-only-input reconstruction setting and
  must be evaluated before it replaces Run 30.

Final script:

```text
scripts/kaggle/kaggle_run30_rgbd_source_depth_correction.py
```

Coverage validation script:

```text
scripts/kaggle/kaggle_run31_rgbd_coverage_stress_test.py
```

Direct RGB-D baseline script:

```text
scripts/kaggle/kaggle_run32_direct_rgbd_backprojection_baseline.py
```

MV-DUSt3R+ only RGB baseline script:

```text
scripts/kaggle/kaggle_run33_mvdust3r_only_rgb_baseline.py
```

Predicted-depth full-data fine-tuning script:

```text
scripts/kaggle/kaggle_run37_depth_estimator_full_finetune.py
```

## MV-DUSt3R+ Only RGB Baseline

Run 33 is a final diagnostic baseline, not a new reconstruction method. It
reuses existing Run 30 rows that are already clean MV-DUSt3R+ RGB-only
outputs:

- `all_candidates` -> `mvdust3r_raw_all_candidates`
- `confidence_fixed_final` -> `mvdust3r_confidence_fixed`

Run 33 explicitly sets source-depth inference, source-depth correction, and
direct RGB-D backprojection flags to false in `run_config.json`. It reruns no
MV-DUSt3R+ inference; it extracts the baseline numbers from Run 30 output so
the comparison uses the same groups, metrics, and hard subsets.

| Split/subset | Best MV-DUSt3R+ RGB-only | Run 30 RGB-D selected | Delta |
| --- | ---: | ---: | ---: |
| Val overall | 0.1194 | 0.1753 | +0.0559 |
| Val occlusion | 0.1064 | 0.2522 | +0.1458 |
| Val ambiguity | 0.1623 | 0.3000 | +0.1377 |
| Test overall | 0.1758 | 0.2764 | +0.1007 |
| Test occlusion | 0.2111 | 0.3146 | +0.1035 |
| Test ambiguity | 0.1621 | 0.2948 | +0.1327 |

Gate outcome: `run30_adds_value_over_mvdust3r_only`. This confirms that the
Run 30 RGB-D source-depth correction adds value over MV-DUSt3R+ candidate
reconstruction alone under the controlled evaluator. It remains the true-depth
reference while the new target setting tests fine-tuned estimated-depth
correction.

## Direct RGB-D Backprojection Baseline

Run 32 directly lifts valid input depth pixels into 3D with camera intrinsics
and poses, attaches source RGB colors, voxel-downsamples to the same 3,500-point
budget, and evaluates the same Run 30 validation/test groups. It does not use
MV-DUSt3R+.

Direct RGB-D backprojection is not source-depth correction. It is a
depth-only/RGB-D baseline. Source-depth correction specifically refers to
correcting MV-DUSt3R+ candidate points using source depth residuals.

Run 32 also records an evaluator warning: the controlled GT cloud is currently
built from the same selected input depth maps. Direct backprojection therefore
shares its depth source with the evaluation target and is not an independent
mesh/laser-scan benchmark.

Run 32 result:

| Split/subset | Direct voxel RGB-D | Run 30 selected | Run 30 - direct |
| --- | ---: | ---: | ---: |
| Val overall | 0.0874 | 0.1753 | +0.0879 |
| Val occlusion | 0.1162 | 0.2522 | +0.1360 |
| Val ambiguity | 0.0998 | 0.3000 | +0.2002 |
| Test overall | 0.1359 | 0.2764 | +0.1405 |
| Test occlusion | 0.1196 | 0.3146 | +0.1951 |
| Test ambiguity | 0.1837 | 0.2948 | +0.1111 |

The pre-registered primary direct method is the voxelized
`direct_rgbd_backprojection`, and the gate outcome is
`run30_adds_value_over_direct`. Run 32 also logs
`direct_rgbd_backprojection_sampled`, which scores very high
(`0.8500` validation overall, `0.8666` test overall) because it shares the same
input-depth source as the controlled GT cloud. That sampled diagnostic is
therefore treated as evaluator-circularity evidence, not as an official
benchmark result.

Key final-result documents:

- `docs/experiments/final_rgbd_result.md`
- `docs/experiments/project_full_report_run30.md`
- `docs/experiments/experiment_results_summary.md`
- `scripts/kaggle/README.md`

## How To Reproduce

1. Read `scripts/kaggle/README.md` and
   `docs/experiments/experiment_run_order.md`.
2. Reproduce the baseline sequence if needed.
3. Use Run 30 as the validated RGB-D/source-depth reference.
4. Use Run 37 outputs as the estimated-depth source for the next correction
   run.
5. Compare the generated Run 30 outputs with
   `docs/experiments/final_rgbd_result.md`.

Run 30 expected output files:

```text
correction_label_summary.csv
policy_selection.csv
metrics.csv
summary.csv
limit_summary.csv
gate_decision.csv
run_config.json
```

`run_config.json` records the coverage audit fields
`max_eval_groups_raw`, `max_eval_groups_resolved`,
`num_discovered_scenes`, `num_total_eval_groups_before_cap`,
`num_eval_groups_after_cap`, `evaluated_scene_count`, and
`evaluated_scene_ids`. These fields make the distinction between all-scene
discovery and all-group metric evaluation explicit.

Latest final Kaggle kernel:

- [Run 30 RGB-D Source-Depth Correction](https://www.kaggle.com/code/nguynnminh/mv-dust3r-run-30-rgbd-source-depth-correction)
- [Run 31 RGB-D Coverage Stress Test](https://www.kaggle.com/code/nguynnminh/mv-dust3r-run-31-rgbd-coverage-stress-test)
- [Run 32 Direct RGB-D Backprojection](https://www.kaggle.com/code/nguynnminh/mv-dust3r-run-32-direct-rgbd-backprojection)
- [Run 33 MV-DUSt3R+ Only RGB Baseline](https://www.kaggle.com/code/nguynnminh/mv-dust3r-run-33-mvdust3r-only-rgb-baseline)
- [Run 34 Qualitative 3D Exports](https://www.kaggle.com/code/nguynnminh/mv-dust3r-run-34-qualitative-3d-exports)

Run 34 also provides `dense_scene_exports/` for each selected group. These
files preserve the camera and input-image geometry from the original
MV-DUSt3R+ scene while replacing the main point cloud with a dense RGB-only,
Run 30 corrected, or direct RGB-D cloud. Use these files when a scene-like
visualization comparable to `00_original_mvdust3r_inference_scene.glb` is
preferred. The 3,500-point files remain the fair quantitative comparison.

### Convert Run 34 Point Clouds To Meshes

The three comparable Run 34 outputs are point clouds by design. To create
colored visualization meshes with Ball Pivoting:

```powershell
uv run --python 3.12 scripts/convert_run34_pointclouds_to_mesh.py
```

The generated files are written to:

```text
downloads/kaggle_run34_qualitative_3d_exports/
  outputs/run_34_qualitative_3d_exports/meshes_ball_pivoting/
```

These meshes are presentation artifacts only. Ball Pivoting connects nearby
samples and preserves unsupported holes; it does not add reconstruction
evidence and does not replace point-cloud evaluation against geometry GT.

## Build The Slides

From `pdf/`, use the required cleanup workflow:

```bash
latexmk -pdf main.tex
latexmk -c main.tex
find . -maxdepth 1 -type f \( \
  -name "*.aux" -o -name "*.log" -o -name "*.out" -o -name "*.toc" -o \
  -name "*.fls" -o -name "*.fdb_latexmk" -o -name "*.synctex.gz" -o \
  -name "*.bbl" -o -name "*.blg" \
\) -delete
```

The generated deck is `pdf/main.pdf`.

On Windows PowerShell, the same workflow is wrapped by:

```powershell
.\tools\build_pdf.ps1
```

## Push Workflow

For local maintenance, `tools/push_to_github.ps1` stages the curated project
files, verifies that LaTeX intermediate files are absent, commits with a
Conventional Commit message, and pushes to `origin main`.
