# Sparse-View RGB-D 3D Reconstruction

This repository contains the Computer Vision 2025.2 project on sparse-view
indoor 3D reconstruction. The final technical contribution is Run 30:
MV-DUSt3R+ candidate reconstruction + input RGB-D source depth maps + known
camera poses/intrinsics + source-ray correction.

```text
sparse posed RGB-D views
+ known camera intrinsics/extrinsics
+ MV-DUSt3R+ candidate reconstruction
+ source-depth / source-ray correction
-> improved point cloud reconstruction under sparse views, occlusion, and
   repeated/wrong-depth ambiguity
```

Runs 0-32 are the full experiment history. RGB-only experiments are kept as a
strong baseline and negative-analysis track, not as the final solved setting.
Run 30 is the final technical contribution. Run 31 is a coverage stress test of
that frozen method. Run 32 adds a direct RGB-D backprojection baseline to test
whether Run 30 adds value beyond source depth alone.

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
gates.

## Final Supported Claim

Under the RGB-D/source-depth inference setting, Run 30 solves the remaining
occlusion and repeated/wrong-depth limits by using input source depth maps with
camera poses/intrinsics for source-ray correction. RGB-only learned extensions
did not pass reconstruction-level gates and are kept as diagnostics/baselines.

Use this exact framing in reports and slides:

```text
RGB-only learned extensions did not pass reconstruction-level gates. After switching the inference contract to RGB-D, Run 30 uses input source depth maps with known camera poses/intrinsics for source-ray correction and passes the overall, occlusion, and ambiguity gates on held-out scenes.
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

The staged Kaggle scripts live in `scripts/kaggle/`. Runs 0-31 are the complete
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
- Run 30: RGB-D source-depth correction, final accepted contribution.
- Run 31: frozen-policy coverage stress test with 12 sparse-view groups per
  scene across all 30 scenes.
- Run 32: direct RGB-D source-depth backprojection without MV-DUSt3R+;
  comparison against Run 30 is pending.

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
mesh/laser-scan benchmark. No claim that Run 30 beats direct RGB-D
backprojection is made before Run 32 results are reviewed.

Key final-result documents:

- `docs/experiments/final_rgbd_result.md`
- `docs/experiments/project_full_report_run30.md`
- `docs/experiments/experiment_results_summary.md`
- `scripts/kaggle/README.md`

## How To Reproduce

1. Read `scripts/kaggle/README.md` and
   `docs/experiments/experiment_run_order.md`.
2. Reproduce the baseline sequence if needed.
3. Use Run 30 as the final RGB-D/source-depth method.
4. Compare the generated Run 30 outputs with
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

Latest final Kaggle kernel:

- [Run 30 RGB-D Source-Depth Correction](https://www.kaggle.com/code/nguynnminh/mv-dust3r-run-30-rgbd-source-depth-correction)
- [Run 31 RGB-D Coverage Stress Test](https://www.kaggle.com/code/nguynnminh/mv-dust3r-run-31-rgbd-coverage-stress-test)
- [Run 32 Direct RGB-D Backprojection](https://www.kaggle.com/code/nguynnminh/mv-dust3r-run-32-direct-rgbd-backprojection)

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
