# Sparse-View RGB-D 3D Reconstruction with MV-DUSt3R+ and Source-Depth Correction

This repository contains the Computer Vision 2025.2 project on sparse-view
indoor 3D reconstruction. The final project setting is now RGB-D/source-depth:

```text
sparse posed RGB-D views
+ known camera intrinsics/extrinsics
+ MV-DUSt3R+ candidate reconstruction
+ source-depth / source-ray correction
-> improved point cloud reconstruction under sparse views, occlusion, and
   repeated/wrong-depth ambiguity
```

Runs 0-30 are the full experiment history. RGB-only experiments are kept as a
strong baseline and negative-analysis track, not as the final solved setting.
Run 30 is the final technical contribution.

## Portfolio Summary

This is my main computer vision research project. The goal is to make
sparse-view indoor 3D reconstruction more reliable when only a few posed views
are available and scenes contain occlusion, repeated structures, or weak
overlap.

The project first builds a strong MV-DUSt3R+ sparse-view baseline with view
selection and fixed confidence thresholding. It then stress-tests RGB-only
learned reliability, match-disambiguation, candidate filtering, and monodepth
correction. Those RGB-only learned extensions do not pass reconstruction-level
gates. The final Run 30 switches the inference contract to RGB-D/source-depth
and passes the overall, occlusion, and ambiguity gates.

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

## What Is Still Not Claimed

This repository does not claim:

- RGB-only learned extensions solve occlusion or repeated structures.
- OARH/RSDH/RAJAH are final reconstruction modules.
- generic RGB-only monodepth is enough for source-ray correction.
- full MV-DUSt3R+ backbone fine-tuning was proven useful.
- the controlled ScanNet-style subset is a definitive full benchmark.

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

The staged Kaggle scripts live in `scripts/kaggle/`. Runs 0-30 are the complete
history:

- Runs 0-11: baseline, view selection, confidence thresholding, heuristic
  fusion/filtering ablations, final fixed-threshold baseline.
- Runs 12-18: first OARH/RSDH proxy and validation-gated learned extensions.
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

Final script:

```text
scripts/kaggle/kaggle_run30_rgbd_source_depth_correction.py
```

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
