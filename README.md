# Sparse-View 3D Reconstruction

This repository contains the Computer Vision 2025.2 project on sparse-view indoor
3D reconstruction with MV-DUSt3R+. It includes the proposal slide deck, Kaggle
experiment scripts, ablation notes, and the planned supervised extension for
occlusion-aware reliability and repeated-structure disambiguation.

## Repository Layout

```text
.
├── docs/
│   ├── experiments/   # run order, Kaggle guide, result summaries
│   ├── method/        # supervised OARH/RSDH extension notes
│   └── proposal/      # original proposal sources and team PDF
├── notebooks/         # notebook-based sanity checks
├── pdf/               # LaTeX Beamer source and generated main.pdf
├── scripts/
│   └── kaggle/        # reproducible staged Kaggle experiments
└── tools/             # local maintenance helpers
```

Generated Kaggle submission folders, downloaded outputs, local credentials, and
the cloned upstream `mvdust3r/` repository are intentionally ignored.

## Experiment Scripts

The staged Kaggle scripts live in `scripts/kaggle/`:

- `kaggle_run1_run2_eval_baseline.py`: evaluator smoke test and B0 baseline
- `kaggle_run3_confidence_sweep.py`: confidence threshold sweep
- `kaggle_run4_view_selection.py`: random/diversity/overlap/hybrid view ablation
- `kaggle_run5_basic_fusion.py`: F0/F1/F2/F3 fusion ablation
- `kaggle_run6_occlusion_fusion.py`: occlusion-aware filtering ablation
- `kaggle_run7_repeated_structure_filtering.py`: repeated-structure ablation
- `kaggle_run8_full_pipeline.py`: B0/B1/V/F/O/A/Full comparison
- `kaggle_run9_final_stress_test.py`: case-specific stress test
- `kaggle_run10_sensitivity_visualization.py`: confidence sensitivity figures
- `kaggle_run11_final_validation_3seeds.py`: fixed-threshold B0 vs Final over 3 seeds
- `kaggle_run12_supervised_reliability.py`: frozen-backbone OARH proxy training
- `kaggle_run13_match_disambiguation.py`: RSDH proxy match-validity training

The notebook sanity check is in `notebooks/kaggle_run0_mvdust3r_sanity.ipynb`.

## Current Finding

The strongest verified pipeline is view selection plus fixed confidence
thresholding and baseline fusion. Heuristic occlusion and ambiguity filters were
kept out of the final pipeline because they reduced F-score/completeness in the
ablations. The next planned phase is a supervised extension:

- OARH: Occlusion-Aware Reliability Head
- RSDH: Repeated-Structure Disambiguation Head
- optional light fine-tuning of confidence layers / last decoder blocks

See `docs/experiments/experiment_results_summary.md` and
`docs/method/supervised_extension_run_order.md`.

Latest Kaggle kernels:

- [Run 12 Supervised Reliability](https://www.kaggle.com/code/minhhuyen3012nguyen/mv-dust3r-run-12-supervised-reliability)
- [Run 13 Match Disambiguation](https://www.kaggle.com/code/minhhuyen3012nguyen/mv-dust3r-run-13-match-disambiguation)

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
