# Sparse-View 3D Reconstruction

This repository contains the Computer Vision 2025.2 project on sparse-view indoor
3D reconstruction with MV-DUSt3R+. It includes the proposal slide deck, Kaggle
experiment scripts, ablation notes, and the supervised extension plan for
occlusion-aware reliability and repeated-structure disambiguation.

## Portfolio Summary

This is my main computer vision research project. The goal is to make
sparse-view indoor 3D reconstruction more reliable when only a few views are
available and scenes contain occlusion, repeated structures, or weak overlap.

The repository is organized as a staged experiment log rather than a single
notebook dump. Runs 1-20 cover baseline evaluation, confidence thresholding,
view selection, fusion ablations, occlusion filtering, repeated-structure
analysis, supervised reliability proxies, and hard-case mining.

## Highlights

- Built a reproducible Kaggle experiment sequence around MV-DUSt3R+.
- Compared confidence filtering, view selection, and fusion strategies.
- Tested occlusion-aware and repeated-structure reliability ideas through
  ablation scripts.
- Added supervised proxy experiments for OARH/RSDH-style reliability heads.
- Documented negative results where learned or heuristic filters hurt
  reconstruction quality.

## My Contribution

- Designed the staged run order and experiment tracking structure.
- Implemented the Kaggle scripts for baseline, ablation, validation-gated, and
  supervised extension runs.
- Wrote the project documentation, method notes, and slide/report assets.
- Reported limitations explicitly instead of presenting every learned extension
  as a final improvement.

## Results Snapshot

| Area | Finding |
|---|---|
| Baseline reliability | Fixed confidence thresholding remains the safest final policy. |
| View selection | Diversity/overlap-aware selection is useful for sparse views. |
| OARH proxy | Helps selected larger-view cases but can regress 2/3-view cases. |
| RSDH proxy | Strong proxy F1, but currently best treated as an upper-bound result. |
| Final direction | Use validation-gated learned reliability instead of unconditional learned filtering. |

## Repository Layout

```text
.
+-- docs/
|   +-- experiments/   # run order, Kaggle guide, result summaries
|   +-- method/        # supervised OARH/RSDH extension notes
|   +-- proposal/      # original proposal sources and team PDF
+-- notebooks/         # notebook-based sanity checks
+-- pdf/               # LaTeX Beamer source and generated main.pdf
+-- scripts/
|   +-- kaggle/        # reproducible staged Kaggle experiments
+-- tools/             # local maintenance helpers
```

Generated Kaggle submission folders, downloaded outputs, local credentials, and
the cloned upstream `mvdust3r/` repository are intentionally ignored.
Local Hugging Face token files such as `HF.json` and `.env` are also ignored;
use Kaggle Secrets (`HF_TOKEN`) for runs that download from the Hub.

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
- `kaggle_run11_final_validation_3seeds.py`: fixed-threshold B0 vs Final over 3 seeds, with a P100 Torch compatibility fallback
- `kaggle_run12_supervised_reliability.py`: frozen-backbone OARH proxy training
- `kaggle_run13_match_disambiguation.py`: RSDH proxy match-validity training
- `kaggle_run14_validation_gated_learned_pipeline.py`: validation-gated OARH fallback policy
- `kaggle_run15_mast3r_reciprocal_features.py`: MASt3R reciprocal match feature extraction with explicit fallback logging
- `kaggle_run16_rsdh_descriptor_cycle.py`: RSDH descriptor/cycle-feature training from Run 15 kernel-source features
- `kaggle_run17_light_finetune_decision.py`: validation-based fine-tune gate
- `kaggle_run18_learned_full_evaluation_summary.py`: final learned-extension summary
- `kaggle_run19_supervised_label_cache.py`: scalable visibility/occlusion label cache for OARH v2 and RSDH v2
- `kaggle_run20_occlusion_ambiguity_subset_mining.py`: mines occlusion-heavy, low-overlap, and hard-negative subsets from Run 19
- `kaggle_run21_oarh_v2_multitask.py`: trains an OARH v2 keep/visibility/depth-residual multitask head from Run 20 labels
- `kaggle_run22_oarh_v2_reconstruction_integration.py`: evaluates whether Run 21 OARH v2 improves reconstruction F-score on Run 20 final-eval groups

The notebook sanity check is in `notebooks/kaggle_run0_mvdust3r_sanity.ipynb`.

## Current Finding

The strongest verified pipeline is view selection plus fixed confidence
thresholding and baseline fusion. Heuristic occlusion and ambiguity filters were
kept out of the final pipeline because they reduced F-score/completeness in the
ablations.

Run 12 shows that the frozen-backbone OARH proxy is not safe as an unconditional
replacement for confidence thresholding: it helps only some larger-view cases
and hurts 2/3-view reconstruction. Run 13 shows strong proxy match-validity
learning, but it should be reported as a supervised proxy result rather than a
full MASt3R-based repeated-structure solution. Run 14 therefore tests a
validation-gated learned policy that falls back to confidence-only filtering
when the learned reliability head does not win on validation. The first Run 14
log shows that gating avoids the large 2/3/5-view OARH regressions but still
slightly overfits at 4 views, so the final tested reconstruction policy remains
fixed confidence thresholding rather than unconditional learned reliability.
Run 15 successfully used the MASt3R backend for reciprocal match extraction.
Run 16 then trained RSDH from the Run 15 kernel-source features and reached
near-perfect held-out proxy match F1, but this should be reported as a
GT-depth-assisted proxy/upper-bound result because the features include direct
3D disagreement derived from available depth. The next phase starts with Run 19,
which builds a scene-level supervised label cache for visibility, occlusion, and
wrong-depth candidates before retraining OARH/RSDH with stronger held-out
evidence.
Run 20 consumes that cache and produces balanced OARH v2/RSDH v2 manifests so
the next training runs can target occlusion-heavy and hard-negative cases
instead of sampling generic points.
Run 21 trains the OARH v2 multitask head from the Run 20 balanced label file
and reports split-level plus group-level metrics for occlusion-heavy held-out
groups. It intentionally excludes direct label-leakage inputs such as
`candidate_type` and `visibility_label`; reconstruction-level proof is still
reserved for the follow-up integration run.
The pasted Run 21 log confirms strong held-out proxy metrics, with test
`keep_f1` near 0.9996 and test `occluded_f1` near 0.9795. Run 22 therefore
integrates the Run 21 checkpoint into reconstruction candidate filtering and
compares it with the fixed-confidence final policy on the Run 20 final-eval
groups.

See `docs/experiments/experiment_results_summary.md` and
`docs/method/supervised_extension_run_order.md`.

## How To Reproduce

1. Read `scripts/kaggle/README.md` and
   `docs/experiments/experiment_run_order.md`.
2. Run the staged Kaggle scripts in order, starting from the baseline and
   confidence sweep.
3. Compare the generated summaries with
   `docs/experiments/experiment_results_summary.md`.

The project expects the upstream model/data setup to be handled in Kaggle. Local
credentials and downloaded model artifacts are intentionally ignored.

Latest Kaggle kernels:

- [Run 12 Supervised Reliability](https://www.kaggle.com/code/minhhuyen3012nguyen/mv-dust3r-run-12-supervised-reliability)
- [Run 13 Match Disambiguation](https://www.kaggle.com/code/minhhuyen3012nguyen/mv-dust3r-run-13-match-disambiguation)
- [Run 14 Validation-Gated Learned Pipeline](https://www.kaggle.com/code/minhhuyen3012nguyen/mv-dust3r-run-14-validation-gated-learned-pipeline)
- [Run 15 MASt3R Reciprocal Features](https://www.kaggle.com/code/minhhuyen3012nguyen/mv-dust3r-run-15-mast3r-reciprocal-features)
- [Run 16 RSDH Descriptor Cycle](https://www.kaggle.com/code/minhhuyen3012nguyen/mv-dust3r-run-16-rsdh-descriptor-cycle)
- [Run 17 Light Finetune Decision](https://www.kaggle.com/code/minhhuyen3012nguyen/mv-dust3r-run-17-light-finetune-decision)
- [Run 18 Learned Full Evaluation Summary](https://www.kaggle.com/code/minhhuyen3012nguyen/mv-dust3r-run-18-learned-full-evaluation-summary)
- [Run 19 Supervised Label Cache](https://www.kaggle.com/code/minhhuyen3012nguyen/mv-dust3r-run-19-supervised-label-cache)
- [Run 20 Occlusion Ambiguity Subset Mining](https://www.kaggle.com/code/minhhuyen3012nguyen/mv-dust3r-run-20-occlusion-ambiguity-subset-mining)
- [Run 21 OARH v2 Multitask](https://www.kaggle.com/code/minhhuyen3012nguyen/mv-dust3r-run-21-oarh-v2-multitask)
- [Run 22 OARH v2 Reconstruction Integration](https://www.kaggle.com/code/minhhuyen3012nguyen/mv-dust3r-run-22-oarh-v2-reconstruction-integration)

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
