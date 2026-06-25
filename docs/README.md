# Project Documentation

This directory keeps project documentation separate from executable experiment
scripts.

- `experiments/`: run order, Kaggle execution notes, Run 30 reference
  summaries, and the RGB-only + estimated-depth target setting
- `method/`: OARH/RSDH diagnostic notes and the RGB-only to RGB-D transition
- `proposal/`: original proposal sources and team deliverables

Current project reports:

- `experiments/final_rgbd_result.md`: concise validated RGB-D/source-depth
  reference result and the next RGB-only estimated-depth correction target
- `experiments/project_full_report_run30.md`: project overview, Runs 0-36,
  solved limits, RGB-only negative evidence, and the Run 30 RGB-D reference
  result
- `experiments/run34_complete_results.md`: complete Run 34 configuration,
  aggregate/per-group metrics, dense exports, conclusions, and limitations
- `experiments/project_full_report_run35_run36_predicted_depth.md`: design,
  input contract, metrics, failed correction gate, and direct-depth diagnostic
- `experiments/run37_training_eval_plots.md`: Run 37 training workload,
  held-out depth metrics, per-frame distributions, and baseline-versus-fine-tuned
  plots
- `experiments/sparse_view_rgb_estimated_depth_report_vi.tex`: Vietnamese
  paper-style report for the RGB-only + fine-tuned estimated-depth correction
  setting, including Run 37 fine-tuning figures and historical reconstruction
  tables
- `experiments/sparse_view_rgb_estimated_depth_report_en.tex`: English version
  of the RGB-only + fine-tuned estimated-depth correction report
- Run 37/next-run direction: use the fine-tuned RGB depth estimator output for
  MV-DUSt3R+ source-ray correction and evaluate the resulting 3D reconstruction.
