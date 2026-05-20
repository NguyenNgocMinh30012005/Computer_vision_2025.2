# Kaggle Experiment Scripts

These scripts reproduce the staged MV-DUSt3R+ experiments on Kaggle T4 x2.
Each script is self-contained: it clones the upstream MV-DUSt3R+ repository into
`/kaggle/temp/mvdust3r`, filters heavy Kaggle-managed dependencies, runs the
selected experiment, and writes outputs under `/kaggle/working/outputs`.

Run order:

1. `kaggle_run1_run2_eval_baseline.py`
2. `kaggle_run3_confidence_sweep.py`
3. `kaggle_run4_view_selection.py`
4. `kaggle_run5_basic_fusion.py`
5. `kaggle_run6_occlusion_fusion.py`
6. `kaggle_run7_repeated_structure_filtering.py`
7. `kaggle_run8_full_pipeline.py`
8. `kaggle_run9_final_stress_test.py`
9. `kaggle_run10_sensitivity_visualization.py`
10. `kaggle_run11_final_validation_3seeds.py`
11. `kaggle_run12_supervised_reliability.py`
12. `kaggle_run13_match_disambiguation.py`

The final validation script uses fixed thresholds selected before test-time
evaluation, rather than tuning on the final test rows.

Run 12 and Run 13 are Stage-A supervised extensions. They freeze MV-DUSt3R+ and
train small MLP heads on proxy labels generated from ScanNet posed depth:

- Run 12 trains an occlusion-aware reliability proxy head for point filtering.
- Run 13 trains a repeated-structure match-disambiguation proxy head.
