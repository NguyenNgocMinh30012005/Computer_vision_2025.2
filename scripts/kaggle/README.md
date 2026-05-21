# Kaggle Experiment Scripts

These scripts reproduce the staged MV-DUSt3R+ experiments on Kaggle T4 x2.
Each script is self-contained: it clones the upstream MV-DUSt3R+ repository into
`/kaggle/temp/mvdust3r`, filters heavy Kaggle-managed dependencies, runs the
selected experiment, and writes outputs under `/kaggle/working/outputs`.

For higher Hugging Face rate limits on Kaggle, add the token as a Kaggle secret
named `HF_TOKEN`, `HUGGINGFACE_TOKEN`, or `HUGGINGFACE_HUB_TOKEN`. Run 11 and
the later helper-based scripts automatically export the secret before downloading
the MV-DUSt3R+ checkpoint.

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
13. `kaggle_run14_validation_gated_learned_pipeline.py`
14. `kaggle_run15_mast3r_reciprocal_features.py`
15. `kaggle_run16_rsdh_descriptor_cycle.py`
16. `kaggle_run17_light_finetune_decision.py`
17. `kaggle_run18_learned_full_evaluation_summary.py`

The final validation script uses fixed thresholds selected before test-time
evaluation, rather than tuning on the final test rows.

Run 12 and Run 13 are Stage-A supervised extensions. They freeze MV-DUSt3R+ and
train small MLP heads on proxy labels generated from ScanNet posed depth:

- Run 12 trains an occlusion-aware reliability proxy head for point filtering.
- Run 13 trains a repeated-structure match-disambiguation proxy head.
- Run 14 applies a validation gate: OARH is used only for view counts where it
  beats confidence-only filtering on the proxy validation scene; otherwise the
  pipeline falls back to fixed confidence filtering.
- Run 15 extracts reciprocal match features with MASt3R when available and logs
  an explicit ORB fallback if the MASt3R dependency stack is unavailable on
  Kaggle.
- Run 16 trains an RSDH descriptor/cycle-feature MLP from the Run 15
  `match_features.csv` kernel-source output when available, avoiding another
  expensive MASt3R extraction pass.
- Run 17 records a validation-based decision on whether light MV-DUSt3R+
  fine-tuning is justified before spending GPU time on backbone updates.
- Run 18 writes a final learned-extension summary comparing the verified
  confidence-only final policy against the learned/gated extensions.

Latest pushed kernels:

- Run 15: <https://www.kaggle.com/code/minhhuyen3012nguyen/mv-dust3r-run-15-mast3r-reciprocal-features>
- Run 16: <https://www.kaggle.com/code/minhhuyen3012nguyen/mv-dust3r-run-16-rsdh-descriptor-cycle>
- Run 17: <https://www.kaggle.com/code/minhhuyen3012nguyen/mv-dust3r-run-17-light-finetune-decision>
- Run 18: <https://www.kaggle.com/code/minhhuyen3012nguyen/mv-dust3r-run-18-learned-full-evaluation-summary>
