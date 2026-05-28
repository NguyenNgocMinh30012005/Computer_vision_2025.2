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
18. `kaggle_run19_supervised_label_cache.py`
19. `kaggle_run20_occlusion_ambiguity_subset_mining.py`
20. `kaggle_run21_oarh_v2_multitask.py`
21. `kaggle_run22_oarh_v2_reconstruction_integration.py`

The final validation script uses fixed thresholds selected before test-time
evaluation, rather than tuning on the final test rows. Run 11 prefers T4 x2,
but if Kaggle allocates a P100 with the default Torch build that cannot execute
CUDA kernels, it installs a P100-compatible Torch 2.5.1 cu121 build and restarts
once before continuing with the available GPU.

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
- Run 19 starts the stricter Phase 3 path. It creates a scalable supervised
  label cache with per-view visibility, occlusion, floating/wrong-depth, and
  geometry-consistent match labels for OARH v2 and RSDH v2.
- Run 20 consumes the Run 19 kernel output and mines focused manifests for
  OARH v2/RSDH v2, including occlusion-heavy, low-overlap/far, and hard-negative
  subsets.
- Run 21 trains an OARH v2 multitask head from the Run 20 balanced labels. It
  predicts point keep/reject, visibility class, and clipped depth residual while
  excluding direct target-label leakage features from the input.
- Run 22 uses the Run 21 checkpoint on MV-DUSt3R reconstruction candidates and
  compares OARH v2 filtering against the fixed-confidence final policy on the
  Run 20 final-eval groups.

Latest pushed kernels:

- Run 15: <https://www.kaggle.com/code/minhhuyen3012nguyen/mv-dust3r-run-15-mast3r-reciprocal-features>
- Run 16: <https://www.kaggle.com/code/minhhuyen3012nguyen/mv-dust3r-run-16-rsdh-descriptor-cycle>
- Run 17: <https://www.kaggle.com/code/minhhuyen3012nguyen/mv-dust3r-run-17-light-finetune-decision>
- Run 18: <https://www.kaggle.com/code/minhhuyen3012nguyen/mv-dust3r-run-18-learned-full-evaluation-summary>
- Run 19: <https://www.kaggle.com/code/minhhuyen3012nguyen/mv-dust3r-run-19-supervised-label-cache>
- Run 20: <https://www.kaggle.com/code/minhhuyen3012nguyen/mv-dust3r-run-20-occlusion-ambiguity-subset-mining>
- Run 21: <https://www.kaggle.com/code/minhhuyen3012nguyen/mv-dust3r-run-21-oarh-v2-multitask>
- Run 22: <https://www.kaggle.com/code/minhhuyen3012nguyen/mv-dust3r-run-22-oarh-v2-reconstruction-integration>
