# Kaggle Experiment Scripts

These scripts reproduce the staged MV-DUSt3R+ experiments on Kaggle T4 x2.
Each script is self-contained: it clones the upstream MV-DUSt3R+ repository into
`/kaggle/temp/mvdust3r`, filters heavy Kaggle-managed dependencies, runs the
selected experiment, and writes outputs under `/kaggle/working/outputs`.

For higher Hugging Face rate limits on Kaggle, add the token as a Kaggle secret
named `HF_TOKEN`, `HUGGINGFACE_TOKEN`, or `HUGGINGFACE_HUB_TOKEN`. Run 11 and
the later helper-based scripts automatically export the secret before downloading
the MV-DUSt3R+ checkpoint.

Final status: Run 30 is the final recommended script and final technical
contribution. It switches the project to sparse posed RGB-D/source-depth
inference and passes the overall, occlusion, and ambiguity gates. Runs 12-29
are kept as RGB-only learned diagnostics and negative evidence. Run 31 is a
coverage stress test of the frozen Run 30 method. Run 32 is a direct RGB-D
backprojection diagnostic without MV-DUSt3R+.

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
22. `kaggle_run23_reconstruction_candidate_calibration.py`
23. `kaggle_run24_rsdh_v2_image_only.py`
24. `kaggle_run25_rsdh_v2_reconstruction_integration.py`
25. `kaggle_run26_rsdh_v2_diagnostic_gate.py`
26. `kaggle_run27_joint_candidate_acceptance.py`
27. `kaggle_run28_ray_depth_correction.py`
28. `kaggle_run29_monodepth_ray_correction.py`
29. `kaggle_run30_rgbd_source_depth_correction.py`
30. `kaggle_run31_rgbd_coverage_stress_test.py`
31. `kaggle_run32_direct_rgbd_backprojection_baseline.py`

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
- Run 18 writes a learned-extension summary comparing the then-current Run 11
  RGB-only baseline against the learned/gated extensions.
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
  compares OARH v2 filtering against the Run 11-style fixed-confidence
  RGB-only baseline on the Run 20 final-eval groups.
- Run 23 responds to the Run 22 regression by training a reconstruction-candidate
  reliability head on actual MV-DUSt3R candidate points labeled by GT geometry.
  It evaluates learned ranking ratios against fixed confidence and gates the
  result by validation reconstruction F-score.
- Run 24 shifts to repeated-structure ambiguity. It trains RSDH v2 from Run 20
  hard-negative match labels using only image patch, coordinate, and view-policy
  features, then gates the learned match head against image-only baselines.
- Run 25 integrates the Run 24 image-only RSDH v2 checkpoint into actual
  reconstruction candidate scoring. It compares fixed confidence, RSDH
  thresholding, RSDH top-ratio ranking, and combined confidence/RSDH ranking on
  the Run 20 final-eval groups, then gates the result by validation
  reconstruction F-score.
- Run 26 is a diagnostic follow-up to Run 25. It adds `all_candidates` and
  confidence top-k baselines, uses exact top-k masks so score ties cannot
  silently keep every candidate, and gates RSDH against the best non-learned
  candidate-retention baseline. The pasted Run 26 result selects
  `all_candidates` and keeps RSDH v2 out of reconstruction because the best
  learned method only ties the best baseline by keeping all candidates.
- Run 27 is the self-contained reconstruction-aware branch for the two
  remaining limits. It does not depend on private Run 20/24 outputs. It learns
  a bounded residual over MV-DUSt3R confidence from actual reconstruction
  candidates, self-geometry support, and aggregated raw image-patch signals.
  The loss combines candidate BCE, differentiable top-k precision/GT-coverage
  F-score, hard-negative ranking, keep-ratio calibration, and residual
  regularization. Scene-level internal validation fixes the keep ratio; the
  external gate additionally requires non-regression on the hardest occlusion
  and ambiguity thirds.
- Completed Run 27 selects `all_candidates`. Learned ranking improves over
  fixed confidence but loses to full retention on validation and test, so Run
  28 changes the operation from rejection to source-ray supervised 3D
  correction while preserving candidate count.
- Run 28 uses source depth and poses only to construct train targets. It aligns
  those targets into the prediction coordinate system, trains a three-seed
  residual/trust ensemble, selects correction strength on held-out train
  scenes, and gates corrected geometry against `all_candidates` on overall,
  occlusion, and ambiguity validation subsets. The source-depth oracle is
  diagnostic only and cannot enter model selection.
- Completed Run 28 keeps `all_candidates`: validation all-candidate F-score is
  `0.1194`, learned ray-depth correction is `0.1139`, and fixed confidence is
  `0.1123`. The source-depth oracle is much stronger (`0.1936` overall,
  `0.2759` occlusion, `0.3263` ambiguity), so Run 29 replaces the small learned
  correction head with pretrained RGB-only monocular depth aligned through known
  input poses/intrinsics. It tries raw, inverse, and inverse-disparity variants
  and gates the selected monodepth correction against the same baselines.
- Completed Run 29 also keeps `all_candidates`: validation all-candidate
  F-score is `0.1208`, selected monodepth correction is `0.0949`, and fixed
  confidence is `0.1131`. The monodepth route regresses occlusion by `-0.0338`
  and ambiguity by `-0.0520`, while the source-depth diagnostic remains strong
  at `0.1979`. Run 30 therefore makes the extra resource explicit: input RGB-D
  source depth maps are allowed at inference, and full/selective source-ray
  correction policies are gated against all-candidate retention. Completed Run
  30 selects `rgbd_source_depth_selected`, uses internal policy
  `rgbd_residual_ge_0.30`, and passes all limits with validation gains of
  `+0.0559` overall, `+0.1458` on occlusion, and `+0.1377` on ambiguity.

Run 30 expected outputs:

```text
correction_label_summary.csv
policy_selection.csv
metrics.csv
summary.csv
limit_summary.csv
gate_decision.csv
run_config.json
```

Run 31 freezes `rgbd_residual_ge_0.30` and evaluates 360 sparse-view groups:
12 per scene across 30 scenes. It keeps 3/4/5 views, hybrid and
diversity-aware policies, and two deterministic frame variants per
configuration. It does not train, tune, or select a new method.

Run 31 expected outputs:

```text
group_manifest.csv
coverage_summary.csv
correction_label_summary.csv
metrics.csv
summary.csv
limit_summary.csv
paired_group_deltas.csv
stability_summary.csv
view_count_stability.csv
gate_decision.csv
run_config.json
```

Run 32 directly lifts valid source depth pixels with intrinsics/poses, attaches
RGB colors, and uses fixed voxel/sampled downsampling at the 3,500-point budget.
It mounts Run 30 output to compare the same validation/test groups and hard
subsets without test tuning. The completed primary voxel baseline is below Run
30 selected:

| Split / subset | Direct voxel RGB-D | Run 30 selected | Run 30 - direct |
| --- | ---: | ---: | ---: |
| Val overall | 0.0874 | 0.1753 | +0.0879 |
| Val occlusion | 0.1162 | 0.2522 | +0.1360 |
| Val ambiguity | 0.0998 | 0.3000 | +0.2002 |
| Test overall | 0.1359 | 0.2764 | +0.1405 |
| Test occlusion | 0.1196 | 0.3146 | +0.1951 |
| Test ambiguity | 0.1837 | 0.2948 | +0.1111 |

The auxiliary sampled direct diagnostic is much higher (`0.8500` validation
overall, `0.8666` test overall) because it shares the same input-depth source
as the controlled proxy target. Treat that as an evaluator-circularity warning,
not as an official benchmark result.

Direct RGB-D backprojection is not source-depth correction. It is a
depth-only/RGB-D baseline. Source-depth correction specifically refers to
correcting MV-DUSt3R+ candidate points using source depth residuals.

Run 32 expected outputs:

```text
metrics.csv
summary.csv
limit_summary.csv
gate_decision.csv
qualitative_manifest.csv
run_config.json
```

Latest pushed kernels:

- Run 15: <https://www.kaggle.com/code/minhhuyen3012nguyen/mv-dust3r-run-15-mast3r-reciprocal-features>
- Run 16: <https://www.kaggle.com/code/minhhuyen3012nguyen/mv-dust3r-run-16-rsdh-descriptor-cycle>
- Run 17: <https://www.kaggle.com/code/minhhuyen3012nguyen/mv-dust3r-run-17-light-finetune-decision>
- Run 18: <https://www.kaggle.com/code/minhhuyen3012nguyen/mv-dust3r-run-18-learned-full-evaluation-summary>
- Run 19: <https://www.kaggle.com/code/minhhuyen3012nguyen/mv-dust3r-run-19-supervised-label-cache>
- Run 20: <https://www.kaggle.com/code/minhhuyen3012nguyen/mv-dust3r-run-20-occlusion-ambiguity-subset-mining>
- Run 21: <https://www.kaggle.com/code/minhhuyen3012nguyen/mv-dust3r-run-21-oarh-v2-multitask>
- Run 22: <https://www.kaggle.com/code/minhhuyen3012nguyen/mv-dust3r-run-22-oarh-v2-integration>
- Run 23: <https://www.kaggle.com/code/minhhuyen3012nguyen/mv-dust3r-run-23-candidate-calibration>
- Run 24: <https://www.kaggle.com/code/minhhuyen3012nguyen/mv-dust3r-run-24-rsdh-v2-image-only>
- Run 25: <https://www.kaggle.com/code/minhhuyen3012nguyen/mv-dust3r-run-25-rsdh-v2-integration>
- Run 26: <https://www.kaggle.com/code/minhhuyen3012nguyen/mv-dust3r-run-26-rsdh-v2-diagnostic-gate>
- Run 27: <https://www.kaggle.com/code/nguynnminh/mv-dust3r-run-27-reconstruction-aware>
- Run 28: <https://www.kaggle.com/code/nguynnminh/mv-dust3r-run-28-ray-depth-correction>
- Run 29: <https://www.kaggle.com/code/nguynnminh/mv-dust3r-run-29-monodepth-ray-correction>
- Run 30: <https://www.kaggle.com/code/nguynnminh/mv-dust3r-run-30-rgbd-source-depth-correction>
- Run 31: <https://www.kaggle.com/code/nguynnminh/mv-dust3r-run-31-rgbd-coverage-stress-test>
- Run 32: <https://www.kaggle.com/code/nguynnminh/mv-dust3r-run-32-direct-rgbd-backprojection>
