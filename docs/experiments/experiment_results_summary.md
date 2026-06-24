# Experiment Results Summary

This file summarizes the Kaggle experiment outputs for the MV-DUSt3R+
sparse-view reconstruction study.

## Current Target Setting

The project now separates the validated RGB-D reference from the target
RGB-only estimated-depth setting. The target reconstruction contract is:

```text
sparse posed RGB views
+ known camera intrinsics/extrinsics
+ MV-DUSt3R+ candidate reconstruction
+ fine-tuned estimated source depth from RGB
+ predicted-depth / source-ray correction
```

Run 30 remains the validated RGB-D/source-depth reference contribution.
RGB-only Runs 12-29 are kept as baseline and negative-analysis evidence.
Run 37 fine-tunes the depth estimator that should provide the estimated source
depth for the next reconstruction correction run.

Validated reference claim:

```text
RGB-only learned extensions did not pass reconstruction-level gates. After
switching the inference contract to RGB-D, Run 30 uses input source depth maps
with known camera poses/intrinsics for source-ray correction and passes the
overall, occlusion, and ambiguity gates on held-out scenes.
```

Target-setting claim to test next:

```text
RGB-only sparse views are reconstructed by MV-DUSt3R+, while a fine-tuned depth estimator predicts per-view source depth from RGB. The predicted depth is back-projected with known camera poses/intrinsics and used to correct MV-DUSt3R+ candidates. This must pass reconstruction gates before replacing the Run 30 RGB-D reference.
```

## Final RGB-D Result

Run 30 selected `rgbd_source_depth_selected` over the best baseline
`all_candidates`, using the internal policy `rgbd_residual_ge_0.30`
(`mode=residual`, `alpha=1.0`, `residual_threshold_m=0.30`). The internal mean
reconstruction F-score was `0.2327`, with mean correction ratio `0.4705`.

| Split / subset | Baseline | RGB-D selected | Delta |
| --- | ---: | ---: | ---: |
| Val overall | 0.1194 | 0.1753 | +0.0559 |
| Val occlusion | 0.1064 | 0.2522 | +0.1458 |
| Val ambiguity | 0.1623 | 0.3000 | +0.1377 |
| Test overall | 0.1758 | 0.2764 | +0.1007 |
| Test occlusion | 0.2111 | 0.3146 | +0.1035 |
| Test ambiguity | 0.1621 | 0.2948 | +0.1327 |

Gate status: `pass_all_limits = 1` with margin `0.005`.

## Strong RGB-only Baseline

The strongest RGB-only baseline pipeline remains:

```text
Final = best view selection + fixed confidence threshold + F0 baseline fusion
```

This baseline solves the sparse-view/view-selection reliability limit, but it
does not solve occlusion or repeated/wrong-depth ambiguity. Those two remaining
limits are solved only after the Run 30 RGB-D/source-depth transition.

Occlusion-aware filtering and repeated-structure ambiguity filtering are
disabled in the RGB-only baseline because their ablations reduced F-score on
the smoke/proxy evaluation.

Fixed thresholds selected from Run 10:

| Views | Final confidence percentile |
| ---: | ---: |
| 2 | 3.0 |
| 3 | 0.2 |
| 4 | 3.0 |
| 5 | 0.5 |

## Main Run 8 Comparison

| Views | B0 F-score | B1 F-score | Full F-score | Full - B0 |
| ---: | ---: | ---: | ---: | ---: |
| 2 | 0.6630 | 0.7104 | 0.7385 | +0.0755 |
| 3 | 0.5100 | 0.5193 | 0.8118 | +0.3018 |
| 4 | 0.7148 | 0.7034 | 0.8416 | +0.1268 |
| 5 | 0.7193 | 0.7494 | 0.8190 | +0.0997 |

## Ablation Outcome

| Component | Best tested setting | Result | Report interpretation |
| --- | --- | --- | --- |
| Confidence threshold | Tuned per view count | Improved over B0 in most settings | Keep in final pipeline |
| View selection | Hybrid or diversity-aware depending on view count | Strongest contribution, especially 3/4/5 views | Keep in final pipeline |
| Basic fusion | F0 baseline | F1/F2/F3 reduced F-score | Do not use custom fusion in final |
| Occlusion-aware filtering | O0 baseline | O1 front-depth filtering strongly reduced F-score | Disable; report as failed ablation |
| Repeated-structure filtering | A0 no ambiguity filtering | A1-A4 reduced F-score | Disable; report as failed ablation |

## Run 9 Stress Test

Mean F-score over 2 scenes:

| Case | 2 views | 3 views | 4 views | 5 views |
| --- | ---: | ---: | ---: | ---: |
| Normal | 0.7378 | 0.7666 | 0.8194 | 0.7955 |
| Far-reference | 0.7301 | 0.7666 | 0.7637 | 0.7955 |
| Occlusion-heavy | 0.6219 | 0.5981 | 0.7138 | 0.7536 |
| Repeated-structure | 0.7378 | 0.7847 | 0.8194 | 0.7644 |

The weakest group is occlusion-heavy scenes. The best average setting is usually 4 views.

## Run 10 Sensitivity

Best confidence threshold by view count:

| Views | Best confidence percentile | Best F-score |
| ---: | ---: | ---: |
| 2 | 3.0 | 0.7564 |
| 3 | 0.2 | 0.8166 |
| 4 | 3.0 | 0.8654 |
| 5 | 0.5 | 0.8190 |

Generated figures:

- `fig_conf_sensitivity.png`
- `fig_runtime_vs_views.png`
- `fig_case_summary.png`

## Run 11 Final Validation

Run 11 should be used as the final B0-vs-Final table because it uses fixed thresholds selected from Run 10 and reruns B0 with 3 seeds.
The script now tolerates Kaggle assigning a P100 instead of T4 x2 by reinstalling a P100-compatible Torch build and restarting once, while still reporting the actual GPU setup in the log.

Expected output files:

- `summary_final_vs_b0_3seeds.csv`
- `metrics.csv`
- `fig_qualitative_b0_vs_final.png`

Interpretation: Run 11 is the final RGB-only baseline table, not the final
technical contribution after the project switches to RGB-D.

## Run 12 Supervised Reliability

Run 12 freezes MV-DUSt3R+ and trains a small OARH proxy MLP on GT-depth-derived point labels. The validation point-label F1 rises to about `0.969`, but reconstruction F-score on the held-out scene is mixed:

| Views | Confidence-only F-score | OARH F-score | Delta |
| ---: | ---: | ---: | ---: |
| 2 | 0.4268 | 0.2405 | -0.1863 |
| 3 | 0.5981 | 0.4265 | -0.1715 |
| 4 | 0.6513 | 0.6737 | +0.0224 |
| 5 | 0.5605 | 0.5685 | +0.0079 |

Interpretation: the learned reliability proxy is not safe as an unconditional replacement for confidence filtering. It should be validation-gated or reported as a failed/partial learned ablation, especially for 2/3-view sparse reconstruction.

## Run 13 Match Disambiguation

Run 13 trains a proxy RSDH match-validity MLP using nearest-surface consistency labels. The proxy task is strong on the held-out scene:

| Views | Match precision | Match recall | Match F1 |
| ---: | ---: | ---: | ---: |
| 3 | 0.9894 | 0.9930 | 0.9912 |
| 4 | 0.9915 | 0.9928 | 0.9922 |
| 5 | 0.9937 | 0.9951 | 0.9944 |

Interpretation: match disambiguation is promising, but this is still a supervised proxy using GT-depth-derived labels and simplified features. The report should not claim that full MASt3R-based repeated-structure disambiguation is solved yet.

## Run 14 Validation-Gated Learned Pipeline

Run 14 applies a validation gate before using OARH. The gate selects OARH only when it beats confidence-only by more than `0.005` F-score on the train-scene proxy validation.

Validation gate decisions:

| Views | Validation confidence F-score | Validation OARH F-score | Selected method |
| ---: | ---: | ---: | --- |
| 2 | 0.3772 | 0.3772 | Confidence |
| 3 | 0.6150 | 0.3411 | Confidence |
| 4 | 0.7234 | 0.7487 | OARH |
| 5 | 0.7218 | 0.3926 | Confidence |

Held-out `scene0000_01` result:

| Views | Confidence F-score | OARH F-score | Gated F-score | Gated choice |
| ---: | ---: | ---: | ---: | --- |
| 2 | 0.4252 | 0.3710 | 0.4252 | Confidence |
| 3 | 0.6029 | 0.4755 | 0.6029 | Confidence |
| 4 | 0.6763 | 0.6738 | 0.6738 | OARH |
| 5 | 0.5901 | 0.5704 | 0.5901 | Confidence |

Interpretation: validation gating avoids the large OARH regressions at 2/3/5 views, but the 4-view gate slightly overfits the proxy validation scene and underperforms confidence-only by about `0.0025` F-score on the held-out scene. This supports keeping confidence-only as the then-current RGB-only reconstruction policy while presenting OARH as a partial learned ablation that needs better labels/features.

## Run 15 MASt3R Reciprocal Features

Run 15 successfully loaded the real MASt3R backend, not the ORB fallback. It extracted reciprocal match features from `scene0000_00` as the train proxy and `scene0000_01` as the held-out scene:

| Split | Scene | Views | Backend | Matches | Positive ratio | Depth-valid ratio |
| --- | --- | ---: | --- | ---: | ---: | ---: |
| Train proxy | scene0000_00 | 3 | MASt3R | 1419 | 0.4264 | 0.8295 |
| Train proxy | scene0000_00 | 4 | MASt3R | 2837 | 0.4681 | 0.9027 |
| Train proxy | scene0000_00 | 5 | MASt3R | 2315 | 0.5076 | 0.8056 |
| Held-out | scene0000_01 | 3 | MASt3R | 2206 | 0.3305 | 0.7566 |
| Held-out | scene0000_01 | 4 | MASt3R | 4263 | 0.7361 | 0.9191 |
| Held-out | scene0000_01 | 5 | MASt3R | 1672 | 0.5879 | 0.8439 |

Interpretation: the feature extraction path is now real MASt3R reciprocal matching. Some far-view pairs produce very few or zero reciprocal matches, which is useful evidence for repeated/far-reference ambiguity handling.

## Run 16 RSDH Descriptor/Cycle Features

Run 16 consumes the successful Run 15 `match_features.csv` as a Kaggle kernel source, so it can train the small RSDH MLP even when Kaggle allocates a non-T4 GPU. The run used CPU because Kaggle allocated a P100 that was incompatible with the installed PyTorch CUDA build. Validation selected probability threshold `0.55`.

Held-out `scene0000_01` proxy match metrics:

| Views | Match precision | Match recall | Match F1 | Pairs | Positive ratio |
| ---: | ---: | ---: | ---: | ---: | ---: |
| 3 | 1.0000 | 1.0000 | 1.0000 | 965 | 0.7554 |
| 4 | 1.0000 | 1.0000 | 1.0000 | 3485 | 0.9004 |
| 5 | 1.0000 | 1.0000 | 1.0000 | 1193 | 0.8240 |

Interpretation: RSDH is very strong on this GT-depth-assisted proxy task, but this should be framed as an upper-bound/sanity result rather than a final repeated-structure solution. The features include 3D disagreement and cycle-proxy terms derived from available depth, so the result does not yet prove image-only disambiguation on unseen repeated structures.

## Submitted Learned and Phase 3 Runs

| Run | Kernel | Purpose |
| --- | --- | --- |
| 12 | `mv-dust3r-run-12-supervised-reliability` | Train a frozen-backbone OARH proxy MLP and compare learned reliability against confidence-only filtering |
| 13 | `mv-dust3r-run-13-match-disambiguation` | Train a proxy RSDH match-validity MLP using GT-depth nearest-surface consistency |
| 14 | [`mv-dust3r-run-14-validation-gated-learned-pipeline`](https://www.kaggle.com/code/minhhuyen3012nguyen/mv-dust3r-run-14-validation-gated-learned-pipeline) | Use OARH only when it beats confidence-only on the validation proxy; otherwise fall back to confidence filtering |
| 15 | [`mv-dust3r-run-15-mast3r-reciprocal-features`](https://www.kaggle.com/code/minhhuyen3012nguyen/mv-dust3r-run-15-mast3r-reciprocal-features) | Extract reciprocal match features with MASt3R when available, or an explicitly logged ORB fallback if MASt3R setup is unavailable |
| 16 | [`mv-dust3r-run-16-rsdh-descriptor-cycle`](https://www.kaggle.com/code/minhhuyen3012nguyen/mv-dust3r-run-16-rsdh-descriptor-cycle) | Train RSDH on descriptor, margin, reciprocal, 3D-disagreement, and cycle-proxy features from the Run 15 kernel-source output |
| 17 | [`mv-dust3r-run-17-light-finetune-decision`](https://www.kaggle.com/code/minhhuyen3012nguyen/mv-dust3r-run-17-light-finetune-decision) | Decide whether light MV-DUSt3R+ fine-tuning is justified from validation-gated learned results before spending GPU time |
| 18 | [`mv-dust3r-run-18-learned-full-evaluation-summary`](https://www.kaggle.com/code/minhhuyen3012nguyen/mv-dust3r-run-18-learned-full-evaluation-summary) | Summarize B0, current best, OARH/gated reliability, RSDH, and the final learned-extension recommendation |
| 19 | [`mv-dust3r-run-19-supervised-label-cache`](https://www.kaggle.com/code/minhhuyen3012nguyen/mv-dust3r-run-19-supervised-label-cache) | Build scene-level visibility, occlusion, floating/wrong-depth, and match label cache |
| 20 | [`mv-dust3r-run-20-occlusion-ambiguity-subset-mining`](https://www.kaggle.com/code/minhhuyen3012nguyen/mv-dust3r-run-20-occlusion-ambiguity-subset-mining) | Mine balanced occlusion-heavy, low-overlap, and hard-negative subsets from Run 19 |
| 21 | [`mv-dust3r-run-21-oarh-v2-multitask`](https://www.kaggle.com/code/minhhuyen3012nguyen/mv-dust3r-run-21-oarh-v2-multitask) | Train the OARH v2 keep/visibility/depth-residual multitask head from Run 20 balanced labels |
| 22 | [`mv-dust3r-run-22-oarh-v2-integration`](https://www.kaggle.com/code/minhhuyen3012nguyen/mv-dust3r-run-22-oarh-v2-integration) | Test whether the Run 21 OARH v2 head improves reconstruction F-score on final-eval groups |
| 23 | [`mv-dust3r-run-23-candidate-calibration`](https://www.kaggle.com/code/minhhuyen3012nguyen/mv-dust3r-run-23-candidate-calibration) | Train a reliability head on actual MV-DUSt3R reconstruction candidates after Run 22 exposed proxy-to-reconstruction domain shift |
| 24 | [`mv-dust3r-run-24-rsdh-v2-image-only`](https://www.kaggle.com/code/minhhuyen3012nguyen/mv-dust3r-run-24-rsdh-v2-image-only) | Train image-only RSDH v2 match validity from Run 20 hard-negative labels |

## Run 19 Supervised Label Cache

Run 19 starts the stricter phase for solving the remaining occlusion and repeated-structure limitations. It writes a reusable label cache instead of training immediately:

- `label_cache.csv`: per-candidate keep, visibility, occlusion, floating/wrong-depth, and geometry-consistent match labels.
- `label_summary.csv`: group-level visible/occluded/floating ratios.
- `view_group_manifest.csv`: selected sparse-view groups.
- `scene_split.csv`: scene-level train/val/test assignment.
- `occlusion_heavy_groups.csv`: view groups with occlusion ratio above the configured threshold.

Interpretation rule: Run 19 is a data/label validation step. OARH v2 and RSDH v2 should only be trained after this cache shows enough occlusion-heavy and hard-negative examples on held-out scenes.

The first Run 19 log confirms a larger 30-scene scan with scene-level split:
18 train scenes, 6 validation scenes, and 6 test scenes. It found held-out
occlusion-heavy groups, including `scene0011_01_4_diversity_aware`,
`scene0013_00_3_hybrid`, `scene0013_01_4_hybrid`, and
`scene0013_01_5_hybrid`, plus a validation occlusion-heavy group
`scene0009_01_3_hybrid`.

## Run 20 Occlusion/Ambiguity Subset Mining

Run 20 consumes the Run 19 kernel output and mines focused subsets for the next training runs:

- `subset_group_manifest.csv`: all groups with occlusion/low-overlap/wrong-depth classes and priority scores.
- `final_eval_group_manifest.csv`: validation/test groups eligible for final case-specific evaluation.
- `oarh_v2_balanced_labels.csv`: balanced point-label rows for OARH v2.
- `rsdh_v2_hard_negative_labels.csv`: balanced match-positive / hard-negative rows for RSDH v2.
- `sample_bucket_counts.csv`: available vs sampled counts per split and label bucket.

Interpretation rule: Run 20 is the bridge from label creation to real training. Run 21 should train OARH v2 from `oarh_v2_balanced_labels.csv`; Run 24 should train RSDH v2 using image-only patch/coordinate features and `rsdh_v2_hard_negative_labels.csv` labels.

The pasted Run 20 log confirms that it streamed all 4,423,680 Run 19 label
rows, selected 32 final evaluation groups, mined 79 OARH candidate groups and
108 RSDH candidate groups, and sampled 320,692 OARH rows plus 212,100 RSDH rows.
The highest-priority final-eval groups include occlusion-core, borderline
occlusion, low-overlap/far, and wrong-depth hard-negative cases.

## Run 21 OARH v2 Multitask Training

Run 21 consumes `oarh_v2_balanced_labels.csv` from Run 20 and trains a compact
multitask MLP:

- binary keep/reject point reliability,
- 3-way visibility class prediction,
- clipped depth-residual regression.

The input features deliberately exclude direct target-label leakage such as
`candidate_type` and `visibility_label`. Run 21 is therefore the first real
OARH v2 training run in Phase 3, but it remains a label-cache/proxy test until
Run 22 integrates the learned head into reconstruction and reports geometry
metrics on the final occlusion-heavy groups.

The pasted Run 21 log shows a strong proxy result:

- validation rows: 75,841; test rows: 129,766,
- validation `keep_f1`: 0.99995; test `keep_f1`: 0.99958,
- validation `visibility_accuracy`: 0.99119; test `visibility_accuracy`: 0.98081,
- validation `occluded_f1`: 0.98836; test `occluded_f1`: 0.97954,
- test depth residual MAE: 0.05666 m.

Interpretation rule: this is a necessary but not sufficient result. It says the
OARH v2 head learned the Run 20 labels well on held-out groups. It does not yet
prove reconstruction improvement, so Run 22 compares OARH-filtered point clouds
against the then-current Run 11 RGB-only baseline using geometry metrics.

## Run 22 OARH v2 Reconstruction Integration

Run 22 consumes `final_eval_group_manifest.csv` from Run 20 and
`oarh_v2_multitask_head.pt` from Run 21. For each selected final-eval group, it
runs MV-DUSt3R, builds OARH v2 features for reconstruction candidates, and
compares:

- `confidence_fixed_final`,
- `oarh_v2_threshold_0.50`,
- `oarh_v2_threshold_0.70`,
- `oarh_v2_threshold_0.90`,
- `oarh_v2_and_confidence_guard`.

Run 22 writes validation/test summaries and a gate decision. If learned OARH
does not beat fixed confidence on validation by the configured margin, the
project should keep the fixed-confidence reconstruction policy.

The pasted Run 22 log gives a clear negative gate:

| Split | Method | Mean F-score | Delta vs fixed confidence | Mean selected ratio |
| --- | --- | ---: | ---: | ---: |
| Val | Fixed confidence | 0.6716 | 0.0000 | 0.9874 |
| Val | Best learned OARH v2 (`threshold_0.70`) | 0.2160 | -0.4556 | 0.3385 |
| Test | Fixed confidence | 0.6033 | 0.0000 | 0.9877 |
| Test | Best learned OARH v2 (`threshold_0.50`) | 0.1936 | -0.4097 | 0.2507 |

Interpretation: Run 21 solved the proxy labels, but not the reconstruction
policy. The learned OARH v2 head rejects too many real reconstruction
candidates, causing a large recall/completeness collapse. The gate correctly
selects `confidence_fixed_final`.

## Run 23 Reconstruction Candidate Calibration

Run 23 is the direct follow-up to the Run 22 failure. Instead of training on
Run 20 proxy candidate rows, it generates labels from actual MV-DUSt3R
reconstruction candidates:

- run MV-DUSt3R on selected train, validation, and test groups,
- label each predicted candidate point by nearest GT geometry distance,
- train a reconstruction-candidate reliability head from confidence, point
  coordinates, view metadata, and cross-view predicted support,
- evaluate learned ranking ratios against fixed confidence,
- choose any learned policy only when validation reconstruction F-score beats
  fixed confidence by the configured margin.

This run tests whether the occlusion/reliability limit can be improved after
removing the proxy-to-reconstruction domain mismatch seen in Run 22.

The pasted Run 23 log is much closer than Run 22, but still fails the
validation gate:

| Split | Method | Mean F-score | Delta vs fixed confidence | Mean selected ratio |
| --- | --- | ---: | ---: | ---: |
| Val | Fixed confidence | 0.6674 | 0.0000 | 0.9874 |
| Val | Best learned RCRH (`top_ratio_0.995`) | 0.6618 | -0.0056 | 0.9950 |
| Test | Fixed confidence | 0.6014 | 0.0000 | 0.9877 |
| Test | Best learned RCRH (`top_ratio_0.995`) | 0.5923 | -0.0090 | 0.9950 |

Interpretation: training on actual reconstruction candidates removes most of
the catastrophic domain shift from Run 22, but the learned ranker only stays
competitive when it keeps almost every point. It is not yet a real occlusion
solution because lower keep ratios collapse recall and completeness.

## Run 24 RSDH v2 Image-Only Match Validity

Run 24 moves to the remaining repeated-structure / wrong-match limitation. It
consumes `rsdh_v2_hard_negative_labels.csv` from Run 20 and trains a match
validity head using only prediction-time image information:

- source/target normalized pixel coordinates,
- view count and view-policy indicators,
- RGB, grayscale, gradient, and downsampled patch similarity features around
  the source and target pixels.

It deliberately excludes `candidate_type`, `visibility_label`, GT depth
residuals, and group-class labels from inference features. Validation chooses
the learned threshold and gates the learned MLP against simple image-only
baselines before any Run 25 reconstruction integration.

The pasted Run 24 files show that the learned image-only RSDH MLP passes this
gate:

| Split | Best image-only baseline F1 | RSDH v2 MLP F1 | Delta |
| --- | ---: | ---: | ---: |
| Train | 0.5886 | 0.7422 | +0.1536 |
| Val | 0.6212 | 0.6954 | +0.0741 |
| Test | 0.5517 | 0.6596 | +0.1079 |

Validation selects `rsdh_v2_image_only_mlp` over the patch-similarity
threshold baseline. Group-level results are still weaker on low-overlap/far
cases, but the learned head improves the average validation and test group F1.
This is the first Phase 3 learned component that clears its proxy gate without
GT-depth leakage, so Run 25 integrates it into reconstruction candidate scoring.

## Run 25 RSDH v2 Reconstruction Integration

Run 25 consumes `final_eval_group_manifest.csv` from Run 20 and
`rsdh_v2_image_only_head.pt` from Run 24. It reruns MV-DUSt3R on the final
evaluation groups, projects reconstruction candidates into the other selected
views, scores image-only patch/coordinate match features with the RSDH v2 MLP,
and compares:

- `confidence_fixed_final`,
- RSDH threshold-only and RSDH-plus-confidence guards,
- RSDH top-ratio ranking policies,
- combined confidence/RSDH ranking policies.

The gate remains reconstruction-level: the learned RSDH policy should be used
only if validation mean F-score beats fixed confidence by the configured
margin. Otherwise, the project should report Run 24 as a strong image-only
match-validity result but keep fixed confidence for the final reconstruction
pipeline.

The pasted Run 25 log does not clear that gate:

| Split | Method | Mean F-score | Delta vs fixed confidence | Mean selected ratio |
| --- | --- | ---: | ---: | ---: |
| Val | Fixed confidence | 0.1507 | 0.0000 | 0.9874 |
| Val | Best learned RSDH | 0.1542 | +0.0035 | 1.0000 |
| Test | Fixed confidence | 0.2184 | 0.0000 | 0.9877 |
| Test | Best learned RSDH/top-ratio | 0.2372 | +0.0189 | 1.0000 |

Interpretation: RSDH v2 is not rejected as a match-validity model, but the
reconstruction result is not clean enough to claim a solved repeated-structure
module. The best validation learned policy is below the `0.005` margin, and the
best test policies keep all candidates because top-ratio scoring ties collapse
to a full keep mask. Run 26 therefore adds all-candidate and confidence top-k
baselines plus exact top-k tie handling.

## Run 26 RSDH v2 Diagnostic Gate

Run 26 reruns the Run 25 reconstruction-integration setup but changes the
decision test:

- add `all_candidates` as a candidate-retention baseline,
- add confidence top-k baselines at the same ratios used by RSDH,
- use exact top-k masks with deterministic tie-breaking,
- gate RSDH against the best non-learned baseline, not only fixed confidence.

This run decides whether RSDH contributes a real learned ranking signal or
whether the Run 25 gains are explained by simply keeping more reconstruction
candidates.

The pasted Run 26 log answers that diagnostic: RSDH does not beat the
candidate-retention baselines at reconstruction level.

| Split | Method | Mean F-score | Delta vs fixed confidence | Delta vs best baseline | Mean selected ratio |
| --- | --- | ---: | ---: | ---: | ---: |
| Val | Fixed confidence | 0.1455 | 0.0000 | -0.0008 | 0.9874 |
| Val | All candidates / confidence top-ratio 1.0 | 0.1463 | +0.0008 | 0.0000 | 1.0000 |
| Val | Best learned RSDH | 0.1463 | +0.0008 | 0.0000 | 1.0000 |
| Test | Fixed confidence | 0.2271 | 0.0000 | -0.0135 | 0.9877 |
| Test | All candidates / confidence top-ratio 1.0 | 0.2407 | +0.0135 | 0.0000 | 1.0000 |
| Test | Best learned RSDH | 0.2407 | +0.0135 | 0.0000 | 1.0000 |

Gate decision: select `all_candidates`; best baseline is `all_candidates`; best
learned method is `combined_max_top_ratio_1.0000`; validation gain over the
best baseline is `0.0000`, below the `0.005` margin. Interpretation: keep RSDH
v2 out of the reconstruction pipeline. Its Run 24 image-only classifier remains
a useful proxy result, but it does not yet solve repeated-structure ambiguity in
actual geometry reconstruction.

## Run 27 Reconstruction-Aware Joint Acceptance

Run 27 is redesigned to remove both failure sources exposed by Runs 22--26:
proxy-to-reconstruction domain shift and private upstream kernel dependencies.
It is self-contained apart from the public ScanNet dataset and MV-DUSt3R
checkpoint. It creates deterministic scene-level train/validation/test splits
and trains directly on actual reconstruction candidates.

The inference features combine:

- MV-DUSt3R confidence, candidate layout, and cross-view point support;
- aggregated source/target RGB, grayscale, gradient, correlation, and patch
  disagreement signals computed directly from the selected images;
- no Run 20 labels, Run 24 checkpoint, GT residual, candidate type, or
  scene/group class as an inference feature.

The learned score is a bounded residual over the confidence logit:

```text
score_i = logit(confidence_rank_i) + 2 tanh(MLP(feature_i))
```

For each training keep ratio, a differentiable top-k mask `k_i` optimizes a
soft reconstruction F-score:

```text
soft_precision = sum(k_i y_i) / sum(k_i)
soft_recall = sum(k_i coverage_i) / number_of_GT_points
soft_F1 = 2 soft_precision soft_recall / (soft_precision + soft_recall)
```

`coverage_i` counts GT surface points for which candidate `i` is the nearest
valid reconstruction point. The complete objective also includes candidate
BCE, hard-negative ranking, keep-ratio calibration, and a residual penalty.
Valid low-support points receive extra weight so occluded geometry is not
discarded, while high-confidence wrong-depth candidates receive extra negative
ranking weight.

The keep ratio is selected on held-out training scenes, before external
validation. Three independently trained heads are ensembled. The final gate
passes only when the fixed learned policy:

1. beats the best non-learned validation baseline by at least `0.005` F-score;
2. does not regress on the most occlusion-challenging validation third;
3. does not regress on the most ambiguity-challenging validation third.

This design prevents both previous false successes: improving proxy labels
without improving geometry, and matching the baseline only by retaining every
candidate.

The completed Run 27 gate is negative:

| Split | Method | Mean F-score | Delta vs fixed confidence | Delta vs best baseline | Mean selected ratio |
| --- | --- | ---: | ---: | ---: | ---: |
| Val | Fixed confidence | 0.1149 | 0.0000 | -0.0077 | 0.9835 |
| Val | All candidates | 0.1226 | +0.0077 | 0.0000 | 1.0000 |
| Val | RAJAH selected ratio | 0.1211 | +0.0062 | -0.0015 | 0.9951 |
| Test | Fixed confidence | 0.1753 | 0.0000 | -0.0073 | 0.9877 |
| Test | All candidates | 0.1825 | +0.0073 | 0.0000 | 1.0000 |
| Test | RAJAH selected ratio | 0.1797 | +0.0044 | -0.0029 | 0.9951 |

Validation selects `all_candidates`. The learned head fails all three strict
conditions: overall delta `-0.0015`, occlusion-subset delta `-0.0005`, and
ambiguity-subset delta `-0.0018` against each subset's best baseline. The
learned ranking is useful relative to fixed confidence and wins strongly on a
few individual hard groups, but it cannot remove even `0.5%` of candidates
without losing more recall than it gains in precision.

The main diagnosis is stronger than another failed classifier result:
`all_candidates` has both higher F-score and higher mean precision than fixed
confidence. Candidate rejection is therefore the wrong operation for this
benchmark. The next experiment should preserve candidate count and learn a
per-candidate 3D correction from the source-pixel depth supervision available
during training. This can correct repeated/wrong-depth geometry without
discarding valid points that lack cross-view support because of occlusion.

Completed Run 28 is also negative, but it reveals strong correction headroom:

| Split | Method | Mean F-score | Delta vs all candidates |
| --- | --- | ---: | ---: |
| Val | Fixed confidence | 0.1123 | -0.0071 |
| Val | All candidates | 0.1194 | 0.0000 |
| Val | Learned ray-depth correction | 0.1139 | -0.0056 |
| Val | Oracle source-depth correction | 0.1936 | +0.0742 |
| Test | Fixed confidence | 0.1662 | -0.0096 |
| Test | All candidates | 0.1758 | 0.0000 |
| Test | Learned ray-depth correction | 0.1700 | -0.0058 |
| Test | Oracle source-depth correction | 0.2886 | +0.1128 |

The learned correction improves the ambiguity validation subset by `+0.0085`
against all candidates, but it regresses occlusion by `-0.0210`, so the overall
gate fails. In contrast, the oracle improves validation occlusion by `+0.1695`
and ambiguity by `+0.1640`. Run 29 therefore tests pretrained RGB-only
monocular depth aligned through input poses/intrinsics as a non-GT inference
approximation to the source-depth oracle.

Completed Run 29 is a stronger negative RGB-only result:

| Split | Method | Mean F-score | Delta vs all candidates |
| --- | --- | ---: | ---: |
| Val | All candidates | 0.1208 | 0.0000 |
| Val | Fixed confidence | 0.1131 | -0.0078 |
| Val | Selected monodepth correction | 0.0949 | -0.0260 |
| Val | Source-depth diagnostic | 0.1979 | +0.0771 |
| Test | All candidates | 0.1786 | 0.0000 |
| Test | Fixed confidence | 0.1669 | -0.0117 |
| Test | Selected monodepth correction | 0.1528 | -0.0258 |
| Test | Source-depth diagnostic | 0.2726 | +0.0940 |

The selected RGB-only monodepth correction regresses validation occlusion by
`-0.0338` and ambiguity by `-0.0520`. The best diagnostic monodepth variant is
closer but still below all candidates. Run 30 therefore tests the direct
resource-expanded solution: allow source depth maps from the input RGB-D frames
at inference, then gate full/selective source-ray correction policies. Completed
Run 30 solves the two remaining limits only under an explicit RGB-D/source-depth
assumption, not as an RGB-only claim.

Expected outputs for the submitted remaining runs:

- Run 15: `match_features.csv`, `feature_summary.csv`, `run_config.json`
- Run 16: `match_metrics.csv`, `training_history.csv`, `feature_summary.csv`, `rsdh_descriptor_cycle_head.pt`, `run_config.json`
- Run 17: `fine_tune_decision.csv`, `run_config.json`
- Run 18: `final_learned_summary.csv`, `run_config.json`
- Run 19: `label_cache.csv`, `label_summary.csv`, `view_group_manifest.csv`, `scene_split.csv`, `occlusion_heavy_groups.csv`, `run_config.json`
- Run 20: `subset_group_manifest.csv`, `final_eval_group_manifest.csv`, `oarh_v2_balanced_labels.csv`, `rsdh_v2_hard_negative_labels.csv`, `sample_bucket_counts.csv`, `run_config.json`
- Run 21: `training_history.csv`, `split_metrics.csv`, `final_eval_group_metrics.csv`, `oarh_v2_multitask_head.pt`, `run_config.json`
- Run 22: `metrics.csv`, `summary.csv`, `gate_decision.csv`, `run_config.json`
- Run 23: `candidate_label_summary.csv`, `training_history.csv`, `metrics.csv`, `summary.csv`, `gate_decision.csv`, `rcrh_candidate_head.pt`, `run_config.json`
- Run 24: `split_metrics.csv`, `group_metrics.csv`, `training_history.csv`, `feature_summary.csv`, `gate_decision.csv`, `rsdh_v2_image_only_head.pt`, `run_config.json`
- Run 25: `metrics.csv`, `summary.csv`, `gate_decision.csv`, `run_config.json`
- Run 26: `metrics.csv`, `summary.csv`, `gate_decision.csv`, `run_config.json`
- Run 27: `candidate_label_summary.csv`, `training_history.csv`, `model_selection.csv`, `metrics.csv`, `summary.csv`, `limit_summary.csv`, `gate_decision.csv`, `scene_split.csv`, `view_group_manifest.csv`, `joint_candidate_acceptance_head.pt`, `run_config.json`
- Run 28: `correction_label_summary.csv`, `training_history.csv`, `metrics.csv`, `summary.csv`, `limit_summary.csv`, `gate_decision.csv`, `ray_depth_correction_head.pt`, `run_config.json`
- Run 29: `correction_label_summary.csv`, `policy_selection.csv`, `metrics.csv`, `summary.csv`, `limit_summary.csv`, `gate_decision.csv`, `run_config.json`
- Run 30: `correction_label_summary.csv`, `policy_selection.csv`, `metrics.csv`, `summary.csv`, `limit_summary.csv`, `gate_decision.csv`, `run_config.json`
- Run 31: `group_manifest.csv`, `coverage_summary.csv`, `correction_label_summary.csv`, `metrics.csv`, `summary.csv`, `limit_summary.csv`, `paired_group_deltas.csv`, `stability_summary.csv`, `view_count_stability.csv`, `gate_decision.csv`, `run_config.json`
- Run 32: `metrics.csv`, `summary.csv`, `limit_summary.csv`, `gate_decision.csv`, `qualitative_manifest.csv`, `run_config.json`
- Run 33: `metrics.csv`, `summary.csv`, `limit_summary.csv`, `gate_decision.csv`, `run_config.json`, optional `qualitative_manifest.csv`
- Run 34: `qualitative_3d_manifest.csv`, `normalized_metric_summary.csv`,
  `dense_scene_manifest.csv`, scene-like GLBs, and `run_config.json`

Run 34 verified means over its three selected 3,500-point comparison groups:

| Method | Mean normalized distance | Mean DAc@0.2 normalized |
| --- | ---: | ---: |
| `rgbd_source_depth_selected` | **0.0515** | **0.9841** |
| `direct_rgbd_backprojection` | 0.0926 | 0.8617 |
| `mvdust3r_rgb_only_all_candidates` | 0.1308 | 0.7967 |

These auxiliary normalized metrics agree with the Run 30 shape-level advantage
on this small qualitative subset. They do not replace the full Run 30
validation/test gate or provide an independent official-mesh benchmark.

## Run 30 RGB-D Source-Depth Reference

Run 30 is the accepted RGB-D reference result. It changes the inference
contract from RGB-only to sparse posed RGB-D. Source depth maps from input
frames anchor each source ray to metric geometry, so the method corrects
wrong-depth candidates instead of deleting candidates and losing recall.

Run 30 preserves the negative evidence from Runs 22-29:

- proxy F1 did not transfer to reconstruction;
- filtering hurt recall/completeness;
- all-candidate retention was a strong baseline;
- RGB-only monodepth did not recover metric/source depth well enough.

The RGB-D gate passes all required conditions:

- overall validation delta is above `0.005`;
- occlusion delta is non-negative and large;
- ambiguity delta is non-negative and large.

## Run 31 RGB-D Coverage Stress Test

Run 31 is a validation-only follow-up. It freezes the selected Run 30 policy
and increases coverage to 360 sparse-view groups over the same 30 scenes:

```text
30 scenes
x 3 view counts (3/4/5)
x 2 view policies (hybrid/diversity-aware)
x 2 deterministic frame variants
= 360 sparse-view groups
```

No training, policy search, or method selection is performed. The primary
outputs are paired F-score deltas versus `all_candidates`, view-count
breakdowns, and scene-cluster bootstrap 95% confidence intervals. Status:
submitted/pending result.

## Run 32 Direct RGB-D Backprojection Baseline

Run 32 is a no-MV-DUSt3R+ baseline. For each Run 30 sparse-view group it:

1. reads input RGB, depth, intrinsics, and camera poses;
2. back-projects every valid source depth pixel into the first-camera frame;
3. attaches source RGB colors;
4. applies a fixed 2 cm voxel downsample and 3,500-point cap;
5. evaluates the same metrics and hard subsets as Run 30.

Direct RGB-D backprojection is not source-depth correction. It is a
depth-only/RGB-D baseline. Source-depth correction specifically refers to
correcting MV-DUSt3R+ candidate points using source depth residuals.

The script mounts Run 30 outputs and reports direct-minus-Run-30 deltas. Run 32
is complete. The pre-registered primary direct method is the voxelized
`direct_rgbd_backprojection`, and the gate outcome is
`run30_adds_value_over_direct`.

| Split / subset | Direct voxel RGB-D | Run 30 selected | Run 30 - direct |
| --- | ---: | ---: | ---: |
| Val overall | 0.0874 | 0.1753 | +0.0879 |
| Val occlusion | 0.1162 | 0.2522 | +0.1360 |
| Val ambiguity | 0.0998 | 0.3000 | +0.2002 |
| Test overall | 0.1359 | 0.2764 | +0.1405 |
| Test occlusion | 0.1196 | 0.3146 | +0.1951 |
| Test ambiguity | 0.1837 | 0.2948 | +0.1111 |

Evaluation warning: `build_gt_cloud` uses depth PNGs from the same selected
input views. Direct backprojection and the evaluation target therefore share a
depth source, so this diagnostic is not an independent full-ScanNet or official
mesh benchmark. The additional `direct_rgbd_backprojection_sampled` diagnostic
scores `0.8500` validation overall and `0.8666` test overall, which confirms
that direct depth sampling can exploit the proxy-target construction. Do not
turn that sampled diagnostic into a final method claim without independent
mesh/laser-scan ground truth.

## Run 33 MV-DUSt3R+ Only RGB Baseline

Run 33 is a final diagnostic baseline for the original RGB-only backbone. It
does not rerun MV-DUSt3R+ inference. Instead, it reuses the Run 30 rows that do
not touch source depth:

- `all_candidates` -> `mvdust3r_raw_all_candidates`;
- `confidence_fixed_final` -> `mvdust3r_confidence_fixed`.

The script writes explicit zero/false flags for source-depth inference,
source-depth correction, and direct RGB-D backprojection. This makes the input
contract unambiguous: selected sparse RGB views only, MV-DUSt3R+ confidence,
and the existing project evaluator.

The best RGB-only MV-DUSt3R+ baseline is `mvdust3r_raw_all_candidates`.

| Split / subset | MV-DUSt3R+ RGB-only | Run 30 selected | Run 30 - RGB-only |
| --- | ---: | ---: | ---: |
| Val overall | 0.1194 | 0.1753 | +0.0559 |
| Val occlusion | 0.1064 | 0.2522 | +0.1458 |
| Val ambiguity | 0.1623 | 0.3000 | +0.1377 |
| Test overall | 0.1758 | 0.2764 | +0.1007 |
| Test occlusion | 0.2111 | 0.3146 | +0.1035 |
| Test ambiguity | 0.1621 | 0.2948 | +0.1327 |

Run 33 gate outcome is `run30_adds_value_over_mvdust3r_only`. The final claim
does not change: the project still claims a Run 30 RGB-D/source-depth result,
not an RGB-only solution for occlusion or repeated/wrong-depth ambiguity.

## Runs 35-36 RGB-only Predicted-Depth Generalization Track

Run 35 uses
`depth-anything/Depth-Anything-V2-Metric-Indoor-Small-hf` to measure raw and
scale-aligned monocular depth quality. Run 36 then uses only RGB, cached
predicted depth, and known poses/intrinsics for correction. True source depth is
evaluation-only in Run 36.

Run 35 is complete on 158 unique frames across 30 scenes:

| Split | AbsRel | RMSE (m) | MAE (m) | delta1 | Scale-aligned RMSE (m) | Scale-aligned MAE (m) |
| --- | ---: | ---: | ---: | ---: | ---: | ---: |
| Val | 0.2207 | 0.4773 | 0.4115 | 0.7175 | 0.2175 | 0.1088 |
| Test | 0.1861 | 0.4642 | 0.3525 | 0.7736 | 0.2786 | 0.1614 |

The train-fit global scale is `0.9123`. Run 36 selected global scale,
`tau_pred = 0.5`, and `alpha = 0.25`.

| Split | RGB-only all candidates | Selected correction | Direct predicted depth | Run 30 true RGB-D |
| --- | ---: | ---: | ---: | ---: |
| Val F-score | 0.1231 | 0.1012 | 0.1747 | 0.1753 |
| Test F-score | 0.1744 | 0.1465 | 0.1894 | 0.2764 |

The selected correction fails the overall, occlusion, and ambiguity gates.
Validation deltas versus RGB-only are `-0.0218`, `-0.0421`, and `-0.0310`.
Direct predicted-depth backprojection is stronger overall, but it regresses on
test occlusion (`0.1840` versus `0.2120`) and does not approach Run 30 test
performance. The final claim remains Run 30 RGB-D/source-depth correction.

## Limitations

- The final solved setting is RGB-D/source-depth, not RGB-only.
- RGB-only OARH/RSDH/RAJAH/monodepth components are diagnostics and baselines,
  not final reconstruction modules.
- The current Kaggle evaluation uses ScanNet posed RGB-D depth as proxy
  geometry, not the full ScanNet++ laser-scan mesh evaluator.
- Run 34 normalized distance and DAc@0.2 remove translation and isotropic
  scale, but still use the same depth-derived proxy GT and do not remove
  rotation.
- The scene subset is small, so results should be framed as a controlled
  prototype/smoke benchmark rather than a definitive benchmark.
- View selection policies are lightweight proxies for diversity/overlap, not
  learned visibility prediction.
- Runtime excludes some dataset loading and setup overhead; report runtime per
  scene from the script, not Kaggle wall-clock setup time.

## Future Work

- Package Run 30 qualitative examples for the final slide deck.
- Evaluate the RGB-D/source-depth method on more scenes and report confidence
  intervals across scene categories.
- Replace the proxy evaluator with an official mesh/laser-scan geometry
  evaluator on scenes with full 3D ground truth.
- If a future project returns to RGB-only, it needs a stronger metric-depth
  calibration or indoor-depth fine-tuning path; Runs 22-29 show current
  RGB-only heads are insufficient.

The diagnostic learned-extension history is documented in:

```text
docs/method/supervised_extension_run_order.md
docs/method/supervised_extension_report_section.md
```
