# Experiment Results Summary

This file summarizes the Kaggle experiment outputs for the MV-DUSt3R+ sparse-view reconstruction study.

## Final Pipeline

The best current pipeline is:

```text
Final = best view selection + fixed confidence threshold + F0 baseline fusion
```

Occlusion-aware filtering and repeated-structure ambiguity filtering are disabled in the final pipeline because their ablations reduced F-score on the smoke/proxy evaluation.

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

Interpretation: validation gating avoids the large OARH regressions at 2/3/5 views, but the 4-view gate slightly overfits the proxy validation scene and underperforms confidence-only by about `0.0025` F-score on the held-out scene. This supports keeping confidence-only as the final tested reconstruction policy while presenting OARH as a partial learned ablation that needs better labels/features.

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
against the fixed-confidence final policy using geometry metrics.

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

## Limitations

- The current Kaggle evaluation uses ScanNet posed RGB-D depth as proxy geometry, not the full ScanNet++ laser-scan mesh evaluator.
- The scene subset is small, so results should be framed as a controlled prototype/smoke benchmark rather than a definitive benchmark.
- The implemented occlusion filter is a simple front-depth heuristic and was too aggressive, reducing completeness and F-score.
- The ambiguity/repeated-structure filter is also heuristic; it reduced global F-score and should not be claimed as a successful module.
- View selection policies are lightweight proxies for diversity/overlap, not learned visibility prediction.
- Runtime excludes some dataset loading and setup overhead; report runtime per scene from the script, not Kaggle wall-clock setup time.

## Future Work

- Replace the proxy evaluator with an official mesh/laser-scan geometry evaluator on ScanNet++ scenes with full 3D ground truth.
- Improve OARH labels/features so point-label F1 translates into reconstruction F-score; Run 22 shows the Run 21 proxy head does not transfer, and Run 23 shows actual-candidate calibration is close but still not enough to beat fixed confidence.
- Analyze the Run 26 diagnostic gate to decide whether RSDH has a reconstruction-level ranking signal beyond all-candidate and confidence top-k baselines.
- Redesign occlusion reasoning using camera geometry, z-buffer consistency, and supervised per-view visibility masks.
- Revisit repeated-structure filtering as match validity learning, not as self-similarity suppression.
- If OARH/RSDH improve validation in future data, lightly fine-tune MV-DUSt3R+ confidence layers and the last decoder blocks; Run 17 records the current decision gate instead of forcing a costly fine-tune.
- Evaluate on more scenes and report confidence intervals across scene categories.
- Add qualitative failure analysis for occlusion-heavy scenes where the final pipeline remains weakest.

The proposed learned extension is documented in:

```text
docs/method/supervised_extension_run_order.md
docs/method/supervised_extension_report_section.md
```
