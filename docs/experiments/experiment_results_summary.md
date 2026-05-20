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

## Submitted Phase 2 Runs

| Run | Kernel | Purpose |
| --- | --- | --- |
| 12 | `mv-dust3r-run-12-supervised-reliability` | Train a frozen-backbone OARH proxy MLP and compare learned reliability against confidence-only filtering |
| 13 | `mv-dust3r-run-13-match-disambiguation` | Train a proxy RSDH match-validity MLP using GT-depth nearest-surface consistency |
| 14 | [`mv-dust3r-run-14-validation-gated-learned-pipeline`](https://www.kaggle.com/code/minhhuyen3012nguyen/mv-dust3r-run-14-validation-gated-learned-pipeline) | Use OARH only when it beats confidence-only on the validation proxy; otherwise fall back to confidence filtering |

## Limitations

- The current Kaggle evaluation uses ScanNet posed RGB-D depth as proxy geometry, not the full ScanNet++ laser-scan mesh evaluator.
- The scene subset is small, so results should be framed as a controlled prototype/smoke benchmark rather than a definitive benchmark.
- The implemented occlusion filter is a simple front-depth heuristic and was too aggressive, reducing completeness and F-score.
- The ambiguity/repeated-structure filter is also heuristic; it reduced global F-score and should not be claimed as a successful module.
- View selection policies are lightweight proxies for diversity/overlap, not learned visibility prediction.
- Runtime excludes some dataset loading and setup overhead; report runtime per scene from the script, not Kaggle wall-clock setup time.

## Future Work

- Replace the proxy evaluator with an official mesh/laser-scan geometry evaluator on ScanNet++ scenes with full 3D ground truth.
- Improve OARH labels/features so point-label F1 translates into reconstruction F-score; current Run 12 results are mixed.
- Add real MASt3R reciprocal matching and train RSDH with descriptor margin and cycle consistency, beyond the Run 13 proxy.
- Redesign occlusion reasoning using camera geometry, z-buffer consistency, and supervised per-view visibility masks.
- Revisit repeated-structure filtering as match validity learning, not as self-similarity suppression.
- If OARH/RSDH improve validation, lightly fine-tune MV-DUSt3R+ confidence layers and the last decoder blocks.
- Evaluate on more scenes and report confidence intervals across scene categories.
- Add qualitative failure analysis for occlusion-heavy scenes where the final pipeline remains weakest.

The proposed learned extension is documented in:

```text
docs/method/supervised_extension_run_order.md
docs/method/supervised_extension_report_section.md
```
