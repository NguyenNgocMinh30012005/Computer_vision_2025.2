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

## Submitted Phase 2 Runs

The first supervised-extension kernels have been pushed to Kaggle and are waiting for user-provided logs/results:

| Run | Kernel | Purpose |
| --- | --- | --- |
| 12 | `mv-dust3r-run-12-supervised-reliability` | Train a frozen-backbone OARH proxy MLP and compare learned reliability against confidence-only filtering |
| 13 | `mv-dust3r-run-13-match-disambiguation` | Train a proxy RSDH match-validity MLP using GT-depth nearest-surface consistency |

## Limitations

- The current Kaggle evaluation uses ScanNet posed RGB-D depth as proxy geometry, not the full ScanNet++ laser-scan mesh evaluator.
- The scene subset is small, so results should be framed as a controlled prototype/smoke benchmark rather than a definitive benchmark.
- The implemented occlusion filter is a simple front-depth heuristic and was too aggressive, reducing completeness and F-score.
- The ambiguity/repeated-structure filter is also heuristic; it reduced global F-score and should not be claimed as a successful module.
- View selection policies are lightweight proxies for diversity/overlap, not learned visibility prediction.
- Runtime excludes some dataset loading and setup overhead; report runtime per scene from the script, not Kaggle wall-clock setup time.

## Future Work

- Replace the proxy evaluator with an official mesh/laser-scan geometry evaluator on ScanNet++ scenes with full 3D ground truth.
- Train OARH, an occlusion-aware reliability head, to distinguish occluded-but-valid points from geometrically wrong points.
- Add MASt3R reciprocal matching and train RSDH, a repeated-structure match disambiguation head.
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
