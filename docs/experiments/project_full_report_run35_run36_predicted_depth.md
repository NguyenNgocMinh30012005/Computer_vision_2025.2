# Runs 35-36: RGB-only Predicted-Depth Generalization Track

## Objective

Run 30 uses true RGB-D source depth and remains the validated RGB-D reference
method. Runs 35-36 test a more general input contract:

```text
selected sparse RGB views
+ MV-DUSt3R+ candidate pointmaps/confidence
+ monocular predicted depth from the same RGB images
+ known camera poses/intrinsics
-> predicted-depth correction
```

This is predicted-depth or pseudo-depth correction, not source-depth
correction.

After Run 37, the target setting is to keep this same correction structure but
replace the generic Depth Anything checkpoint with the fine-tuned depth
checkpoint:

```text
selected sparse RGB views
-> MV-DUSt3R+ candidate pointmaps/confidence
-> Run 37 fine-tuned depth estimator predicts source depth from RGB
-> back-project estimated depth with known pose/intrinsics
-> predicted-depth source-ray correction
-> reconstructed point cloud
```

This is RGB-only with respect to depth input at inference, but it still uses
known camera calibration and must pass reconstruction gates before replacing
the Run 30 RGB-D reference.

## Selected Depth Model

- Model: Depth Anything V2 Metric Indoor Small
- Checkpoint:
  `depth-anything/Depth-Anything-V2-Metric-Indoor-Small-hf`
- Model input: RGB image
- Intended output: indoor metric monocular depth

This checkpoint replaces the relative-depth heuristic used in Run 29.

## Run 35

Run 35 evaluates predicted depth against valid source RGB-D depth pixels.
Reported metrics:

- AbsRel, RMSE, MAE;
- delta1, delta2, delta3;
- scale-aligned RMSE and MAE;
- median and least-squares scale ratios;
- valid-pixel ratio.

Predicted maps are resized to source-depth resolution and cached as compressed
float16 arrays. A single global median scale is estimated from train-fit scenes
only. Validation and test depth never select this scale.

Completed Run 35 results:

| Split | Frames | AbsRel | RMSE (m) | MAE (m) | delta1 | delta2 | delta3 | Scale-aligned RMSE (m) | Scale-aligned MAE (m) |
| --- | ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: |
| Train | 85 | 0.2155 | 0.4751 | 0.3997 | 0.6680 | 0.9299 | 0.9827 | 0.2612 | 0.1688 |
| Validation | 43 | 0.2207 | 0.4773 | 0.4115 | 0.7175 | 0.9369 | 0.9900 | 0.2175 | 0.1088 |
| Test | 30 | 0.1861 | 0.4642 | 0.3525 | 0.7736 | 0.9404 | 0.9711 | 0.2786 | 0.1614 |

The model checkpoint SHA is
`8078d68a9c75a972131914f6afd0c1723be0da7f`. The train-fit global median
scale is `0.912263`. Raw metric depth is useful but imperfect; scale alignment
reduces MAE substantially, so both modes proceed to Run 36.

Run 35 processed 158 unique frames from 30 scenes and recorded
`68.21 s` of model/evaluation runtime after setup.

## Run 36

Run 36 consumes the Run 35 depth cache and evaluates:

- `mvdust3r_rgb_only_all_candidates`;
- `mvdust3r_confidence_fixed`;
- `predicted_depth_direct_backprojection`;
- `predicted_depth_correction_raw`;
- `predicted_depth_correction_scale_aligned`;
- `run30_rgbd_source_depth_selected` as a reference only.

For candidate \(X\) and predicted-depth backprojection \(X_p\):

\[
r=\|X-X_p\|_2,
\qquad
X_{\mathrm{corr}}=
\begin{cases}
(1-\alpha)X+\alpha X_p,&r\geq\tau_{\mathrm{pred}},\\
X,&r<\tau_{\mathrm{pred}}.
\end{cases}
\]

Run 36 selects \(\tau_{\mathrm{pred}}\), \(\alpha\), and raw/global-scale mode
on held-out train scenes. Test scenes do not participate in selection.

The selected policy is:

```text
depth_scale_mode = global_scale
tau_pred = 0.5
alpha = 0.25
```

### Reconstruction Results

| Split | RGB-only all candidates | Fixed confidence | Correction raw | Correction scale-aligned | Direct predicted depth | Run 30 true RGB-D |
| --- | ---: | ---: | ---: | ---: | ---: | ---: |
| Validation F-score | 0.1231 | 0.1179 | 0.0977 | 0.1012 | 0.1747 | 0.1753 |
| Test F-score | 0.1744 | 0.1686 | 0.1433 | 0.1465 | 0.1894 | 0.2764 |

Normalized metrics support the same result:

| Split | Method | Normalized distance | DAc@0.2 normalized |
| --- | --- | ---: | ---: |
| Validation | RGB-only all candidates | 0.1154 | 0.8467 |
| Validation | Selected correction | 0.1311 | 0.7879 |
| Validation | Direct predicted depth | 0.0779 | 0.9446 |
| Test | RGB-only all candidates | 0.1196 | 0.8685 |
| Test | Selected correction | 0.1272 | 0.8025 |
| Test | Direct predicted depth | 0.0993 | 0.8890 |

The selected correction fails every gate. Its validation deltas versus
RGB-only are `-0.0218` overall, `-0.0421` on occlusion-challenging groups, and
`-0.0310` on ambiguity-challenging groups. The direct predicted-depth baseline
is stronger than RGB-only overall and on validation hard subsets, but it
regresses on test occlusion (`0.1840` versus `0.2120`) and remains below Run 30
on test (`0.1894` versus `0.2764`).

Run 32 direct true-depth backprojection remains a reference with a different
point-selection/voxel protocol (`0.0874` validation and `0.1359` test), so it
is not inserted into the strict Run 36 gate table.

Run 36 contains 216 metric rows: 36 groups, 18 validation groups, 18 test
groups, and six methods. Its recorded runtime is `10067.13 s` (about
`2.80 h`).

## Local Result Artifacts

```text
downloads/kaggle_run35_predicted_depth_quality_diagnostic/
  outputs/run_35_predicted_depth_quality_diagnostic/

downloads/kaggle_run36_predicted_depth_correction/
  outputs/run_36_predicted_depth_correction/
```

The decisive files are `depth_summary.csv`, `summary.csv`,
`limit_summary.csv`, `policy_selection.csv`, `gate_decision.csv`, and
`run_config.json`.

## Claim Boundary

Run 36 is an RGB-only image-input extension with predicted monocular depth and
known pose/intrinsics. It is more general than RGB-D Run 30 because it does not
use true source depth for inference, but it is not the fully pose-free
MV-DUSt3R+ paper setting.

The final Run 30 claim is unchanged. Run 36 shows that the predicted metric
depth prior is useful, but blending it into MV-DUSt3R+ candidates with the
tested residual rule is harmful. The result does not solve RGB-only occlusion
or repeated/wrong-depth ambiguity.

Additional limitations:

- Run 36 still assumes known camera poses and intrinsics.
- The evaluator uses depth-derived proxy geometry, so direct depth
  backprojection has an evaluation-circularity advantage.
- Run 35/36 cover the controlled 30-scene sparse-view protocol, not the full
  official ScanNet++ benchmark.
