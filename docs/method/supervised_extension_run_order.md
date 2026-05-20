# Supervised Extension Run Order

This document defines the next experiment phase after the heuristic ablations. The goal is to move from hand-written post-processing to a supervised occlusion- and ambiguity-aware sparse-view reconstruction pipeline.

Current best pipeline to keep:

```text
best view selection + tuned/fixed confidence + F0 baseline fusion
```

New target pipeline:

```text
Selected sparse views
-> MV-DUSt3R+ pretrained
-> candidate point cloud + confidence + pointmaps
-> learned visibility / reliability / ambiguity heads
-> learned filtering + fusion
-> refined point cloud
```

Because `Run 11` is already used for final fixed-threshold validation in this repo, this phase starts at `Run 12`.

## Module Names

### OARH: Occlusion-Aware Reliability Head

Input features per candidate point:

| Feature | Meaning |
| --- | --- |
| `C(p)` | original MV-DUSt3R+ confidence |
| `visible_view_count` | number of views where the point is visible and depth-consistent |
| `occluded_view_count` | number of views where the point is behind observed GT depth |
| `reprojection_error` | projection error into other views |
| `depth_disagreement` | predicted-vs-GT or cross-view depth disagreement |
| `view_baseline_angle` | angular baseline between source/reference views |
| `multi_view_agreement` | 3D agreement across pointmaps |
| `local_image_feature` | local image/backbone feature near the pixel |

Output:

```text
r(p) in [0, 1]
```

`r(p)` is the probability that candidate point `p` should be retained.

### RSDH: Repeated-Structure Disambiguation Head

Input features per candidate match:

| Feature | Meaning |
| --- | --- |
| `reciprocal_flag` | whether A->B and B->A agree |
| `feature_distance` | descriptor distance |
| `nearest_second_margin` | ambiguity margin between best and second-best matches |
| `reprojection_error` | geometric reprojection error |
| `3d_disagreement` | pointmap disagreement in 3D |
| `cycle_error` | A->B->C->A cycle error |
| `semantic_edge_cue` | optional cue for repeated structures |

Output:

```text
m(i, j) in [0, 1]
```

`m(i, j)` is the probability that a dense match is valid.

## Labels

Use GT geometry from ScanNet++ or the available ScanNet RGB-D proxy.

For each predicted point `X_pred`:

```text
d = distance(X_pred, nearest_GT_surface)
y_keep = 1 if d < tau_good
y_keep = 0 if d > tau_bad
```

Recommended thresholds:

```text
tau_good = 0.03m or 0.05m
tau_bad  = 0.10m
```

Visibility / occlusion labels:

```text
project X_pred into view u
compare z_pred with z_GT

if abs(z_pred - z_GT) <= tau_z:
    visible / consistent
elif z_pred > z_GT + tau_z:
    occluded
elif z_pred < z_GT - tau_z:
    floating / wrong geometry
```

Match labels:

```text
y_match = 1 if two pixels map to the same GT surface or nearby 3D point
y_match = 0 if visually similar pixels map to different/far GT 3D surfaces
```

## Losses

OARH:

```text
L_oarh =
    BCE(y_keep, r)
  + lambda_1 BCE(y_visible, v)
  + lambda_2 SmoothL1(depth_residual)
```

Use focal loss if positive/negative point labels are imbalanced.

RSDH:

```text
L_match = BCE(y_match, m)
```

Optional contrastive/triplet loss:

```text
positive = same GT surface or small 3D distance
negative = repeated-looking region but different GT surface
```

Partial fine-tuning:

```text
L_total =
    L_pointmap
  + lambda_1 L_conf_calibration
  + lambda_2 L_visibility
  + lambda_3 L_match_disambiguation
  + lambda_4 L_multiview_consistency_visible_only
```

Important rule:

```text
Do not enforce consistency through occluded views.
```

## New Experiment Runs

| Run | Name | Goal |
| ---: | --- | --- |
| 12 | Build GT labels | Generate `y_keep`, `y_visible`, `y_occluded`, `y_match` |
| 13 | Train OARH | Train occlusion-aware reliability head with frozen MV-DUSt3R+ |
| 14 | Evaluate OARH | Compare against confidence-only on occlusion-heavy scenes |
| 15 | Add MASt3R matches | Extract reciprocal matches, descriptor margin, cycle error |
| 16 | Train RSDH | Train repeated-structure match disambiguation head |
| 17 | Evaluate RSDH | Test on chairs/windows/cabinets/floor tiles or proxy repeated regions |
| 18 | Light fine-tune MV-DUSt3R+ | Unfreeze confidence head and last decoder blocks |
| 19 | Learned full evaluation | Compare B0, current best, OARH, RSDH, and learned full pipeline |

Minimum viable version if time is short:

```text
Run 12 - Build GT labels
Run 13 - Train OARH
Run 14 - Evaluate OARH
Run 16 - Train RSDH
Run 19 - Learned full evaluation
```

## Dataset Split

Use scene-level split only.

Recommended:

```text
train scenes: 60%
val scenes:   20%
test scenes:  20%
```

If data is small:

```text
train: 6 scenes
val:   2 scenes
test:  2 scenes
```

Rules:

- Test scenes must never appear in train.
- Thresholds and model selection are chosen on validation scenes only.
- Final numbers are reported on test scenes with fixed thresholds.

## Evaluation Tables

### Occlusion-heavy Subset

| Method | Occlusion-heavy F-score | Precision | Recall | Floating ratio |
| --- | ---: | ---: | ---: | ---: |
| B0 | TBD | TBD | TBD | TBD |
| Best view + conf | TBD | TBD | TBD | TBD |
| OARH | TBD | TBD | TBD | TBD |
| OARH + fine-tune | TBD | TBD | TBD | TBD |

### Repeated-structure Subset

| Method | Repeated-scene F-score | Match precision | False match rate | Cycle error |
| --- | ---: | ---: | ---: | ---: |
| B0 | TBD | TBD | TBD | TBD |
| Confidence-only | TBD | TBD | TBD | TBD |
| MASt3R reciprocal | TBD | TBD | TBD | TBD |
| RSDH | TBD | TBD | TBD | TBD |

## Implementation Priority

Most practical version:

```text
MV-DUSt3R+ frozen
+ MLP reliability head
+ MASt3R reciprocal matching
+ MLP match disambiguation head
```

Reliability MLP:

```text
input: 16-64 point features
hidden: 128 -> 64 -> 32
output: keep probability
loss: BCE or focal loss
```

Match MLP:

```text
input: match score, margin, reciprocal flag, reprojection error, 3D disagreement, cycle error
hidden: 128 -> 64
output: valid match probability
loss: BCE
```

Optional stronger version:

```text
MV-DUSt3R+ partial fine-tune
+ learned reliability head
+ learned match disambiguation head
```

Unfreeze only:

- confidence head
- last decoder block
- optionally cross-reference-view block

Do not full fine-tune the whole backbone first.

## Expected Contribution

| Limitation | Learned solution |
| --- | --- |
| Far-reference views | visibility-aware / diversity-aware view selection |
| Severe occlusion | supervised occlusion-aware reliability head |
| Repeated structures | learned match disambiguation with reciprocal/cycle consistency |

If the learned phase succeeds, the project becomes:

```text
Supervised Occlusion- and Ambiguity-Aware Sparse-View Reconstruction
```
