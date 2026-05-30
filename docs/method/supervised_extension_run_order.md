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
| 12 | OARH proxy | Train and evaluate a frozen-backbone reliability MLP against confidence-only filtering |
| 13 | RSDH proxy | Train and evaluate a proxy repeated-structure match-validity MLP |
| 14 | Validation-gated learned pipeline | Use OARH only when it wins on validation, otherwise fall back to confidence-only |
| 15 | Add real MASt3R matches | Extract reciprocal matches, descriptor margin, cycle error |
| 16 | RSDH with descriptor/cycle features | Re-evaluate match disambiguation beyond nearest-surface proxy features |
| 17 | Light fine-tune MV-DUSt3R+ | Unfreeze confidence head and last decoder blocks only if validation justifies it |
| 18 | Learned full evaluation | Compare B0, current best, gated OARH, RSDH, and learned full pipeline |
| 19 | Supervised label cache | Build scene-level visibility, occlusion, wrong-depth, and match labels for OARH v2 / RSDH v2 |
| 20 | Occlusion and ambiguity subset mining | Select occlusion-heavy and repeated/hard-negative view groups from the label cache |
| 21 | OARH v2 multitask | Train keep, visibility-class, and depth-residual heads using the Run 20 OARH labels |
| 22 | OARH v2 reconstruction integration | Evaluate learned visibility-aware fusion on occlusion-heavy held-out groups |
| 23 | Reconstruction candidate calibration | Train reliability from actual MV-DUSt3R candidate points after Run 22 proxy-to-reconstruction failure |
| 24 | RSDH v2 image-only features | Train match validity from MASt3R descriptors, margins, reciprocal flags, and cycle cues without GT-derived inference features |
| 25 | RSDH v2 reconstruction integration | Use validated matches in reconstruction/fusion and evaluate repeated-structure held-out groups |
| 26 | RSDH diagnostic gate | Compare RSDH against all-candidate and confidence top-k baselines with exact top-k masks |
| 27 | Joint candidate acceptance | Train one reconstruction-candidate head that combines confidence, self-geometry support, and RSDH image-match scores |
| 28 | Final held-out evaluation | Compare B0, fixed-confidence final, and learned full pipeline on unseen scenes |

Minimum viable version if time is short:

```text
Run 12 - OARH proxy
Run 13 - RSDH proxy
Run 14 - Validation-gated learned pipeline
Run 18 - Learned full evaluation
```

Current note after Runs 12--14: OARH is mixed. A validation gate avoids the
large 2/3/5-view regressions, but still slightly overfits at 4 views on the
held-out scene. RSDH is promising but still proxy-based until real MASt3R
descriptors/cycle features are included.

Current execution note after submitting Runs 15--18:

- Run 15 is the real reciprocal-feature extraction step. It attempts MASt3R
  first and records an explicit fallback backend if Kaggle cannot install or
  load the full MASt3R stack.
- Run 16 trains RSDH on descriptor, margin, reciprocal, 3D-disagreement, and
  cycle-proxy features produced by Run 15. It should consume the successful Run
  15 `match_features.csv` as a Kaggle kernel source when available, so it does
  not depend on receiving a T4x2 allocation again.
- Run 17 is intentionally a decision gate. It should skip light backbone
  fine-tuning unless the validation-gated learned pipeline beats the verified
  confidence-only final policy by a meaningful margin.
- Run 18 summarizes the learned extension honestly: the final deployable policy
  remains the verified confidence-only reconstruction unless the new learned
  runs clearly improve validation and held-out metrics.

Current note after Runs 15--16 completed:

- Run 15 successfully used MASt3R, not the ORB fallback, and produced reciprocal
  match features for train-proxy `scene0000_00` and held-out `scene0000_01`.
- Run 16 reached near-perfect held-out proxy match F1 from those features.
  However, this is an upper-bound/sanity result because labels and some
  features use GT-depth-derived 3D disagreement. It supports the RSDH direction
  but should not be claimed as a solved image-only repeated-structure module.

Current Phase 3 execution note:

- Run 19 is the first stricter run for solving the remaining two limitations.
  It does not train a model yet. Instead, it writes `label_cache.csv`,
  `label_summary.csv`, `view_group_manifest.csv`, `scene_split.csv`, and
  `occlusion_heavy_groups.csv` so later OARH v2/RSDH v2 experiments can train
  from explicit visibility, occlusion, floating/wrong-depth, and match labels.
- Run 20 consumes the Run 19 kernel-source output and writes focused manifests:
  `subset_group_manifest.csv`, `final_eval_group_manifest.csv`,
  `oarh_v2_balanced_labels.csv`, and `rsdh_v2_hard_negative_labels.csv`.
  This keeps later training/evaluation concentrated on occlusion-heavy,
  low-overlap/far, and hard-negative cases instead of generic easy pixels.
- Run 21 trains the first Phase 3 OARH v2 multitask head from the Run 20
  balanced labels. It predicts keep/reject, visibility class, and depth
  residual, while excluding direct label-leakage inputs such as
  `candidate_type` and `visibility_label`.
- Run 22 integrates the Run 21 checkpoint into reconstruction candidate
  filtering. It compares fixed-confidence point selection against several OARH
  v2 thresholds and an OARH-plus-confidence guard on Run 20 final-eval groups.
- Run 22 fails the validation gate: fixed confidence reaches 0.6716 validation
  mean F-score while the best learned OARH v2 variant reaches only 0.2160.
  Test shows the same pattern, 0.6033 fixed confidence versus 0.1936 best
  learned. This means the Run 21 proxy labels do not transfer to real
  reconstruction candidate filtering.
- Run 23 retrains reliability on actual MV-DUSt3R candidate points with
  GT-geometry labels and prediction-only features. It treats the learned head as
  a ranking policy and selects it only if validation reconstruction F-score wins.
- Run 23 is close but still fails the gate: the best learned ranking keeps
  99.5% of points and reaches 0.6618 validation F-score, below fixed confidence
  at 0.6674. It reduces the Run 22 collapse but does not solve occlusion.
- Run 24 therefore moves to RSDH v2 for repeated-structure ambiguity. It trains
  from Run 20 hard-negative match labels using image-only patch, coordinate, and
  view-policy features, with no GT-depth residuals or label-leakage fields as
  inference inputs.
- Run 24 passes the image-only match-validity gate: validation F1 improves from
  the best image-only baseline at 0.6212 to the learned RSDH v2 MLP at 0.6954,
  and test F1 improves from 0.5517 to 0.6596.
- Run 25 integrates that checkpoint into reconstruction candidate scoring. It
  compares fixed confidence with RSDH threshold, RSDH top-ratio, and combined
  confidence/RSDH ranking policies, then keeps the learned policy only if
  validation reconstruction F-score wins.
- Run 25 does not pass that gate: the validation gain is only +0.0035 F-score,
  below the 0.005 margin, and the best test policies keep all candidates.
- Run 26 therefore adds all-candidate and confidence top-k baselines with exact
  top-k tie handling, then gates RSDH against the best non-learned baseline.
- Run 26 confirms the reconstruction-level gain is not learned: validation
  selects `all_candidates` at 0.1463 F-score, the best learned RSDH policy only
  ties it by keeping all candidates, and the delta over the best baseline is
  0.0000 below the 0.005 margin. Keep RSDH v2 out of the final reconstruction
  policy.
- Run 27 starts a new solver branch for the remaining occlusion and
  repeated/wrong-match limits. It trains on actual MV-DUSt3R candidates and
  uses only inference-available features: confidence, point layout,
  cross-view self-support, and Run 24 image-only RSDH scores. It must beat the
  best non-learned candidate-retention baseline to pass.
- A strong final claim requires Runs 21--28 to show held-out reconstruction
  improvement on occlusion-heavy and repeated-structure subsets, not only high
  proxy classification F1.

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
