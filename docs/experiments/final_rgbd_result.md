# Final RGB-D Result

Ngay cap nhat: 2026-06-22

## Final Setting

The project final setting is sparse-view RGB-D reconstruction:

```text
sparse posed RGB-D views
+ known camera intrinsics/extrinsics
+ MV-DUSt3R+ candidate reconstruction
+ source-depth / source-ray correction
-> improved point cloud reconstruction
```

RGB-only experiments are kept as baselines and negative evidence. The final
technical contribution is Run 30.

## Input Contract

At inference, Run 30 uses:

- selected sparse RGB frames;
- source depth maps from the same posed RGB-D frames;
- known camera intrinsics and extrinsics;
- MV-DUSt3R+ candidate points and confidence.

This is not an RGB-only claim.

## Pipeline

```text
select sparse posed RGB-D views
-> run MV-DUSt3R+ to get candidate geometry
-> keep the strong all-candidate baseline as reference
-> use source depth to anchor each candidate source ray to metric geometry
-> apply selective source-ray correction
-> evaluate overall, occlusion, and ambiguity held-out gates
```

## Run 30 Selected Policy

| Field | Value |
| --- | --- |
| selected_method | `rgbd_source_depth_selected` |
| best_baseline_method | `all_candidates` |
| selected internal policy | `rgbd_residual_ge_0.30` |
| mode | `residual` |
| alpha | `1.0` |
| residual_threshold_m | `0.30` |
| internal mean reconstruction F-score | `0.2327` |
| mean correction ratio | `0.4705` |
| gate margin | `0.005` |
| pass_all_limits | `1` |

## Validation Results

| Subset | Best baseline | RGB-D selected | Delta |
| --- | ---: | ---: | ---: |
| Overall | 0.1194 | 0.1753 | +0.0559 |
| Occlusion challenging | 0.1064 | 0.2522 | +0.1458 |
| Ambiguity challenging | 0.1623 | 0.3000 | +0.1377 |

## Test Results

| Subset | All candidates | RGB-D selected | Delta |
| --- | ---: | ---: | ---: |
| Overall | 0.1758 | 0.2764 | +0.1007 |
| Occlusion challenging | 0.2111 | 0.3146 | +0.1035 |
| Ambiguity challenging | 0.1621 | 0.2948 | +0.1327 |

## What Is Solved

- Limit 1, sparse-view/view-selection instability, was solved earlier by the
  Run 11 RGB-only baseline pipeline.
- Limit 2, occlusion/low-overlap valid geometry, is solved under Run 30's
  RGB-D/source-depth setting.
- Limit 3, repeated/wrong-depth ambiguity, is solved under Run 30's
  RGB-D/source-depth setting.

## What Is Not Claimed

- RGB-only learned extensions do not solve occlusion or repeated structures.
- OARH, RSDH, RAJAH, and monodepth correction are diagnostics, not final
  reconstruction modules.
- The project does not claim full benchmark generality beyond the controlled
  ScanNet-style subset.

## Required Final Wording

```text
RGB-only learned extensions did not pass reconstruction-level gates. After switching the inference contract to RGB-D, Run 30 uses input source depth maps with known camera poses/intrinsics for source-ray correction and passes the overall, occlusion, and ambiguity gates on held-out scenes.
```

## Reproduction Pointer

Run 30 script:

```text
scripts/kaggle/kaggle_run30_rgbd_source_depth_correction.py
```

Expected output files:

```text
correction_label_summary.csv
policy_selection.csv
metrics.csv
summary.csv
limit_summary.csv
gate_decision.csv
run_config.json
```

## Run 31 Coverage Validation

Run 31 does not introduce or select a new method. It freezes the Run 30 policy
`rgbd_residual_ge_0.30` and increases sparse-view coverage to:

- 30 scenes;
- 12 groups per scene;
- 360 groups total;
- 3, 4, and 5 views;
- hybrid and diversity-aware selection;
- two deterministic frame variants per configuration.

The run reports paired F-score deltas against `all_candidates` and
scene-cluster bootstrap confidence intervals. Results remain pending until the
Kaggle output is reviewed.

## Direct RGB-D Backprojection Baseline

Run 32 tests the boundary between Run 30 source-depth correction and direct
depth-only reconstruction. It directly back-projects valid depth pixels with
intrinsics and camera poses, attaches RGB colors, and evaluates a fixed
voxel-downsampled cloud without MV-DUSt3R+.

Direct RGB-D backprojection is not source-depth correction. It is a
depth-only/RGB-D baseline. Source-depth correction specifically refers to
correcting MV-DUSt3R+ candidate points using source depth residuals.

The comparison uses the same Run 30 validation/test groups and hard subsets.
Run 32 is complete. The pre-registered primary method is the voxelized
`direct_rgbd_backprojection`, and its gate outcome is
`run30_adds_value_over_direct`.

| Split / subset | Direct voxel RGB-D | Run 30 selected | Run 30 - direct |
| --- | ---: | ---: | ---: |
| Val overall | 0.0874 | 0.1753 | +0.0879 |
| Val occlusion | 0.1162 | 0.2522 | +0.1360 |
| Val ambiguity | 0.0998 | 0.3000 | +0.2002 |
| Test overall | 0.1359 | 0.2764 | +0.1405 |
| Test occlusion | 0.1196 | 0.3146 | +0.1951 |
| Test ambiguity | 0.1837 | 0.2948 | +0.1111 |

Important evaluator limitation: the current controlled GT cloud is built from
the same selected input depth maps. Direct backprojection therefore shares its
depth source with the target; this is not an independent official mesh or
laser-scan evaluation. This warning is concrete in Run 32: the additional
`direct_rgbd_backprojection_sampled` diagnostic reaches `0.8500` validation
overall and `0.8666` test overall because the sampled depth cloud is too close
to the proxy target. Treat that as evaluator-circularity evidence, not as a
new final method claim.
