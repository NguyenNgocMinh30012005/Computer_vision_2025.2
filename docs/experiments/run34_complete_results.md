# Run 34 Complete Results

## 1. Run Identification

- Run: `run_34_qualitative_3d_exports`
- Final successful Kaggle version: v4
- Status: completed
- Runtime: 322.33 seconds
- Kaggle: <https://www.kaggle.com/code/nguynnminh/mv-dust3r-run-34-qualitative-3d-exports>
- Local output:
  `../../downloads/kaggle_run34_qualitative_3d_exports_v4/outputs/run_34_qualitative_3d_exports/`

Run 34 exports comparable 3D outputs and evaluates three reconstruction
families:

1. `mvdust3r_rgb_only_all_candidates`: MV-DUSt3R+ RGB-only baseline.
2. `rgbd_source_depth_selected`: Run 30 source-depth correction.
3. `direct_rgbd_backprojection`: Run 32-style direct RGB-D baseline.

The run uses three test groups, 3,500 predicted points per method, 50,000 proxy
GT points, and the metric-space threshold of 0.05 m.

## 2. Evaluated Groups

| Group | Views | Policy | Occlusion proxy | Ambiguity proxy |
| --- | ---: | --- | ---: | ---: |
| `scene0012_00_4_diversity_aware` | 4 | diversity-aware | 0.1200 | 0.9503 |
| `scene0012_00_5_diversity_aware` | 5 | diversity-aware | 0.2077 | 0.9560 |
| `scene0012_01_4_diversity_aware` | 4 | diversity-aware | 0.4943 | 0.9623 |

## 3. Metrics

The conventional metrics are:

- Accuracy: mean prediction-to-GT distance; lower is better.
- Completeness: mean GT-to-prediction distance; lower is better.
- Precision: predicted-point fraction within 0.05 m of GT; higher is better.
- Recall: GT-point fraction within 0.05 m of prediction; higher is better.
- F-score: harmonic mean of precision and recall; higher is better.
- Chamfer: accuracy plus completeness; lower is better.

For a point cloud \(X\), Run 34 independently removes translation and isotropic
RMS scale:

\[
\hat{X} =
\frac{X-\bar{X}}
{\sqrt{\frac{1}{|X|}\sum_{x\in X}\|x-\bar{X}\|_2^2}}.
\]

The symmetric normalized distance is:

\[
\mathrm{ND}_{sym}(P,G)=\frac{1}{2}
\left[
\frac{1}{|P|}\sum_{p\in \hat P}d(p,\hat G)
+
\frac{1}{|G|}\sum_{g\in \hat G}d(g,\hat P)
\right].
\]

Lower ND is better. Normalized DAc@0.2 is:

\[
\mathrm{DAc}@0.2 =
\frac{|\{p\in\hat P:d(p,\hat G)\leq0.2\}|}{|P|}.
\]

Higher DAc@0.2 is better.

## 4. Aggregate Results

Means over the three matched 3,500-point groups:

| Method | Accuracy (lower) | Completeness (lower) | Precision (higher) | Recall (higher) | F-score (higher) | Chamfer (lower) | ND (lower) | DAc@0.2 (higher) |
| --- | ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: |
| Run 30 RGB-D selected | **0.0841** | **0.0930** | **0.4318** | 0.2753 | 0.3305 | **0.1770** | **0.0515** | **0.9841** |
| Direct RGB-D | 0.1562 | 0.1693 | 0.4073 | **0.2836** | **0.3335** | 0.3255 | 0.0926 | 0.8617 |
| MV-DUSt3R+ RGB-only | 0.2273 | 0.2284 | 0.2501 | 0.2031 | 0.2231 | 0.4558 | 0.1308 | 0.7967 |

Run 30 is best on accuracy, completeness, precision, Chamfer, normalized
distance, and DAc@0.2. Direct RGB-D has a slightly higher mean F-score on this
small three-group subset because it performs strongly on the two
`scene0012_00` groups. This does not reverse the full Run 30 versus Run 32
result, where Run 30 test F-score is 0.2764 and direct RGB-D test F-score is
0.1359.

## 5. Per-Group Results

### 5.1 `scene0012_00_4_diversity_aware`

| Method | Accuracy (lower) | Completeness (lower) | Precision (higher) | Recall (higher) | F-score (higher) | Chamfer (lower) | ND (lower) | DAc@0.2 (higher) |
| --- | ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: |
| MV-DUSt3R+ RGB-only | 0.2873 | 0.2835 | 0.0426 | 0.0290 | 0.0345 | 0.5708 | 0.1637 | 0.8054 |
| Run 30 RGB-D selected | 0.0710 | 0.0765 | 0.5149 | 0.2229 | 0.3111 | **0.1474** | **0.0423** | 0.9880 |
| Direct RGB-D | **0.0653** | 0.0833 | **0.5340** | **0.3744** | **0.4402** | 0.1486 | 0.0426 | **0.9974** |

### 5.2 `scene0012_00_5_diversity_aware`

| Method | Accuracy (lower) | Completeness (lower) | Precision (higher) | Recall (higher) | F-score (higher) | Chamfer (lower) | ND (lower) | DAc@0.2 (higher) |
| --- | ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: |
| MV-DUSt3R+ RGB-only | 0.1395 | 0.1234 | **0.6511** | **0.5531** | **0.5981** | 0.2629 | 0.0809 | 0.8491 |
| Run 30 RGB-D selected | 0.0901 | **0.1025** | 0.4709 | 0.3709 | 0.4150 | **0.1926** | **0.0593** | **0.9720** |
| Direct RGB-D | **0.0840** | 0.1099 | 0.6483 | 0.4630 | 0.5402 | 0.1939 | 0.0597 | 0.9431 |

This is the selected Run 30 regression example: RGB-only has the highest
F-score, while Run 30 still improves metric distance, Chamfer, ND, and
DAc@0.2. It demonstrates why no claim is made that Run 30 wins every group.

### 5.3 `scene0012_01_4_diversity_aware`

| Method | Accuracy (lower) | Completeness (lower) | Precision (higher) | Recall (higher) | F-score (higher) | Chamfer (lower) | ND (lower) | DAc@0.2 (higher) |
| --- | ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: |
| MV-DUSt3R+ RGB-only | 0.2552 | 0.2783 | 0.0566 | 0.0272 | 0.0367 | 0.5336 | 0.1477 | 0.7354 |
| Run 30 RGB-D selected | **0.0911** | **0.1000** | **0.3097** | **0.2319** | **0.2652** | **0.1911** | **0.0529** | **0.9923** |
| Direct RGB-D | 0.3192 | 0.3148 | 0.0397 | 0.0136 | 0.0202 | 0.6340 | 0.1756 | 0.6446 |

This is the strongest hard-group evidence in Run 34. It has the largest
occlusion proxy ratio, and Run 30 wins every reported metric.

## 6. Dense Scene Exports

These exports preserve scene-like camera and image geometry for visual
inspection. They are not used for the fair 3,500-point metric tables.

| Group | Dense method | Exported points | Correction/raw points |
| --- | --- | ---: | ---: |
| `scene0012_00_4_diversity_aware` | RGB-only dense | 200,704 | correction 0.0000 |
| `scene0012_00_4_diversity_aware` | Run 30 RGB-D dense | 200,704 | correction 0.5349 |
| `scene0012_00_4_diversity_aware` | Direct RGB-D dense | 220,000 | 1,153,358 raw |
| `scene0012_00_5_diversity_aware` | RGB-only dense | 250,880 | correction 0.0000 |
| `scene0012_00_5_diversity_aware` | Run 30 RGB-D dense | 250,880 | correction 0.4361 |
| `scene0012_00_5_diversity_aware` | Direct RGB-D dense | 220,000 | 1,454,872 raw |
| `scene0012_01_4_diversity_aware` | RGB-only dense | 200,704 | correction 0.0000 |
| `scene0012_01_4_diversity_aware` | Run 30 RGB-D dense | 200,704 | correction 0.5851 |
| `scene0012_01_4_diversity_aware` | Direct RGB-D dense | 220,000 | 1,175,338 raw |

Each group directory contains:

```text
00_original_mvdust3r_inference_scene.glb
01_mvdust3r_rgb_only_all_candidates.glb
02_rgbd_source_depth_selected.glb
03_direct_rgbd_backprojection.glb
dense_scene_exports/
```

## 7. Main Interpretation

1. Run 30 produces the best normalized geometry on average:
   ND = 0.0515 and DAc@0.2 = 0.9841.
2. It substantially improves over RGB-only in the high-occlusion,
   high-ambiguity group.
3. Direct RGB-D can be strong on some groups, but it is unstable on the
   difficult `scene0012_01` group.
4. The additional normalized metrics support the shape-level advantage of
   source-depth correction after removing global translation and isotropic
   scale.
5. Scientific method selection remains based on the broader Run 30, Run 32,
   and Run 33 gates, not only these three qualitative groups.

## 8. Limitations

- Only three selected test groups are included in Run 34.
- Proxy GT is constructed from selected input depth maps, not an independent
  official ScanNet++ laser-scan mesh.
- Normalization removes translation and isotropic scale, but not rotation.
- DAc@0.2 is prediction-side only and does not directly measure completeness.
- Dense scene exports are for visualization and have unequal point counts.
- Meshes generated from these clouds are visualization surfaces and must not
  replace point-cloud metrics.

## 9. Source Files

- `qualitative_3d_manifest.csv`: all nine fair-comparison rows.
- `normalized_metric_summary.csv`: aggregate ND and DAc@0.2.
- `dense_scene_manifest.csv`: dense visualization exports.
- `run_config.json`: run configuration and metric contract.
- `mv-dust3r-run-34-qualitative-3d-exports.log`: complete Kaggle log.
