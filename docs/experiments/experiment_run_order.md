# Experiment Run Order

File nay quy dinh thu tu chay thi nghiem cho de tai sparse-view 3D reconstruction dua tren MV-DUSt3R+. Muc tieu la chay theo tung lop: dau tien kiem tra pipeline co hoat dong, sau do tao baseline manh, roi moi them view selection, fusion, occlusion va repeated-structure filtering.

## Global Protocol

Truoc khi chay baseline chinh thuc, can chot mot file config co dinh, vi du:

```text
configs/experiment_protocol.yaml
```

Config nay nen ghi ro:

- Dataset/subset: 5-10 ScanNet++ scenes nho, co geometry ground truth.
- Khong dung `nvs_test` de danh gia reconstruction geometry, vi split nay khong co day du 3D information nhu mesh hoac iPhone depth map.
- View counts: `2, 3, 4, 5`.
- Resolution anh dau vao.
- Random seed va cach sample views.
- Split scenes: sanity / validation-ablation / final-test.
- Metrics: accuracy, completeness/recall, precision, F-score, Chamfer distance neu kip, runtime per scene, number of retained points.
- Output folder convention:

```text
  outputs/
  run_00_sanity_check/
  run_01_evaluation_pipeline/
  run_02_baseline_b0/
  run_03_baseline_b1_confidence/
  run_04_view_selection/
  run_05_basic_fusion/
  run_06_occlusion_fusion/
  run_07_repeated_structure/
  run_08_full_pipeline/
  run_09_final_stress_test/
  run_10_sensitivity_visualization/
  run_11_final_validation_3seeds/
  run_12_supervised_reliability/
  run_13_match_disambiguation/
  run_14_validation_gated_learned_pipeline/
  run_15_mast3r_reciprocal_features/
  run_16_rsdh_descriptor_cycle/
  run_17_light_finetune_decision/
  run_18_learned_full_evaluation_summary/
  run_19_supervised_label_cache/
  run_20_occlusion_ambiguity_subset_mining/
  run_21_oarh_v2_multitask/
  run_22_oarh_v2_reconstruction/
  run_23_repeated_hard_negative_dataset/
  run_24_rsdh_v2_image_only/
  run_25_rsdh_v2_reconstruction/
  run_26_rsdh_v2_diagnostic_gate/
  run_27_conditional_light_finetune/
  run_28_final_heldout_learned_eval/
```

Chi thay doi dung bien dang duoc test trong tung run. Cac thanh phan khac phai giu nguyen de ket qua ablation co y nghia.

## Run 0 - Sanity Check

Chay demo/checkpoint MV-DUSt3R+ tren 1-2 scene nho.

Muc tieu khong phai lay so dep, ma la tra loi duoc:

```text
input images -> MV-DUSt3R+ -> point cloud -> visualize duoc chua?
```

Khong chay metric o buoc nay.

Can luu:

- Input image list.
- Checkpoint/model name.
- Predicted point cloud.
- Visualization screenshot hoac viewer output.
- Runtime tham khao.

Dieu kien qua buoc:

- Point cloud render/visualize duoc.
- Pipeline inference khong loi.
- Nhom hieu ro input/output path.

## Run 1 - Evaluation Pipeline

Truoc khi them module, phai co script do:

```text
pred_point_cloud vs gt_point_cloud
```

Metrics bat buoc:

- Accuracy.
- Completeness / recall.
- Precision.
- F-score.
- Runtime per scene.
- Number of retained points.

Metrics nen co neu kip:

- Chamfer distance.
- Consistency theo multi-view neu da co script on dinh.

Voi ScanNet++, nen dung split/subset co geometry ground truth. Dataset co laser scans, DSLR images va iPhone RGB-D, nhung `nvs_test` khong phu hop de danh gia reconstruction geometry neu khong co mesh/depth ground truth.

Can luu:

- Script evaluation.
- Huong dan chay evaluation.
- Mot file ket qua mau tren 1-2 scenes.
- Log runtime va so diem giu lai.

Dieu kien qua buoc:

- Evaluation script chay duoc tu dau den cuoi.
- Cung mot prediction va ground truth cho ra metric lap lai duoc.
- Output metric duoc ghi thanh CSV/JSON de so sanh ve sau.

## Run 2 - Baseline B0

Chay baseline mac dinh:

```text
B0 = MV-DUSt3R+ default + random views
```

View counts:

- 2 views.
- 3 views.
- 4 views.
- 5 views.

Them 4 views du cho proposal co the chi ghi 2/3/5 views, vi 4 views la moc trung gian huu ich de thay trend.

Output toi thieu:

| Method | 2 views | 3 views | 4 views | 5 views |
| --- | --- | --- | --- | --- |
| MV-DUSt3R+ random | metric | metric | metric | metric |

Nen bao cao rieng tung metric:

- Accuracy table.
- Completeness/recall table.
- Precision table.
- F-score table.
- Runtime table.
- Retained-points table.

Dieu kien qua buoc:

- Co bang B0 day du cho tat ca view counts.
- Co logs de reproduce random seed/view selection.

## Run 3 - Baseline B1: Confidence Threshold

Chay baseline manh hon bang confidence threshold tuning:

```text
B1 = MV-DUSt3R+ + confidence threshold tuning
```

Sweep de xuat:

```text
conf_thres = 0.1, 0.2, 0.3, 0.5, 0.7
```

Buoc nay rat quan trong. Neu full pipeline chi hon B0 nhung khong hon B1, dong gop se yeu. Vi vay coi B1 la baseline manh.

Can luu:

- Bang metric theo `conf_thres`.
- Best threshold theo validation scenes.
- Visualization cho threshold qua thap va qua cao.

Dieu kien qua buoc:

- Chon duoc confidence threshold tot nhat cho cac run tiep theo.
- Ghi ro trade-off precision vs completeness.

## Run 4 - View Selection

Giu reconstruction/fusion giong baseline B1, chi thay cach chon view.

Policies:

```text
V0 = random
V1 = diversity-aware
V2 = predicted-overlap / coverage-aware
V3 = hybrid diversity + overlap
```

Khong them occlusion hay ambiguity o buoc nay.

Muc tieu:

- Chung minh view selection giup giam far-reference cases.
- So sanh random, coverage-aware, diversity-aware va hybrid policies.
- Chon policy co F-score cao hon random va runtime khong tang qua nhieu.

Can luu:

- View indices duoc chon cho tung scene.
- Metric theo policy va view count.
- Runtime overhead cua tung policy.
- Mot so visualization far-reference cases.

Dieu kien qua buoc:

- Chon duoc best view policy.
- Neu policy tot nhat khong phai hybrid, van ghi ro ly do dua tren metric/runtime.

## Run 5 - Basic Fusion

Sau khi chon duoc view policy tot nhat, bat dau them scoring/fusion tu nhe den nang.

Settings:

```text
F0 = baseline fusion
F1 = C only
F2 = C + S_mv
F3 = C + S_mv + M
```

Chua dung occlusion `O(p)` va ambiguity `A(p)` o buoc nay.

Tin hieu can kiem tra:

- `C(p)`: confidence.
- `S_mv(p)`: multi-view support.
- `M(p)`: multi-view agreement.

Muc tieu la biet tin hieu nao that su co ich truoc khi chay full formula.

Can luu:

- Metric theo F0/F1/F2/F3.
- Runtime overhead.
- Retained points.
- Visualization truoc/sau fusion.

Dieu kien qua buoc:

- Xac dinh duoc fusion setting tot nhat khong dung occlusion/ambiguity.
- Ghi ro signal nao tang precision, signal nao anh huong completeness.

## Run 6 - Occlusion-Aware Fusion

Chay occlusion-aware ablation:

```text
O0 = C + S_mv + M
O1 = C + S_mv + M + O
```

Nen chon rieng cac scene/case co occlusion nang. Neu metric tong the tang it nhung visualization dep hon o vung occlusion thi van co gia tri bao cao.

Can visualize:

- Baseline point cloud.
- Without occlusion filtering.
- With occlusion filtering.

Can luu:

- Metric global.
- Metric rieng cho occlusion-heavy cases.
- Figure so sanh vung bi che khuat.
- Runtime overhead cua occlusion term.

Dieu kien qua buoc:

- Ket luan ro occlusion term co giup precision/F-score hoac qualitative clarity khong.
- Ghi ro truong hop nao occlusion filtering lam giam completeness.

## Run 7 - Repeated-Structure Filtering

Day la module rui ro nhat, nen chay sau cung.

Subset nen co:

- Chairs.
- Windows.
- Cabinets.
- Repeated wall/floor patterns.

Ablation:

```text
A0 = no ambiguity filtering
A1 = U only = 1 - C
A2 = U + R
A3 = S + U + R
A4 = S + U + R + anchor support
```

Khong nen ky vong module nay luon tang metric global. No co the tang precision nhung giam completeness.

Can bao cao:

- Precision tang bao nhieu.
- Recall/completeness giam bao nhieu.
- F-score co tang khong.
- Case visualization co sach hon khong.

Can luu:

- Metric tren repeated-structure subset.
- Case study figures.
- Error examples khi filtering qua manh.

Dieu kien qua buoc:

- Chon duoc ambiguity setting tot nhat hoac quyet dinh khong dung neu lam giam F-score qua nhieu.
- Co giai thich precision-completeness trade-off.

## Run 8 - Full Pipeline

Sau khi biet module nao co ich, moi chay full pipeline:

```text
Full = best view selection
     + best confidence/multi-view fusion
     + occlusion filtering
     + repeated-structure filtering
```

Bang so sanh cuoi:

| ID | Method |
| --- | --- |
| B0 | MV-DUSt3R+ default |
| B1 | MV-DUSt3R+ + confidence threshold tuning |
| V | B1 + best view selection |
| F | V + multi-view fusion |
| O | F + occlusion-aware fusion |
| A | F + ambiguity filtering |
| Full | V + F + O + A |

Can luu:

- Full metrics tren validation/final subset.
- Runtime per scene.
- Retained points.
- Qualitative comparison figures.
- Error cases.

Dieu kien qua buoc:

- Full pipeline tot hon baseline manh B1 tren metric chinh hoac co trade-off duoc giai thich tot.
- Co bang ablation ro rang cho tung module.

## Run 9 - Final Stress Test

Chay full pipeline tren tat ca settings:

- 2 views.
- 3 views.
- 4 views.
- 5 views.

Chia case:

- Far-reference.
- Occlusion-heavy.
- Repeated-structure.
- Normal scenes.

Bang cuoi nen bam theo proposal: accuracy, completeness, consistency, runtime va case-based analysis tren far-reference, occlusion-heavy, repeated-structure scenarios.

Can luu:

- Final quantitative tables.
- Per-case-group tables.
- Sensitivity sweep neu con thoi gian.
- Runtime analysis.
- Visualization error cases.
- Figures san sang dua vao report.

Dieu kien hoan tat:

- Co ket qua tren 2/3/4/5 views.
- Co bang so sanh B0, B1, V, F, O, A, Full.
- Co visualization cho thanh cong va that bai.
- Co nhan xet ro module nao dang gia tri nhat.

## Run 10 - Sensitivity + Visualization

Chay sensitivity sau khi da co full pipeline tot nhat. Khong dung buoc nay de thay doi module, chi dung de chon threshold bao cao va tao figure.

Can chay:

- Sweep confidence percentile tren full pipeline.
- Plot F-score theo confidence threshold.
- Plot F-score/runtime theo view count.
- Plot case-summary heatmap tu Run 9.

Can luu:

- `confidence_sensitivity.csv`.
- `best_sensitivity_by_view_count.csv`.
- `fig_conf_sensitivity.png`.
- `fig_runtime_vs_views.png`.
- `fig_case_summary.png`.

Dieu kien qua buoc:

- Chon duoc fixed confidence threshold cho tung view count.
- Figure san sang dua vao report.

## Run 11 - Final Validation With Fixed Thresholds

Chay validation cuoi sau khi da chon threshold tu Run 10. Khong tune lai tren test.

Can chay:

- B0 vs Final voi 3 random seeds.
- Final dung threshold co dinh:
  - 2 views: `3.0`.
  - 3 views: `0.2`.
  - 4 views: `3.0`.
  - 5 views: `0.5`.
- Qualitative comparison B0 vs Final cho 3 case: normal, far-reference, occlusion-heavy.

Can luu:

- `metrics.csv`.
- `summary_final_vs_b0_3seeds.csv`.
- `fig_qualitative_b0_vs_final.png`.

Dieu kien qua buoc:

- Bao cao mean/std F-score cua B0 va Final theo view count.
- Co figure qualitative B0 vs Final.
- Ket luan trung thuc module nao co ich va module nao fail.

## Recommended Execution Summary

Thu tu chay ngan gon:

```text
Run 0  - Sanity check
Run 1  - Evaluation pipeline
Run 2  - Baseline B0
Run 3  - Baseline B1: confidence threshold
Run 4  - View selection
Run 5  - Basic fusion
Run 6  - Occlusion-aware fusion
Run 7  - Repeated-structure filtering
Run 8  - Full pipeline
Run 9  - Final stress test
Run 10 - Sensitivity + visualization
Run 11 - Final validation with fixed thresholds
```

## Phase 2 - Supervised Learned Extension

Sau Run 11, neu co compute de training/fine-tune, khong nen fine-tune full MV-DUSt3R+ ngay. Huong hop ly hon la train module supervised tren output cua MV-DUSt3R+:

```text
Selected sparse views
-> MV-DUSt3R+ pretrained
-> candidate point cloud + confidence + pointmaps
-> learned visibility / reliability / ambiguity heads
-> learned filtering + fusion
-> refined point cloud
```

De khong ghi de Run 11 da dung cho final validation, phase nay danh so tiep:

```text
Run 12 - Train and evaluate OARH proxy
Run 13 - Train and evaluate RSDH proxy
Run 14 - Validation-gated learned pipeline
Run 15 - Add real MASt3R reciprocal matches
Run 16 - Re-evaluate RSDH with descriptor/cycle features
Run 17 - Optional light fine-tune MV-DUSt3R+
Run 18 - Learned full evaluation
Run 19 - Build supervised visibility/occlusion label cache
Run 20 - Mine occlusion-heavy and ambiguity-heavy subsets
Run 21 - Train OARH v2 multitask labels
Run 22 - Integrate OARH v2 into reconstruction
Run 23 - Train reconstruction-candidate calibration head
Run 24 - Train RSDH v2 from image-only patch/coordinate features
Run 25 - Integrate RSDH v2 into reconstruction
Run 26 - Diagnostic RSDH gate against candidate-retention baselines
Run 27 - Conditional light fine-tune if validation wins
Run 28 - Final held-out learned evaluation
```

Chi tiet nam trong:

```text
docs/method/supervised_extension_run_order.md
```

Nguyen tac quan trong:

- Scene-level split: train / val / test, khong de scene test xuat hien trong train.
- Chon threshold va checkpoint tren validation scenes.
- Final test dung threshold/checkpoint co dinh, khong tune lai.
- Khong ep multi-view consistency qua occluded views.
- Neu OARH/RSDH fail, bao cao trung thuc nhu cac ablation heuristic da lam.

## Submitted Phase 2 Kaggle Kernels

Da submit cac kernel dau cho phase learned extension:

| Run | Kaggle kernel | Ghi chu |
| --- | --- | --- |
| 12 | `mv-dust3r-run-12-supervised-reliability` | Freeze MV-DUSt3R+, tao label proxy tu GT depth, train OARH MLP, so sanh voi confidence-only tren held-out scene |
| 13 | `mv-dust3r-run-13-match-disambiguation` | Train RSDH proxy tren pair labels tu nearest-surface consistency; chua dung MASt3R descriptors day du |
| 14 | `mv-dust3r-run-14-validation-gated-learned-pipeline` | Chi ap dung OARH neu thang confidence-only tren validation proxy; neu khong thi fallback ve confidence |
| 15 | `mv-dust3r-run-15-mast3r-reciprocal-features` | Extract MASt3R reciprocal match features neu dependency stack san sang; neu khong, dung ORB fallback va log ro backend |
| 16 | `mv-dust3r-run-16-rsdh-descriptor-cycle` | Train RSDH voi descriptor/margin/reciprocal/3D disagreement/cycle-proxy features |
| 17 | `mv-dust3r-run-17-light-finetune-decision` | Dung ket qua validation-gated de quyet dinh co nen fine-tune nhe backbone hay khong |
| 18 | `mv-dust3r-run-18-learned-full-evaluation-summary` | Tong hop B0, current best, OARH/gated, RSDH va khuyen nghi final learned extension |
| 19 | `mv-dust3r-run-19-supervised-label-cache` | Tao label cache visibility/occlusion/floating/match de chuan bi OARH v2 va RSDH v2 |
| 20 | `mv-dust3r-run-20-occlusion-ambiguity-subset-mining` | Mine subset occlusion-heavy, low-overlap/far va hard-negative tu Run 19 |
| 21 | `mv-dust3r-run-21-oarh-v2-multitask` | Train OARH v2 multitask head tren Run 20 balanced labels |
| 22 | `mv-dust3r-run-22-oarh-v2-integration` | Tich hop OARH v2 vao reconstruction candidate filtering va so voi fixed-confidence final |
| 23 | `mv-dust3r-run-23-candidate-calibration` | Sau khi Run 22 fail, train reliability head tren actual MV-DUSt3R candidates va gate bang validation reconstruction F-score |
| 24 | `mv-dust3r-run-24-rsdh-v2-image-only` | Train RSDH v2 image-only match-validity head tu Run 20 hard-negative labels |
| 25 | `mv-dust3r-run-25-rsdh-v2-integration` | Tich hop Run 24 RSDH v2 vao reconstruction candidate scoring va gate bang validation reconstruction F-score |
| 26 | `mv-dust3r-run-26-rsdh-v2-diagnostic-gate` | Them all-candidate/confidence top-k baselines va exact top-k de kiem tra RSDH co thang that khong |

Ket qua hien tai:

- Run 12: OARH co validation label F1 cao nhung reconstruction F-score chi tang nhe o 4/5 views va giam manh o 2/3 views. Vi vay khong duoc claim OARH unconditional la thanh cong.
- Run 13: RSDH proxy dat match F1 tren 0.99 tren held-out scene, nhung day van la proxy label/feature, chua phai full MASt3R repeated-structure solution.
- Run 14: validation gate tranh duoc cac regression lon o 2/3/5 views, nhung gate 4-view hoi overfit validation proxy va thua confidence-only nhe tren held-out scene. Vi vay final policy van nen la confidence-only fixed threshold; OARH la partial/failed learned ablation can cai tien.
- Run 15--18: da submit de chay not phase learned extension. Run 15/16 kiem tra reciprocal feature va descriptor/cycle RSDH; Run 17 khong fine-tune mu quang neu validation chua ung ho; Run 18 tao summary cuoi cho bao cao.
- Run 19: bat dau phase nghiem ngat hon de giai quyet occlusion va repeated structures. Run nay tao label cache truoc, chua train model, de cac run sau khong con dua vao heuristic/proxy mong.
- Run 20: doc output Run 19 lam kernel source, tao balanced labels/manifests cho OARH v2 va RSDH v2 de train dung vao cac case kho.
- Run 21: train OARH v2 that su tren `oarh_v2_balanced_labels.csv`, du doan keep/reject, visibility class va depth residual. Feature input khong dung truc tiep `candidate_type` hay `visibility_label` de tranh leak nhan.
- Run 22: dung checkpoint Run 21 tren output MV-DUSt3R va do lai F-score reconstruction. Ket qua pasted log cho thay fixed confidence thang ro: val F-score 0.6716 so voi best learned 0.2160, test F-score 0.6033 so voi best learned 0.1936. Vi vay OARH v2 proxy khong duoc dung lam final reconstruction policy.
- Run 23: sua dung nguyen nhan domain shift cua Run 22 bang cach train tren actual reconstruction candidates, gan nhan bang GT geometry, va chi chon learned ranking neu validation reconstruction F-score thang fixed confidence. Ket qua pasted log gan hon nhieu nhung van fail gate: val fixed confidence 0.6674 so voi RCRH top_ratio_0.995 la 0.6618; test fixed 0.6014 so voi 0.5923.
- Run 24: chuyen sang limit repeated-structure/wrong-match, train RSDH v2 tu `rsdh_v2_hard_negative_labels.csv` bang image-only patch/coordinate features. Ket qua da pass gate proxy: validation learned F1 0.6954 so voi best image-only baseline 0.6212, test learned F1 0.6596 so voi 0.5517.
- Run 25: tich hop checkpoint Run 24 vao actual MV-DUSt3R reconstruction candidates. Ket qua khong pass gate: validation fixed confidence 0.1507, best learned 0.1542, delta +0.0035 nho hon margin 0.005. Test co policy RSDH cao hon nhung selected_ratio = 1.0, nen chua chung minh learned ranking thang that.
- Run 26: diagnostic da xong. Validation chon `all_candidates`: F-score 0.1463 so voi fixed confidence 0.1455; best learned RSDH cung chi dat 0.1463 tai selected_ratio = 1.0, delta vs best baseline = 0.0000 < margin 0.005. Vi vay RSDH v2 khong duoc dung trong final reconstruction pipeline.

Output can gui lai sau khi Kaggle chay xong:

- Run 12: `metrics.csv`, `training_history.csv`, `run_config.json`
- Run 13: `match_metrics.csv`, `training_history.csv`, `run_config.json`
- Run 14: `metrics.csv`, `gate_decisions.csv`, `training_history.csv`, `run_config.json`
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

Nguyen tac dung:

- Khong chay Run 2 neu Run 1 chua do metric on dinh.
- Khong chay Run 4 neu B1 chua duoc chon.
- Khong chay Run 6/7 truoc khi Run 5 cho thay fusion co loi.
- Khong chay Full pipeline truoc khi da co ket qua ablation rieng tung module.
