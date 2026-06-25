# Presentation Script - Sparse-View RGB 3D Reconstruction

## Slide 1 - Title

Chào thầy và các bạn. Nhóm em xin trình bày project Computer Vision với đề tài **Sparse-View RGB 3D Scene Reconstruction with Fine-Tuned Estimated-Depth Correction**.

Bài toán chính của nhóm là tái tạo hình học 3D của cảnh trong nhà khi chỉ có một số lượng rất ít ảnh RGB đầu vào. Trong project này, nhóm sử dụng MV-DUSt3R+ làm backbone tái tạo sparse-view RGB, kết hợp với depth estimator để tạo tín hiệu độ sâu ước lượng, rồi dùng depth correction để căn chỉnh đầu ra 3D.

## Slide 2 - Member Contributions

Slide này tóm tắt phân công công việc của nhóm.

Bạn **Tường Minh** phụ trách phần backbone sparse-view RGB reconstruction: tìm hiểu MV-DUSt3R/MV-DUSt3R+, chuẩn bị input sparse-view 3/4/5 ảnh RGB, chạy baseline RGB-only và phân tích lỗi tái tạo.

Bạn **Tùng** phụ trách phần monocular depth estimation: xử lý dữ liệu RGB-D, dùng source depth làm supervision, đánh giá depth estimator bằng AbsRel, MAE, RMSE và delta accuracy.

Bạn **Ngọc Minh** phụ trách phần depth correction, tích hợp pipeline và đánh giá 3D: thiết kế module hiệu chỉnh candidate geometry bằng estimated depth, kết hợp output của MV-DUSt3R+ và depth estimator, sau đó đánh giá bằng Accuracy, Completeness, Precision, Recall, F-score và Chamfer distance.

## Slide 3 - Motivation

Động lực của project đến từ ba khó khăn chính trong sparse-view 3D reconstruction.

Thứ nhất, input rất thưa. Mỗi group chỉ có khoảng 3 đến 5 ảnh, nên overlap giữa các view có thể yếu. Khi overlap thấp, việc tìm correspondence và hợp nhất hình học dễ bị sai.

Thứ hai, bài toán có ambiguity về depth. Các vùng ít texture, bị che khuất, hoặc có cấu trúc lặp lại có thể khiến mô hình RGB-only dự đoán geometry không ổn định.

Thứ ba, measured RGB-D depth rất hữu ích, nhưng trong setting triển khai thực tế, nhóm muốn hướng tới input RGB-only. Vì vậy, ý tưởng chính là: dùng MV-DUSt3R+ để tạo candidate geometry, dùng depth estimator để dự đoán metric depth từ RGB, rồi dùng estimated depth như một anchor hình học để sửa candidate 3D points.

## Slide 4 - Objective & Contributions

Câu hỏi nghiên cứu chính là: **Liệu estimated metric depth, được học từ RGB-D supervision, có thể sửa hình học sparse-view của MV-DUSt3R+ trong khi vẫn giữ inference input là RGB-only hay không?**

Project có ba đóng góp chính.

Một là xác định target setting RGB-only: pipeline cuối nhận sparse RGB views, còn ground-truth depth và pose chỉ dùng cho training hoặc evaluation.

Hai là dùng depth prior đã được fine-tune từ dữ liệu RGB-D trong nhà, cụ thể là Depth Anything V2 Metric Indoor.

Ba là thiết kế module depth-corrected geometry: các candidate pointmaps từ MV-DUSt3R+ được kiểm tra bằng residual gate, sau đó chỉ các điểm lệch nhiều mới được kéo về phía source-depth ray ước lượng.

Nhóm cũng giới hạn claim rõ ràng: đây là controlled project protocol, không phải official ScanNet hoặc ScanNet++ benchmark.

## Slide 5 - Dataset Overview

Về dataset, nhóm sử dụng dữ liệu RGB-D kiểu ScanNet từ Kaggle.

Ban đầu, nhóm phát triển và kiểm thử trên controlled subset gồm 30 scenes, với các sparse-view groups có 3, 4 hoặc 5 view. Sau đó, pipeline được mở rộng sang full discovered Kaggle data, gồm 1513 scenes và hơn 248 nghìn RGB-D frames.

Depth estimator được đánh giá trên held-out validation và test split. Reconstruction 3D được đánh giá trên sparse eval groups.

Điểm quan trọng là cần phân biệt hai khái niệm: **all-scene discovery** nghĩa là runner quét toàn bộ scene tìm được trong dataset; còn **all-group metric evaluation** nghĩa là đánh giá toàn bộ eval groups đã phát hiện, không giới hạn bằng cap. Dù vậy, các kết quả này vẫn là protocol nội bộ của project, không phải benchmark chính thức.

## Slide 6 - Pipeline Diagram

Slide này mô tả pipeline tổng thể bằng tiếng Anh.

Input đầu vào là các sparse RGB views, ký hiệu là \(I_1, ..., I_N\). Từ cùng input này, pipeline tách thành hai nhánh.

Nhánh thứ nhất đưa ảnh RGB vào MV-DUSt3R+ để sinh ra candidate pointmaps, tức là hình học 3D dự đoán ban đầu.

Nhánh thứ hai đưa ảnh RGB vào depth estimator đã được tinh chỉnh để sinh estimated depth maps.

Sau đó, module correction dùng estimated depth để căn chỉnh candidate pointmaps. Output cuối là point cloud 3D đã được hiệu chỉnh.

Ý nghĩa của pipeline là: nhóm không thay thế MV-DUSt3R+, mà bổ sung một tín hiệu depth ước lượng để sửa những vùng candidate geometry bị lệch.

## Slide 7 - Overall Pipeline

Slide này tóm tắt pipeline theo ba stage.

Stage 1: sparse RGB views được đưa vào MV-DUSt3R+ để tạo candidate pointmaps.

Stage 2: depth estimator dự đoán metric source depth từ chính các RGB views đó.

Stage 3: residual-gated correction sẽ di chuyển các candidate point không đáng tin về gần estimated source-depth geometry hơn.

Như vậy, phần đóng góp chính của project nằm ở cách kết hợp hai loại tín hiệu: multi-view RGB geometry từ MV-DUSt3R+ và metric depth prior từ depth estimator.

## Slide 8 - MV-DUSt3R+ Backbone

Ở phần backbone, nhóm sử dụng MV-DUSt3R+ như một mô hình single-stage sparse-view RGB reconstruction.

Mô hình nhận nhiều view RGB và sinh ra pixel-aligned 3D pointmaps. Đây là candidate geometry ban đầu cho toàn bộ pipeline.

Điểm quan trọng là nhóm không đưa source depth vào trong backbone MV-DUSt3R+. Backbone vẫn hoạt động theo input RGB-only. Phần depth chỉ được dùng ở module bên ngoài để hiệu chỉnh geometry sau khi candidate points đã được tạo.

Vì vậy, inference contract của setting này vẫn là sparse RGB views; estimated depth được sinh bởi depth model, không lấy từ depth sensor.

## Slide 9 - Depth Estimator Fine-Tuning

Depth estimator được dùng là **Depth Anything V2 Metric Indoor**.

Nhóm fine-tune hoặc adapt mô hình depth này trên dữ liệu RGB-D trong nhà. Dữ liệu controlled split có 1210 scenes và gần 199 nghìn frames. Full-data deployment có 1513 scenes và hơn 248 nghìn frames.

Mỗi setting được chạy một epoch. Mục tiêu không phải huấn luyện một depth model hoàn toàn mới, mà là điều chỉnh model có sẵn để depth prediction phù hợp hơn với domain indoor RGB-D của project.

Việc fine-tune depth estimator rất quan trọng, vì nếu estimated depth không đủ chính xác thì depth correction có thể kéo geometry về sai vị trí.

## Slide 10 - Estimated-Depth Correction

Slide này là công thức của module correction.

Đầu tiên, với mỗi pixel \((u,v)\), depth estimator dự đoán giá trị \(\hat{z}\). Từ \(\hat{z}\), camera intrinsics \(K_i\), và pose \(T_i\), ta back-project pixel đó thành một điểm 3D trong world coordinate.

Sau đó, nhóm tính residual giữa candidate point của MV-DUSt3R+ và điểm 3D từ estimated depth. Nếu residual nhỏ, nghĩa là hai nguồn geometry đã đồng ý với nhau, ta giữ nguyên candidate point.

Nếu residual lớn hơn threshold \(\tau\), candidate point sẽ được hiệu chỉnh theo công thức interpolation giữa predicted point và source-depth point. Tham số \(\alpha\) điều khiển mức độ kéo về phía estimated depth.

Nói ngắn gọn, module này không sửa tất cả các điểm, mà chỉ sửa những điểm có sai lệch hình học lớn.

## Slide 11 - Evaluation Metrics

Project dùng hai nhóm metric.

Nhóm đầu tiên là metric cho depth estimator. **AbsRel** đo lỗi tương đối trung bình, càng thấp càng tốt. **MAE** và **RMSE** đo lỗi theo đơn vị mét, trong đó RMSE phạt mạnh các lỗi lớn. **delta1** đo tỷ lệ pixel có depth prediction nằm trong ngưỡng đúng theo tỷ lệ 1.25, càng cao càng tốt.

Nhóm thứ hai là metric cho 3D reconstruction. **Accuracy** đo khoảng cách từ predicted point cloud đến reference point cloud. **Completeness** đo chiều ngược lại, từ reference đến prediction.

Precision, Recall và F-score được tính tại threshold 0.05m, trong đó F-score là metric cân bằng chính. Chamfer distance là tổng của accuracy và completeness. Ngoài ra, nhóm cũng có các metric chuẩn hóa như ND và DAc@0.2 để phân tích gần với phong cách paper.

Reference point cloud trong project là proxy reference dựng từ depth maps, không phải laser-scan ground truth độc lập.

## Slide 12 - 3D Reconstruction Comparison

Slide này là bảng so sánh các phương pháp reconstruction chính.

Baseline đầu tiên là RGB-only MV-DUSt3R+, đạt F-score 0.1744. Đây là kết quả khi chỉ dùng candidate geometry từ backbone RGB-only.

Phương pháp direct backprojection từ estimated depth đạt F-score 0.1894. Điều này cho thấy depth estimator có tín hiệu hình học hữu ích, nhưng bản thân nó chưa đủ để thay thế hoàn toàn multi-view reconstruction.

Correction với pretrained estimated depth đạt F-score 0.1465, thấp hơn baseline. Kết quả này cho thấy nếu depth estimator chưa phù hợp domain, dùng nó làm correction anchor có thể làm geometry tệ hơn.

Correction với measured RGB-D source depth đạt F-score 0.2764, cao nhất trong bảng. Đây là upper reference cho module correction vì nó dùng sensor depth thật. Tuy nhiên, nó không phải setting deployment RGB-only.

Kết luận từ bảng là: depth correction có tiềm năng, nhưng chất lượng depth anchor quyết định rất lớn đến chất lượng 3D cuối cùng.

## Slide 13 - Qualitative 3D Outputs

Slide này minh họa các output 3D định tính.

Các visualization giúp ta nhìn trực giác xem point cloud có giữ được cấu trúc cảnh hay không, ví dụ mặt phẳng, đồ vật lớn, hoặc bố cục không gian.

Tuy nhiên, nhóm không chỉ dựa vào hình ảnh định tính, vì ảnh chụp point cloud có thể phụ thuộc góc nhìn và khó so sánh công bằng. Vì vậy, report chính vẫn dùng các metric định lượng như Accuracy, Completeness, F-score và Chamfer distance.

## Slide 14 - Ablation & Interpretation

Slide này tóm tắt ý nghĩa của các ablation.

RGB-only MV-DUSt3R+ cho biết backbone sparse-view RGB mạnh đến đâu khi chưa có depth correction.

Direct estimated-depth backprojection kiểm tra xem monocular depth có tự tạo được geometry hữu ích hay không.

Pretrained-depth correction kiểm tra việc dùng off-the-shelf depth làm correction anchor. Kết quả giảm cho thấy depth prediction chưa đủ metric thì correction có thể gây hại.

Bài học chính là depth chỉ hữu ích khi đủ chính xác so với threshold reconstruction 0.05m. Measured-depth correction đóng vai trò upper reference, còn estimated-depth correction là hướng giúp pipeline tiến gần hơn tới RGB-only deployment.

## Slide 15 - Limitations

Project có một số giới hạn cần nói rõ.

Thứ nhất, evaluation dùng controlled ScanNet-style protocol, không phải official full benchmark.

Thứ hai, proxy reference point cloud được dựng từ depth maps có sẵn, không phải ground truth laser scan độc lập.

Thứ ba, correction và evaluation vẫn dùng intrinsics và camera poses.

Thứ tư, monocular metric depth vẫn có ambiguity về scale và hình học. Fine-tuning có thể giảm lỗi, nhưng không loại bỏ hoàn toàn vấn đề này.

Vì vậy, claim của project nên được giới hạn trong controlled RGB-D supervision setting, không diễn giải thành outperform official MV-DUSt3R+ benchmark.

## Slide 16 - Conclusion & Next Steps

Tóm lại, project đề xuất một hướng kết hợp MV-DUSt3R+ với estimated-depth correction cho sparse-view RGB 3D reconstruction.

Backbone MV-DUSt3R+ cung cấp candidate geometry từ sparse RGB views. Depth estimator cung cấp metric depth prior từ RGB. Module correction dùng residual gate để sửa các candidate point bị lệch nhiều.

Kết quả cho thấy measured-depth correction là upper reference tốt, còn pretrained estimated depth chưa đủ ổn định. Điều này nhấn mạnh vai trò quan trọng của việc cải thiện depth estimator trước khi dùng nó để correction.

Hướng tiếp theo là hoàn thiện final setting trên toàn bộ discovered scenes và all eval groups, cập nhật bảng reconstruction bằng kết quả đầy đủ, và thêm confidence-aware correction để tránh over-correct khi depth estimate không đáng tin.
