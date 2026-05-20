# Đề xuất hướng xử lý cho ba vấn đề: xa reference, occlusion mạnh, cấu trúc lặp

## 1. Mục tiêu

Tài liệu này mô tả một hướng mở rộng khả thi cho project **sparse-view 3D reconstruction on unseen scenes** dựa trên **MV-DUSt3R+**. Mục tiêu là xử lý ba vấn đề thường gặp trong sparse-view reconstruction:

1. **Xa reference**: một số input views có chênh lệch viewpoint quá lớn so với reference view, làm chất lượng geometry giảm.
2. **Occlusion mạnh**: nhiều vùng trong scene bị che khuất, dẫn đến correspondence kém tin cậy và fusion nhiễu.
3. **Cấu trúc lặp**: các đối tượng hoặc vùng có hình dạng/hoa văn giống nhau như ghế, cửa, cửa sổ làm tăng ambiguity và dễ gây match nhầm.

Hướng đề xuất không thay backbone chính của model, mà bổ sung một pipeline chọn view và fusion có xét đến **visibility**, **occlusion**, và **ambiguity**.

---

## 2. Tổng quan phương pháp đề xuất

Phương pháp đề xuất gồm 3 thành phần chính:

### 2.1. Xử lý vấn đề xa reference
Đề xuất một **policy chọn sparse views** thay vì lấy ngẫu nhiên 2/3/5 ảnh.

Các policy có thể so sánh:

- **Random baseline**: chọn ngẫu nhiên 2/3/5 ảnh.
- **Coverage-aware**: chọn ảnh sao cho phủ scene tối đa.
- **Diversity-aware**: chọn ảnh sao cho góc nhìn đa dạng nhưng vẫn còn overlap.
- **Hybrid**: kết hợp `coverage + overlap + visibility score`.

Ý tưởng chính là chọn bộ input views sao cho:
- giảm số view quá xa reference,
- tăng mức độ bao phủ scene,
- tăng co-visibility giữa các ảnh,
- giảm vùng chỉ được quan sát bởi một ảnh.

---

### 2.2. Xử lý vấn đề occlusion mạnh
Đề xuất **occlusion-aware weighted fusion** để điểm 3D không được đóng góp như nhau trong point cloud cuối.

Một công thức thực dụng:

```math
w(p) = c(p) \times s(p) \times (1 - o(p))
```

Trong đó:
- `w(p)`: trọng số fusion của điểm/pixel `p`
- `c(p)`: confidence của model tại `p`
- `s(p)`: multi-view support, tức số lượng hoặc mức độ các view khác cùng ủng hộ geometry này
- `o(p)`: occlusion score, càng cao thì càng có khả năng vùng đó bị che khuất hoặc không đáng tin

Ý nghĩa:
- điểm có **confidence cao** thì nên được giữ mạnh hơn,
- điểm có **nhiều view cùng xác nhận** thì đáng tin hơn,
- điểm có **occlusion score cao** thì nên bị giảm ảnh hưởng.

Nếu một điểm:
- chỉ được 1 view “ủng hộ”, hoặc
- khi chiếu sang view khác bị lệch mạnh,

thì không nên cho đóng góp ngang với những điểm ổn định hơn.

---

### 2.3. Xử lý vấn đề cấu trúc lặp
Đề xuất **ambiguity-aware repeated-structure filtering**.

Ý tưởng chính:
- phát hiện các vùng có khả năng bị nhầm do cấu trúc lặp,
- chỉ cho phép các vùng này đóng góp mạnh vào point cloud cuối nếu có **anchor đáng tin** và **đồng thuận từ nhiều view**.

---

## 3. Pipeline cụ thể cho cấu trúc lặp

## Bước 1: Chạy model gốc
Dùng **MV-DUSt3R+** để lấy:
- pointmap của từng view,
- confidence map của từng view.

Không cần thay đổi backbone ở bước này.

---

## Bước 2: Tạo ambiguity map để phát hiện vùng cấu trúc lặp

Với mỗi pixel `p`, tính một điểm số mơ hồ `A(p)`. Điểm càng cao thì pixel càng dễ thuộc vùng cấu trúc lặp hoặc dễ match nhầm.

Một công thức thực dụng:

```math
A(p) = \alpha S(p) + \beta U(p) + \gamma R(p)
```

Trong đó:

- `S(p)`: **self-similarity score**  
  Đo xem patch quanh pixel `p` có giống quá nhiều patch khác trong cùng ảnh hay không.
  
  Ví dụ:
  - patch ở một cánh cửa giống nhiều patch cửa khác,
  - patch ở ghế giống nhiều patch ghế khác.

- `U(p)`: **uncertainty score**  
  Lấy từ confidence map của model, ví dụ:

  ```math
  U(p) = 1 - C(p)
  ```

  Pixel confidence thấp thường là pixel model không chắc.

- `R(p)`: **reprojection disagreement**  
  Dựng điểm 3D của pixel đó rồi chiếu ngược sang các view khác; nếu vị trí hoặc support giữa các view lệch nhau nhiều thì đây là dấu hiệu ambiguity hoặc match sai.

Ý nghĩa của ambiguity map:
- patch càng lặp,
- confidence càng thấp,
- reprojection càng bất đồng,

thì pixel đó càng đáng nghi.

---

## Bước 3: Chọn anchor pixels đáng tin

Sau khi có ambiguity map, tạo tập **anchor pixels** như sau:

```math
\mathcal{Q} = \{p \mid A(p) < \tau_a,\; C(p) > \tau_c,\; p \text{ visible in at least } k \text{ views}\}
```

Điều kiện để một pixel trở thành anchor:
- ambiguity thấp,
- confidence cao,
- được nhiều view cùng nhìn thấy.

Các anchor pixels là những điểm ổn định, ít có khả năng bị match nhầm do cấu trúc lặp.

---

## Bước 4: Với pixel nghi ngờ, yêu cầu anchor support

Với mỗi pixel nguy cơ cao `p` có `A(p)` lớn:

1. lấy điểm 3D mà model dự đoán,
2. chiếu sang các view khác,
3. kiểm tra xem vùng lân cận có anchor pixels hay không,
4. đánh giá các anchor này có “ủng hộ” cùng một lời giải hình học không.

Định nghĩa một **anchor support score**:

```math
G(p) = \frac{1}{|\mathcal{N}(p) \cap \mathcal{Q}|} \sum_{q \in \mathcal{N}(p) \cap \mathcal{Q}} \mathbf{1}(\|X(p) - \hat{X}_q(p)\| < \epsilon)
```

Trong đó:
- `\mathcal{N}(p)` là vùng lân cận của pixel `p`,
- `\mathcal{Q}` là tập anchor pixels,
- `X(p)` là điểm 3D dự đoán tại `p`,
- `\hat{X}_q(p)` là geometry được anchor `q` gợi ý hoặc xác nhận,
- `\epsilon` là ngưỡng chấp nhận.

Ý nghĩa:
- nếu các anchor quanh pixel `p` đồng ý với geometry hiện tại, điểm này đáng tin hơn,
- nếu anchor không ủng hộ, rất có thể đây là match nhầm do cấu trúc lặp.

---

## Bước 5: Yêu cầu multi-view agreement

Chỉ có anchor support vẫn chưa đủ. Với vùng cấu trúc lặp, một điểm chỉ nên được giữ nếu có **ít nhất 2 view khác cùng xác nhận**.

Định nghĩa một **multi-view agreement score**:

```math
M(p) = \frac{1}{|V_p|} \sum_{v \in V_p} \mathbf{1}(\text{reprojection error}_v(p) < \epsilon_r)
```

Trong đó:
- `V_p` là tập các view quan sát được điểm `p`,
- `\epsilon_r` là ngưỡng reprojection error.

Ý nghĩa:
- nếu nhiều view cùng đồng ý, điểm này đáng tin,
- nếu chỉ có một view ủng hộ hoặc các view khác bất đồng mạnh, nên giảm trọng số hoặc loại bỏ.

---

## Bước 6: Fusion point cloud có trọng số

Kết hợp các thành phần để có trọng số fusion cuối:

```math
W(p) = C(p) \cdot (1 - A(p)) \cdot G(p) \cdot M(p)
```

Trong đó:
- `C(p)`: confidence cao thì tốt,
- `A(p)`: ambiguity cao thì bị phạt,
- `G(p)`: anchor support cao thì được thưởng,
- `M(p)`: multi-view agreement cao thì được thưởng.

Sau đó:
- nếu `W(p)` thấp: giảm trọng số mạnh hoặc loại bỏ,
- nếu `W(p)` cao: giữ lại như điểm đáng tin trong point cloud cuối.

Kết quả kỳ vọng:
- giảm hiện tượng nhầm ghế này sang ghế kia,
- giảm các cụm point bị đặt sai vị trí ở vùng lặp,
- point cloud cuối sạch hơn và nhất quán hơn.

---

## 4. Pipeline tích hợp hoàn chỉnh

Pipeline đầy đủ cho ba vấn đề có thể viết như sau:

1. **View selection**
   - chọn 2/3/5 ảnh bằng policy `coverage-aware`, `diversity-aware`, hoặc `hybrid`
2. **Model inference**
   - chạy MV-DUSt3R+ để lấy pointmap và confidence map
3. **Occlusion-aware scoring**
   - tính `occlusion score`
   - tính `multi-view support`
4. **Ambiguity-aware scoring**
   - tính `ambiguity map`
   - chọn `anchor pixels`
   - tính `anchor support` và `multi-view agreement`
5. **Weighted fusion**
   - fuse point cloud bằng trọng số:

```math
W_{final}(p) = C(p) \cdot (1 - O(p)) \cdot S_{mv}(p) \cdot (1 - A(p)) \cdot G(p) \cdot M(p)
```

Trong đó:
- `O(p)`: occlusion score
- `S_{mv}(p)`: multi-view support
- `A(p)`: ambiguity score
- `G(p)`: anchor support
- `M(p)`: multi-view agreement

---

## 5. Contribution đề xuất của nhóm

Nhóm có thể mô tả contribution như sau:

### Contribution 1: View selection có xét visibility
Đề xuất một policy chọn sparse views thay vì lấy ngẫu nhiên, nhằm giảm vấn đề xa reference và tăng coverage.

### Contribution 2: Occlusion-aware weighted fusion
Đề xuất trọng số fusion có xét confidence, multi-view support và occlusion score.

### Contribution 3: Ambiguity-aware repeated-structure filtering
Đề xuất ambiguity map, anchor pixels và multi-view agreement để xử lý các vùng có cấu trúc lặp.

---

## 6. Cách đánh giá đề xuất

Nên đánh giá theo 3 trục:

### 6.1. So sánh policy chọn view
- Random baseline
- Coverage-aware
- Diversity-aware
- Hybrid

### 6.2. So sánh trước và sau filtering/fusion
- MV-DUSt3R+ gốc
- MV-DUSt3R+ + occlusion-aware weighted fusion
- MV-DUSt3R+ + ambiguity-aware repeated-structure filtering
- MV-DUSt3R+ + full pipeline

### 6.3. Đánh giá theo loại case khó
- far-reference cases
- high-occlusion cases
- repeated-structure cases

Metric có thể giữ theo proposal:
- reconstruction accuracy
- completeness
- consistency
- runtime per scene

---

## 7. Điểm mạnh của hướng đề xuất

- Không cần thay backbone lớn hoặc train lại từ đầu.
- Đánh đúng 3 bottleneck quan trọng của sparse-view reconstruction.
- Dễ kể chuyện trong report vì mỗi thành phần xử lý một lỗi cụ thể.
- Vừa có yếu tố kỹ thuật, vừa khả thi cho project môn học.

---

## 8. Kết luận

Hướng đề xuất phù hợp nhất cho project là xây dựng một pipeline:

- **view selection có xét coverage/visibility** để xử lý vấn đề xa reference,
- **occlusion-aware weighted fusion** để xử lý vùng bị che mạnh,
- **ambiguity-aware repeated-structure filtering** để giảm match nhầm ở các cấu trúc lặp.

Đây là một mở rộng hợp lý cho MV-DUSt3R+, giữ nguyên trục chính của bài toán sparse-view reconstruction nhưng bổ sung contribution rõ ràng, khả thi và đậm chất computer vision.
