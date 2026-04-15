# AGENTS.md

Quy tắc bắt buộc cho mọi thay đổi trong repo này.

## 1) Build PDF từ `.tex` phải dọn sạch file trung gian/cache

Khi build tài liệu trong `pdf/`, luôn dùng quy trình:

```bash
cd pdf
latexmk -pdf main.tex
latexmk -c main.tex
find . -maxdepth 1 -type f \( \
  -name "*.aux" -o -name "*.log" -o -name "*.out" -o -name "*.toc" -o \
  -name "*.fls" -o -name "*.fdb_latexmk" -o -name "*.synctex.gz" -o \
  -name "*.bbl" -o -name "*.blg" \
\) -delete
```

Yêu cầu sau build:
- Giữ lại `pdf/main.pdf`, file nguồn `.tex`, `.bib`.
- Không để sót file trung gian/cache LaTeX trong `pdf/`.

## 2) Khi sửa code, bắt buộc cập nhật tài liệu liên quan

Nếu có thay đổi code (ví dụ trong `src/`, `scripts/`, hoặc file thực thi ở root), bắt buộc:

- Review và cập nhật `README.md` (nếu có và liên quan) cho đúng hành vi/câu lệnh mới.
- Review và cập nhật các file `.tex` liên quan trong `pdf/sections/` (hoặc `pdf/main.tex` nếu cần), sau đó rebuild `pdf/main.pdf`.
- Review và cập nhật mindmap liên quan nếu tồn tại (ví dụ `*.xmind`, `*.drawio`, `*.mm`).

Không được xem task là hoàn tất nếu chưa xử lý đủ các mục trên.

## 3) Checklist trước khi kết thúc task

- [ ] Code đã sửa xong.
- [ ] `README.md` đã được review/cập nhật theo thay đổi code (nếu có và liên quan).
- [ ] Tài liệu PDF liên quan đã được review/cập nhật và build lại.
- [ ] Đã xoá toàn bộ file trung gian/cache LaTeX sau khi build.
- [ ] Mindmap liên quan đã cập nhật, hoặc ghi rõ `N/A` nếu không có.
- [ ] `milestones.md` đã được review/cập nhật (nếu thay đổi có liên quan), hoặc ghi rõ `N/A`.

## 4) Quy tắc commit message

Nếu thực hiện commit, bắt buộc dùng Angular style (Conventional Commits), ví dụ:

- `feat(train): add reward model early stopping`
- `fix(pdf): clean latex intermediate artifacts after build`
- `docs(readme): update rollout command examples`

Format bắt buộc:

`<type>(<scope>): <subject>`

Trong đó:
- `<type>`: `feat`, `fix`, `docs`, `refactor`, `test`, `chore`, ...
- `<scope>`: module/thành phần bị ảnh hưởng (khuyến nghị luôn có).
- `<subject>`: ngắn gọn, dạng mệnh lệnh, không viết hoa chữ cái đầu, không dấu chấm cuối.

Có thể (khuyến nghị) thêm phần chi tiết ở bên dưới bằng commit body, ví dụ:

```text
feat(dataset): add shard merge validation

- validate missing shard indexes before merge
- fail fast when schema mismatch is detected
- update README and pdf section for new merge check
```

Quy ước cho body:
- Để trống 1 dòng sau subject.
- Mô tả chi tiết thay đổi bằng câu ngắn hoặc bullet list.
- Nêu rõ các file/tài liệu đã cập nhật khi phù hợp (`README`, `pdf`, mindmap).

## 5) Quy tắc pre-commit: version bump + cập nhật `pdf/versions`

Trước khi tạo commit, bắt buộc kiểm tra và xử lý version nếu repo có cơ chế versioning:

- Nếu có `VERSION` và/hoặc `versioning.py`:
  - Bump version tự động trước commit.
  - Mặc định bump mức `build`, trừ khi thay đổi yêu cầu `patch/minor/major`.
  - Ví dụ:
    ```bash
    uv run --locked python versioning.py bump build
    ```
- Sau khi bump version, nếu có thư mục `pdf/versions/`:
  - Cập nhật release notes/changelog tương ứng version mới.
  - Đảm bảo file tổng (`pdf/versions/part_versions.tex`) đã include đúng mục version mới.
  - Build lại `pdf/main.pdf` và dọn sạch file trung gian theo quy trình ở Mục 1.
  - Đảm bảo version ở trong các file khác được cập nhật theo:
    - 1) pyproject phải khớp VERSION
    - 2) uv.lock (package local collaborative-morl) phải khớp VERSION
    - 3) README phải hiển thị đúng version hiện tại
    - 4) PDF versioning scheme phải ghi đúng version hiện tại
    - 5) milestones phải đồng bộ nhãn version

Không được xem task là hoàn tất nếu đã commit code nhưng chưa bump version (khi có `VERSION`) hoặc chưa cập nhật phần `pdf/versions` (khi có).

## 6) Quy tắc cập nhật `milestones.md`

- Với mọi thay đổi có liên quan đến tiến độ/kế hoạch/phạm vi/kết quả công việc, bắt buộc review và cập nhật `milestones.md`.
- Các task đã hoàn thành trong `milestones.md` không được xoá khỏi lịch sử.
- Khi task hoàn tất, chỉ được cập nhật trạng thái (ví dụ: done/completed), bổ sung ghi chú hoặc ngày hoàn thành; không xoá dòng task đã làm.

## 7) Quy tắc trả lời câu hỏi từ `questions.md` (đặt dưới phần version)

- Khi có yêu cầu trả lời câu hỏi, nguồn câu hỏi mặc định là `questions.md` (nếu repo đang dùng `questions.txt` thì áp dụng tương tự).
- Chỉ trả lời các câu **chưa có đáp án**; không ghi đè hoặc viết lại các câu đã trả lời.
- Câu trả lời phải được đặt trong phần tài liệu versioned, ở **bên dưới phần version/changelog** (thuộc luồng `pdf/versions/`).
- Đảm bảo phần tổng `pdf/versions/part_versions.tex` vẫn include đúng thứ tự: changelog trước, Q\&A sau.
- Nếu không còn câu nào chưa trả lời, ghi rõ `N/A` trong báo cáo task thay vì tạo nội dung mới.
