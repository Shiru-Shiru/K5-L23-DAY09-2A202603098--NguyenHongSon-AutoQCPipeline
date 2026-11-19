# Báo cáo Day 09 — Nguyễn Hồng Sơn, 2A202603098

## 1. Kết luận trước
- Kết luận: **Hold**.
- Số lượng ảnh: 224 ảnh vào, 203 ảnh ra dataset (khâu intake loại 21 ảnh).
- Lý do: Không còn lỗi bắt buộc (0 error), nhưng còn 44/203 ảnh (21.67%) dính cảnh báo chưa được duyệt, vượt ngưỡng 10.00% của gate.

## 2. Lỗi và phân việc
Bảng lỗi từ `02_error_list_v1` (228 lỗi trên 224 ảnh):

| Mã lỗi | Số lượng | Ai sửa | Vì sao |
|---|---:|---|---|
| `CORRUPT_IMAGE` | 2 ảnh | Máy | File hỏng |
| `BAD_SIZE` | 3 ảnh | Máy | Cạnh ngắn < 250px |
| `DUP_IMAGE_EXACT` | 8 ảnh | Máy | Trùng md5 |
| `DUP_IMAGE_NEAR` | 3 ảnh | Máy | Trùng pHash |
| `LEAK_TEST` | 5 ảnh | Máy | Trùng với kho test/val |
| `BAD_CLASS_NAME` | 2 lỗi | Máy | Sai tên lớp trong data.yaml |
| `BAD_FORMAT` | 3 dòng | Máy | Sai định dạng 5 số |
| `BAD_CLASS_ID` | 6 dòng | Máy | ID lớp ngoài 0..3 |
| `BOX_ZERO` | 4 dòng | Máy | w hoặc h <= 0 |
| `BOX_OUT_OF_IMAGE` | 8 dòng | Máy | Box tràn ngoài [0, 1], clip về biên |
| `DUP_BOX` | 7 dòng | Máy | Trùng box cùng lớp (IoU > 0.9) |
| `MISSING_LABEL_FILE` | 6 ảnh | Người | Không có file nhãn, máy không tự tạo nhãn |
| `BOX_TINY` | 6 box | Người | Cạnh ngắn < 12px, cần người xác nhận vật thật |
| `BOX_HUGE` | 3 box | Người | Diện tích > 0.95, cần người kiểm tra có bao cả nền không |
| `EMPTY_LABEL` | 12 file | Người | File rỗng, cần người xác nhận là ảnh nền hay quên gán |
| `CLASS_RARE` | 4 lớp | Người | Lớp ít box ở mức dataset |
| `POSSIBLE_MISSING` | 68 box | Người | Model thấy vật nhưng nhãn không có |
| `POSSIBLE_WRONG_CLASS` | 18 box | Người | Model và nhãn lệch lớp (IoU >= 0.5) |
| `POSSIBLE_LOOSE_BOX` | 31 box | Người | Box bị lệch hoặc quá rộng (IoU < 0.5) |
| `POSSIBLE_FALSE_POS` | 29 box | Người | Nhãn có box nhưng model không thấy |

- **Vì sao xếp đầu hàng đợi `05_review_queue.csv`:** Xếp theo điểm ưu tiên giảm dần. Ảnh thiếu nhãn (`NEEDS_LABEL`) có trọng số cao nhất (10 điểm) và các ảnh dính nhiều lỗi cùng lúc (`img_0020`, `img_0185`) có điểm cao nhất nên đứng đầu hàng.
- **Số lỗi xem được với ngân sách 80 ảnh:** Người duyệt xem 80 ảnh đầu tiên, giải quyết 6 ảnh thiếu nhãn và 104 lượt cảnh báo ngữ nghĩa (51 missing, 18 wrong class, 12 false pos, 11 loose box, 11 empty label, 1 tiny box). Còn 50 ảnh trong hàng đợi (chứa 44 ảnh có warning) chưa xem được.

## 3. Calibration (2 ngưỡng)
| Tham số | Khởi đầu | Đã chọn | Precision / Recall (trước → sau) | Lý do (đánh đổi) |
|---|---|---|---|---|
| `duplicates.phash_dist` | 2 | 8 | **DUP_IMAGE_NEAR:**<br>P: 1.00 $\rightarrow$ 1.00<br>R: 0.50 $\rightarrow$ 0.67 | Tăng recall bắt ảnh trùng từ 0.50 lên 0.67. Không chọn 16 để tránh xóa nhầm ảnh thật trên raw. |
| `second_opinion.conf` | 0.25 | 0.50 | **POSSIBLE_MISSING:**<br>P: 0.50 $\rightarrow$ 0.57, R: 1.00 $\rightarrow$ 1.00 | Tăng precision từ 50% lên 57% để giảm cảnh báo giả, giữ recall 100%. |

**Một chỗ calibration có thể đánh lừa:**
Tập calib chỉ có 39 ảnh nhân tạo nên tăng `phash_dist` lên 16 vẫn đạt precision 1.00. Nhưng trên raw có 224 ảnh thật với nhiều góc phố giống nhau, để ngưỡng 16 sẽ xóa nhầm ảnh khác. Dùng ngưỡng 8 trên raw loại đúng 3 ảnh near duplicate (`img_0045`, `img_0067`, `img_0109`), không bị xóa nhầm.

## 4. Cửa chặn và điều pipeline KHÔNG bắt được
**(a) Chứng minh gate chặn đúng:**
Chạy lệnh với `--budget 0`:
`python -m pipeline.run --budget 0 --reports work/rep_nb --export-dir work/exp_nb`
- Kết quả: Không có người duyệt, 6 ảnh thiếu file nhãn còn nguyên. Gate kết luận **Rework**, mã thoát 20, pipeline không xuất dataset ra `work/exp_nb`.

**(b) Điểm mù pipeline không bắt được:**
1. **Box lỏng nhẹ (0.5 <= IoU < 0.7):** `second_opinion` coi IoU >= 0.5 cùng lớp là khớp nên bỏ qua.
2. **Model và người cùng đoán sai lớp:** Cả hai cùng nhầm xe van thành bus thì model im lặng, lỗi này bị bỏ sót.
3. **Ảnh trùng bị crop hoặc xoay mạnh:** Khoảng cách pHash vượt quá 8 nên bộ lọc không bắt được.

## 5. Train: raw vs clean (tùy chọn, chấm riêng)
Chưa chạy.

## 6. Quyết định và ghi chú AI
**(a) Quyết định cuối:** **Hold**.
Để thành Release: Cần duyệt thêm tối thiểu 44 ảnh còn cảnh báo trong hàng đợi, hoặc xác nhận các file nhãn rỗng là ảnh nền để tỉ lệ cảnh báo dưới 10.00%.

**(b) Ghi chú AI:**
- Prompt chính: Viết 4 module label_format, gate, autofix, second_opinion theo TDD khớp test suite.
- Cách kiểm: Chạy pytest từng file test và chạy check_submission.py.
- Một lần AI sai: Trong second_opinion.py, AI lúc đầu lọc cả min_area_px cho POSSIBLE_FALSE_POS làm sai test, sau đó đã sửa lại chỉ lọc theo fp_conf.
