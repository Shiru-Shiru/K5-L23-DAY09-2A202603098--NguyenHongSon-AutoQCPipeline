"""CHECK 6 — Ý kiến thứ hai: so nhãn với một model LỚN HƠN (yolo26l).   ★★★ (khó nhất, đáng điểm nhất)

Test:  pytest tests/test_second_opinion.py -q

Máy kiểm quy tắc, người kiểm ý nghĩa (slide). Các lỗi như thiếu nhãn / gán nhầm lớp hợp lệ / box lỏng
KHÔNG bắt được bằng quy tắc cú pháp. Ta nhờ một model mạnh hơn "nhìn lại" và CHỈ CẢNH BÁO (model cũng có thể sai):
những ảnh bị cờ sẽ vào hàng đợi cho NGƯỜI duyệt.

`preds[stem]` = list `[id_lớp, conf, cx, cy, w, h]` của model (lấy qua `get_preds(ds, cfg, "checker")` nếu preds=None).
`L = ds.valid_boxes(stem)` = list `(id_lớp, cx, cy, w, h)` của nhãn (đã bỏ dòng hỏng).
Ngưỡng ở `cfg["second_opinion"]`: conf, min_area_px, fp_conf, iou_missing, iou_match, iou_loose.

Thuật toán gợi ý, với từng ảnh (bỏ qua ảnh không có trong preds hoặc không có file nhãn):
  P = dự đoán có conf >= conf  VÀ diện tích px (w*W * h*H) >= min_area_px
  Với mỗi dự đoán r trong P: lấy các nhãn có IoU(r, nhãn) >= iou_missing  (cands)
   - không có cands                            -> POSSIBLE_MISSING     (model thấy vật mà nhãn không có box)
   - có cands cùng lớp, tốt nhất IoU < iou_loose -> POSSIBLE_LOOSE_BOX  (đúng vật, box lỏng/lệch)
   - không có cands cùng lớp, nhưng cands tốt nhất IoU >= iou_match (khác lớp) -> POSSIBLE_WRONG_CLASS
  Với mỗi dòng nhãn hợp lệ: nếu IoU với MỌI dự đoán (conf >= fp_conf, không lọc diện tích) đều < iou_missing
                                             -> POSSIBLE_FALSE_POS  (box nhãn không khớp vật nào model thấy; Issue.line = số dòng)

Chi tiết các ca biên (test kiểm đúng các ca này)
- `cands` gồm nhãn MỌI lớp có IoU >= iou_missing; "tốt nhất cùng lớp" = cands cùng lớp có IoU lớn nhất; "tốt nhất khác lớp" tương tự với cands khác lớp.
- Thứ tự quyết định cho mỗi dự đoán: (1) cands rỗng -> MISSING. (2) có cands cùng lớp: IoU tốt nhất < iou_loose -> LOOSE_BOX, ngược lại KHÔNG cờ
  (nhãn đúng lớp đã phủ vật này, kể cả khi còn nhãn khác lớp chồng lên). (3) không có cands cùng lớp: nếu IoU tốt nhất khác lớp >= iou_match -> WRONG_CLASS;
  nếu chỉ có khác lớp với IoU trong [iou_missing, iou_match) -> không cờ gì (mơ hồ: để cho FALSE_POS / lần duyệt khác).
- Một dự đoán sinh tối đa MỘT issue (MISSING / LOOSE_BOX / WRONG_CLASS); một ảnh có thể có nhiều issue.
- Thứ tự issue: theo `ds.image_files()` rồi theo thứ tự dự đoán trong `preds[stem]`, cuối cùng là các FALSE_POS theo số dòng; `Issue.image` = stem, `Issue.line` chỉ đặt cho FALSE_POS.

Lưu ý số dòng: dùng `ds.read_label_lines(stem)` (giữ số dòng gốc, kể cả dòng hỏng/trống) để lấy `Issue.line`; `valid_boxes` KHÔNG giữ số dòng.

Tất cả severity WARN; `Issue.extra` nên có {"conf": ...} hoặc {"iou": ...} (hàng đợi duyệt dùng conf để xếp ưu tiên).

Gợi ý
- `iou_xywh` có sẵn. Dòng nhãn hỏng không được làm script chết (dùng `ds.valid_boxes` và `parse_yolo_line` trong try).
- Kết quả phụ thuộc mạnh vào ngưỡng: calibrate chúng ở Bước 3 thay vì đoán.
"""
from ..common import WARN, Issue, iou_xywh, parse_yolo_line
from ..model_cache import get_preds


def run(ds, cfg, preds=None):
    if preds is None:
        preds = get_preds(ds, cfg, "checker")

    so_cfg = cfg["second_opinion"]
    conf_thresh = so_cfg["conf"]
    min_area_px = so_cfg["min_area_px"]
    fp_conf = so_cfg["fp_conf"]
    iou_missing = so_cfg["iou_missing"]
    iou_match = so_cfg["iou_match"]
    iou_loose = so_cfg["iou_loose"]

    issues = []

    for img_path in ds.image_files():
        stem = img_path.stem
        if stem not in preds or not ds.label_path(stem).exists():
            continue

        im_sz = ds.image_size(stem)
        if im_sz is None:
            continue
        im_w, im_h = im_sz

        raw_lines = ds.read_label_lines(stem)
        if raw_lines is None:
            continue

        valid_labels = []
        for line_no, text in raw_lines:
            try:
                c, cx, cy, w, h = parse_yolo_line(text)
            except ValueError:
                continue
            if 0 <= c < len(ds.classes):
                valid_labels.append((line_no, c, (cx, cy, w, h)))

        all_preds = preds.get(stem, [])

        # P = dự đoán có conf >= conf VÀ diện tích px (w*W * h*H) >= min_area_px
        P = []
        for r in all_preds:
            p_cls = int(r[0])
            p_conf = float(r[1])
            p_box = (float(r[2]), float(r[3]), float(r[4]), float(r[5]))
            area_px = p_box[2] * im_w * p_box[3] * im_h
            if p_conf >= conf_thresh and area_px >= min_area_px:
                P.append((p_cls, p_conf, p_box))

        # Kiểm tra từng dự đoán trong P
        for p_cls, p_conf, p_box in P:
            same_cands = []
            diff_cands = []

            for line_no, l_cls, l_box in valid_labels:
                iou = iou_xywh(p_box, l_box)
                if iou >= iou_missing:
                    if l_cls == p_cls:
                        same_cands.append(iou)
                    else:
                        diff_cands.append(iou)

            if not same_cands and not diff_cands:
                # Không có cands -> POSSIBLE_MISSING
                issues.append(Issue(
                    check="second_opinion",
                    severity=WARN,
                    code="POSSIBLE_MISSING",
                    image=stem,
                    message=f"model phát hiện {ds.classes[p_cls]} (conf={p_conf:.2f}) nhưng nhãn thiếu",
                    line=None,
                    extra={"conf": p_conf},
                ))
            elif same_cands:
                best_same_iou = max(same_cands)
                if best_same_iou < iou_loose:
                    # Có cands cùng lớp nhưng tốt nhất < iou_loose -> POSSIBLE_LOOSE_BOX
                    issues.append(Issue(
                        check="second_opinion",
                        severity=WARN,
                        code="POSSIBLE_LOOSE_BOX",
                        image=stem,
                        message=f"box nhãn lỏng/lệch so với model (IoU={best_same_iou:.2f} < {iou_loose})",
                        line=None,
                        extra={"conf": p_conf, "iou": round(best_same_iou, 3)},
                    ))
                # Ngược lại best_same_iou >= iou_loose -> không cờ gì
            else:
                # Không có cands cùng lớp, chỉ có diff_cands
                best_diff_iou = max(diff_cands)
                if best_diff_iou >= iou_match:
                    issues.append(Issue(
                        check="second_opinion",
                        severity=WARN,
                        code="POSSIBLE_WRONG_CLASS",
                        image=stem,
                        message=f"nhãn lệch lớp so với model (IoU={best_diff_iou:.2f} >= {iou_match})",
                        line=None,
                        extra={"conf": p_conf, "iou": round(best_diff_iou, 3)},
                    ))
                # Ngược lại [iou_missing, iou_match) -> không cờ gì (mơ hồ)

        # Kiểm tra POSSIBLE_FALSE_POS
        fp_preds = [
            (float(r[2]), float(r[3]), float(r[4]), float(r[5]))
            for r in all_preds
            if float(r[1]) >= fp_conf
        ]

        for line_no, l_cls, l_box in valid_labels:
            if not fp_preds or all(iou_xywh(l_box, pb) < iou_missing for pb in fp_preds):
                issues.append(Issue(
                    check="second_opinion",
                    severity=WARN,
                    code="POSSIBLE_FALSE_POS",
                    image=stem,
                    message=f"box nhãn không khớp dự đoán nào của model (conf >= {fp_conf})",
                    line=line_no,
                    extra={"iou": 0.0},
                ))

    return issues
