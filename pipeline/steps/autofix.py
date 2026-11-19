"""BƯỚC 2/4 — Sửa tự động những lỗi bắt buộc mà máy sửa được AN TOÀN.   ★★

Test:  pytest tests/test_autofix.py -q

`fix(ds, issues, cfg) -> list[dict]`: duyệt `issues` (chỉ lỗi severity ERROR), sửa thẳng vào thư mục `ds`,
và trả về danh sách thay đổi; mỗi phần tử có khoá: image, line, action, before, after
(đây chính là file `03_fixes.csv` — bằng chứng pipeline đã làm gì, người có thể kiểm lại).

| code lỗi                                       | cách sửa                                    | action           |
|------------------------------------------------|---------------------------------------------|------------------|
| BAD_FORMAT, BAD_CLASS_ID, BOX_ZERO, DUP_BOX    | xoá dòng đó khỏi file nhãn                  | "drop_line"      |
| BOX_OUT_OF_IMAGE                               | cắt box về trong [0,1] (giữ nguyên lớp)     | "clip_box"       |
| MISSING_LABEL_FILE                             | KHÔNG sửa file (máy không thể bịa nhãn): ghi 1 dòng fix, before="(không có file)", after="chuyển người gán" | "needs_label" |
| BAD_CLASS_NAME                                 | ghi lại `names` trong data.yaml đúng `cfg["classes"]` | "rename_classes" |
| DUP_IMAGE_*, LEAK_TEST, CORRUPT_IMAGE, BAD_SIZE| xoá ảnh + nhãn khỏi `ds`                    | "remove_image"   |

Bẫy cần nghĩ tới (có test cho từng cái)
- Một file có NHIỀU dòng lỗi: số dòng trong `issues` là số dòng GỐC. Nếu xoá từng dòng một thì số dòng lệch.
  Gom theo file, rồi ghi lại MỘT lần.
- Sau khi cắt, box có thể không còn diện tích (nằm hẳn ngoài ảnh): lúc đó xoá dòng thay vì ghi box rỗng.
- Một dòng có 2 lỗi (vd. BOX_OUT_OF_IMAGE + DUP_BOX): xoá dòng thắng, chỉ ghi 1 fix `drop_line`, không ghi thêm `clip_box`. Dòng trống ở giữa file vẫn đếm vào số dòng gốc.
- Ảnh vừa bị xoá (DUP_IMAGE_*...) mà còn issue dòng của chính nó: bỏ qua êm, không được báo lỗi/chết.
- Không động tới lỗi WARN (cảnh báo là việc của người). Không sửa file ở `data/raw/` — chỉ sửa `ds`.
- Chạy lần 2 trên dữ liệu đã sửa phải không làm gì thêm (idempotent) — NGOẠI LỆ duy nhất: `needs_label` được ghi lại mỗi lần còn lỗi `MISSING_LABEL_FILE` (vì ảnh vẫn chưa có nhãn).

- `needs_label` KHÔNG tạo file nhãn rỗng (rỗng = "ảnh nền", khác với "chưa gán"). `review_queue` sẽ đẩy ảnh này lên đầu hàng đợi cho người.

Gợi ý: `yaml.safe_load / safe_dump`.
"""
from collections import defaultdict
import yaml

from ..common import ERROR, parse_yolo_line


def fix(ds, issues, cfg):
    error_issues = [i for i in issues if i.severity == ERROR]
    fixes = []

    # 1. remove_image: DUP_IMAGE_*, LEAK_TEST, CORRUPT_IMAGE, BAD_SIZE
    remove_codes = {"LEAK_TEST", "CORRUPT_IMAGE", "BAD_SIZE"}
    to_remove = set()
    for i in error_issues:
        if i.code in remove_codes or i.code.startswith("DUP_IMAGE_"):
            to_remove.add(i.image)

    for stem in sorted(to_remove):
        img_p = ds.image_path(stem)
        if img_p and img_p.exists():
            try:
                img_p.unlink()
            except OSError:
                pass
        lbl_p = ds.label_path(stem)
        if lbl_p and lbl_p.exists():
            try:
                lbl_p.unlink()
            except OSError:
                pass
        fixes.append({
            "image": stem,
            "line": "",
            "action": "remove_image",
            "before": stem,
            "after": "removed",
        })

    # 2. rename_classes: BAD_CLASS_NAME
    has_bad_class_name = any(i.code == "BAD_CLASS_NAME" for i in error_issues)
    if has_bad_class_name and ds.yaml_path.exists():
        try:
            d = yaml.safe_load(ds.yaml_path.read_text(encoding="utf-8")) or {}
            old_names = d.get("names", {})
            d["names"] = list(cfg["classes"])
            ds.yaml_path.write_text(yaml.safe_dump(d, allow_unicode=True), encoding="utf-8")
            fixes.append({
                "image": "data.yaml",
                "line": "",
                "action": "rename_classes",
                "before": str(old_names),
                "after": str(cfg["classes"]),
            })
        except Exception:
            pass

    # 3. needs_label: MISSING_LABEL_FILE
    missing_label_stems = sorted({i.image for i in error_issues if i.code == "MISSING_LABEL_FILE" and i.image not in to_remove})
    for stem in missing_label_stems:
        fixes.append({
            "image": stem,
            "line": "",
            "action": "needs_label",
            "before": "(không có file)",
            "after": "chuyển người gán",
        })

    # 4. drop_line & clip_box
    drop_codes = {"BAD_FORMAT", "BAD_CLASS_ID", "BOX_ZERO", "DUP_BOX"}
    file_issues = defaultdict(lambda: defaultdict(list))
    for i in error_issues:
        if i.image in to_remove or i.image == "data.yaml":
            continue
        if i.code in drop_codes or i.code == "BOX_OUT_OF_IMAGE":
            if i.line is not None:
                file_issues[i.image][i.line].append(i)

    for stem in sorted(file_issues.keys()):
        lbl_p = ds.label_path(stem)
        if not lbl_p.exists():
            continue
        raw_lines = lbl_p.read_text(encoding="utf-8", errors="replace").splitlines()
        line_actions = {}

        for line_no, iss_list in sorted(file_issues[stem].items()):
            if line_no < 1 or line_no > len(raw_lines):
                continue
            raw_line = raw_lines[line_no - 1]
            codes = {it.code for it in iss_list}

            if codes & drop_codes:
                line_actions[line_no] = ("drop_line", raw_line, "", None)
            elif "BOX_OUT_OF_IMAGE" in codes:
                try:
                    c, cx, cy, w, h = parse_yolo_line(raw_line)
                    x1 = cx - w / 2.0
                    y1 = cy - h / 2.0
                    x2 = cx + w / 2.0
                    y2 = cy + h / 2.0

                    if x2 <= 0.0 or x1 >= 1.0 or y2 <= 0.0 or y1 >= 1.0:
                        line_actions[line_no] = ("drop_line", raw_line, "", None)
                    else:
                        x1_c = max(0.0, min(1.0, x1))
                        x2_c = max(0.0, min(1.0, x2))
                        y1_c = max(0.0, min(1.0, y1))
                        y2_c = max(0.0, min(1.0, y2))
                        new_w = x2_c - x1_c
                        new_h = y2_c - y1_c
                        if new_w <= 0.0 or new_h <= 0.0:
                            line_actions[line_no] = ("drop_line", raw_line, "", None)
                        else:
                            new_cx = (x1_c + x2_c) / 2.0
                            new_cy = (y1_c + y2_c) / 2.0
                            new_str = f"{int(c)} {new_cx:.6f} {new_cy:.6f} {new_w:.6f} {new_h:.6f}"
                            line_actions[line_no] = ("clip_box", raw_line, new_str, new_str)
                except ValueError:
                    line_actions[line_no] = ("drop_line", raw_line, "", None)

        new_file_lines = []
        for idx, orig_line in enumerate(raw_lines, 1):
            if idx in line_actions:
                act, bef, aft, new_str = line_actions[idx]
                if act == "drop_line":
                    continue
                elif act == "clip_box":
                    new_file_lines.append(new_str)
            else:
                new_file_lines.append(orig_line)

        lbl_p.write_text("\n".join(new_file_lines) + ("\n" if new_file_lines else ""), encoding="utf-8")

        for line_no, (act, bef, aft, _) in sorted(line_actions.items()):
            fixes.append({
                "image": stem,
                "line": line_no,
                "action": act,
                "before": bef,
                "after": aft,
            })

    return fixes
