# label_quality.py
# 純手部版 Quality Dataset 人工標註工具
#
# 功能：
# 1. 讀取 quality_dataset.csv
# 2. 依 source_path + frame_idx 顯示原始圖片 / 影片幀
# 3. 疊加 Left / Right Hand 21 點骨架
# 4. 讓你人工標註：
#       1 = 品質好
#       0 = 品質差
#       2 = 不確定
# 5. 每次標註後立即寫回 CSV
#
# 注意：
# - 這一版完全不使用 Pose
# - 對應純手部版 collect_quality.py
# - CSV 預期為 139 欄

import os
import csv
import cv2
import shutil
import numpy as np


# =====================
# Config
# =====================
BASE_DIR = os.path.dirname(os.path.abspath(__file__))

CSV_PATH = os.path.join(BASE_DIR, "quality_dataset.csv")
BACKUP_PATH = os.path.join(
    BASE_DIR,
    "quality_dataset.before_labeling.bak.csv"
)

WINDOW_NAME = "Hand Quality Labeler"

IMAGE_EXT = (
    ".jpg", ".jpeg", ".png",
    ".bmp", ".webp"
)

VIDEO_EXT = (
    ".mp4", ".avi", ".mov",
    ".mkv", ".wmv"
)

# MediaPipe Hand connections
HAND_CONNECTIONS = [
    (0, 1), (1, 2), (2, 3), (3, 4),          # thumb
    (0, 5), (5, 6), (6, 7), (7, 8),          # index
    (0, 9), (9, 10), (10, 11), (11, 12),     # middle
    (0, 13), (13, 14), (14, 15), (15, 16),   # ring
    (0, 17), (17, 18), (18, 19), (19, 20),   # pinky
    (5, 9), (9, 13), (13, 17),               # palm
]


# =====================
# CSV
# =====================
def load_csv(path):
    if not os.path.exists(path):
        raise FileNotFoundError(
            f"找不到 CSV：{path}"
        )

    with open(
        path,
        "r",
        newline="",
        encoding="utf-8-sig",
    ) as f:
        reader = csv.DictReader(f)
        fieldnames = reader.fieldnames
        rows = list(reader)

    if not fieldnames:
        raise ValueError(
            "CSV 沒有 header"
        )

    required = [
        "sequence_id",
        "frame_idx",
        "source_path",
        "label",
        "quality",
    ]

    missing = [
        c for c in required
        if c not in fieldnames
    ]

    if missing:
        raise ValueError(
            f"CSV 缺少必要欄位：{missing}"
        )

    # 檢查左右手 landmark 欄位
    hand_cols = []

    for side in ("left", "right"):
        for i in range(21):
            hand_cols.extend([
                f"{side}_x{i}",
                f"{side}_y{i}",
                f"{side}_z{i}",
            ])

    missing_hand_cols = [
        c for c in hand_cols
        if c not in fieldnames
    ]

    if missing_hand_cols:
        raise ValueError(
            "CSV 缺少手部 landmark 欄位，例如："
            f"{missing_hand_cols[:10]}"
        )

    return fieldnames, rows


def save_csv(
    path,
    fieldnames,
    rows,
):
    temp_path = path + ".tmp"

    with open(
        temp_path,
        "w",
        newline="",
        encoding="utf-8-sig",
    ) as f:

        writer = csv.DictWriter(
            f,
            fieldnames=fieldnames
        )

        writer.writeheader()
        writer.writerows(rows)

    os.replace(
        temp_path,
        path
    )


def make_backup():
    if not os.path.exists(BACKUP_PATH):
        shutil.copy2(
            CSV_PATH,
            BACKUP_PATH
        )

        print(
            f"已建立標註前備份："
            f"{BACKUP_PATH}"
        )


# =====================
# Source path / frame
# =====================
def resolve_source_path(source_path):
    source_path = str(
        source_path or ""
    ).strip()

    if not source_path:
        return ""

    source_path = source_path.replace(
        "/",
        os.sep
    )

    if os.path.isabs(source_path):
        return source_path

    return os.path.normpath(
        os.path.join(
            BASE_DIR,
            source_path
        )
    )


def load_source_frame(row):
    path = resolve_source_path(
        row.get("source_path", "")
    )

    if not path:
        return None, "source_path 是空的"

    if not os.path.exists(path):
        return None, (
            f"找不到來源檔案：{path}"
        )

    ext = os.path.splitext(
        path
    )[1].lower()

    # 圖片
    if ext in IMAGE_EXT:
        img = cv2.imread(path)

        if img is None:
            return None, (
                f"無法讀取圖片：{path}"
            )

        return img, ""

    # 影片
    if ext in VIDEO_EXT:
        try:
            frame_idx = int(
                float(
                    row.get(
                        "frame_idx",
                        0
                    ) or 0
                )
            )
        except Exception:
            frame_idx = 0

        cap = cv2.VideoCapture(
            path
        )

        if not cap.isOpened():
            return None, (
                f"無法開啟影片：{path}"
            )

        cap.set(
            cv2.CAP_PROP_POS_FRAMES,
            frame_idx
        )

        ok, frame = cap.read()
        cap.release()

        if not ok or frame is None:
            return None, (
                f"無法讀取影片 frame "
                f"{frame_idx}：{path}"
            )

        return frame, ""

    return None, (
        f"不支援的來源格式：{path}"
    )


# =====================
# Landmark parsing
# =====================
def safe_float(
    value,
    default=0.0,
):
    try:
        v = float(value)

        if np.isfinite(v):
            return v
    except Exception:
        pass

    return default


def get_hand_xyz(
    row,
    side,
):
    pts = []

    for i in range(21):
        x = safe_float(
            row.get(
                f"{side}_x{i}",
                0
            )
        )

        y = safe_float(
            row.get(
                f"{side}_y{i}",
                0
            )
        )

        z = safe_float(
            row.get(
                f"{side}_z{i}",
                0
            )
        )

        pts.append(
            [x, y, z]
        )

    pts = np.asarray(
        pts,
        dtype=np.float32
    )

    # collect_quality.py 對沒有偵測到的手會全部填 0
    present = bool(
        np.any(
            np.abs(pts) > 1e-8
        )
    )

    return pts, present


def mean_hand_distance(
    left_pts,
    right_pts,
):
    """
    幫你顯示左右手骨架是否幾乎重疊。
    只作為人工檢查提示，不會自動改 quality。
    """
    if (
        left_pts is None
        or right_pts is None
    ):
        return None

    diff = (
        left_pts[:, :2]
        - right_pts[:, :2]
    )

    dist = np.sqrt(
        (diff ** 2).sum(axis=1)
    )

    return float(
        dist.mean()
    )


# =====================
# Drawing
# =====================
def norm_to_px(
    x,
    y,
    width,
    height,
):
    px = int(
        round(x * width)
    )

    py = int(
        round(y * height)
    )

    return px, py


def in_reasonable_range(
    x,
    y,
):
    # MediaPipe 偶爾可能略超出 0~1，
    # 留一些容忍值
    return (
        -0.2 <= x <= 1.2
        and -0.2 <= y <= 1.2
    )


def draw_hand(
    img,
    pts,
    present,
    side,
):
    if not present:
        return

    h, w = img.shape[:2]

    if side == "left":
        color = (0, 255, 255)   # 黃
        text = "LEFT"
    else:
        color = (255, 255, 0)   # 青
        text = "RIGHT"

    # 骨架線
    for a, b in HAND_CONNECTIONS:
        xa, ya = pts[a][:2]
        xb, yb = pts[b][:2]

        if not (
            in_reasonable_range(
                xa, ya
            )
            and
            in_reasonable_range(
                xb, yb
            )
        ):
            continue

        pa = norm_to_px(
            xa, ya, w, h
        )

        pb = norm_to_px(
            xb, yb, w, h
        )

        cv2.line(
            img,
            pa,
            pb,
            color,
            2,
            cv2.LINE_AA,
        )

    # 點
    for i, pt in enumerate(pts):
        x = float(pt[0])
        y = float(pt[1])

        if not in_reasonable_range(
            x, y
        ):
            continue

        p = norm_to_px(
            x, y, w, h
        )

        cv2.circle(
            img,
            p,
            4,
            color,
            -1,
            cv2.LINE_AA,
        )

        # 只標重要點，避免太亂
        if i in (
            0, 4, 8,
            12, 16, 20
        ):
            cv2.putText(
                img,
                str(i),
                (
                    p[0] + 4,
                    p[1] - 4
                ),
                cv2.FONT_HERSHEY_SIMPLEX,
                0.45,
                color,
                1,
                cv2.LINE_AA,
            )

    # 顯示 LEFT / RIGHT
    valid_xy = []

    for pt in pts:
        x = float(pt[0])
        y = float(pt[1])

        if in_reasonable_range(
            x, y
        ):
            valid_xy.append(
                [x, y]
            )

    if valid_xy:
        valid_xy = np.asarray(
            valid_xy,
            dtype=np.float32
        )

        xmin = float(
            valid_xy[:, 0].min()
        )

        ymin = float(
            valid_xy[:, 1].min()
        )

        p = norm_to_px(
            xmin,
            ymin,
            w,
            h
        )

        cv2.putText(
            img,
            text,
            (
                max(5, p[0]),
                max(20, p[1] - 10)
            ),
            cv2.FONT_HERSHEY_SIMPLEX,
            0.65,
            color,
            2,
            cv2.LINE_AA,
        )


def add_info_panel(
    img,
    row,
    index,
    total,
    labeled_count,
    left_present,
    right_present,
    overlap_distance,
):
    h, w = img.shape[:2]

    panel_h = 190

    canvas = np.zeros(
        (
            h + panel_h,
            w,
            3
        ),
        dtype=np.uint8
    )

    canvas[:h] = img

    quality = str(
        row.get(
            "quality",
            ""
        )
    ).strip()

    quality_display = (
        quality
        if quality in (
            "0", "1", "2"
        )
        else "EMPTY"
    )

    seq = str(
        row.get(
            "sequence_id",
            ""
        )
    )

    frame_idx = str(
        row.get(
            "frame_idx",
            ""
        )
    )

    label = str(
        row.get(
            "label",
            ""
        )
    )

    source = str(
        row.get(
            "source_path",
            ""
        )
    )

    left_area = safe_float(
        row.get(
            "left_bbox_area",
            0
        )
    )

    left_dist = safe_float(
        row.get(
            "left_mean_dist",
            0
        )
    )

    right_area = safe_float(
        row.get(
            "right_bbox_area",
            0
        )
    )

    right_dist = safe_float(
        row.get(
            "right_mean_dist",
            0
        )
    )

    if overlap_distance is None:
        overlap_text = "N/A"
    else:
        overlap_text = (
            f"{overlap_distance:.4f}"
        )

    lines = [
        (
            f"Row {index + 1}/{total}   "
            f"labeled {labeled_count}/{total}   "
            f"quality={quality_display}"
        ),
        (
            f"sequence={seq}   "
            f"frame={frame_idx}   "
            f"sign_label={label}"
        ),
        (
            f"Left={left_present}   "
            f"Right={right_present}   "
            f"L/R overlap_dist={overlap_text}"
        ),
        (
            f"L area={left_area:.5f} "
            f"dist={left_dist:.5f}   "
            f"R area={right_area:.5f} "
            f"dist={right_dist:.5f}"
        ),
        (
            f"source: {source}"
        ),
        (
            "Keys: "
            "1=GOOD  0=BAD  2=UNCERTAIN  "
            "A=PREV  D=NEXT  N=NEXT EMPTY  "
            "C=CLEAR  S=SAVE  Q/ESC=QUIT"
        ),
    ]

    y = h + 25

    for line in lines:
        cv2.putText(
            canvas,
            line,
            (12, y),
            cv2.FONT_HERSHEY_SIMPLEX,
            0.53,
            (255, 255, 255),
            1,
            cv2.LINE_AA,
        )

        y += 27

    return canvas


def fit_to_screen(
    img,
    max_w=1400,
    max_h=950,
):
    h, w = img.shape[:2]

    scale = min(
        max_w / w,
        max_h / h,
        1.0
    )

    if scale >= 1.0:
        return img

    new_w = max(
        1,
        int(w * scale)
    )

    new_h = max(
        1,
        int(h * scale)
    )

    return cv2.resize(
        img,
        (new_w, new_h),
        interpolation=cv2.INTER_AREA
    )


# =====================
# Navigation
# =====================
def is_labeled(row):
    return str(
        row.get(
            "quality",
            ""
        )
    ).strip() in (
        "0", "1", "2"
    )


def count_labeled(rows):
    return sum(
        is_labeled(r)
        for r in rows
    )


def first_unlabeled(rows):
    for i, row in enumerate(rows):
        if not is_labeled(row):
            return i

    return 0


def find_next_unlabeled(
    rows,
    start_idx,
):
    n = len(rows)

    if n == 0:
        return None

    for step in range(
        1,
        n + 1
    ):
        idx = (
            start_idx + step
        ) % n

        if not is_labeled(
            rows[idx]
        ):
            return idx

    return None


# =====================
# Main
# =====================
def main():
    fieldnames, rows = load_csv(
        CSV_PATH
    )

    if not rows:
        raise ValueError(
            "quality_dataset.csv 沒有資料列"
        )

    make_backup()

    idx = first_unlabeled(
        rows
    )

    print("")
    print("==============================")
    print("Hand Quality Dataset 人工標註")
    print("==============================")
    print(f"CSV：{CSV_PATH}")
    print(f"總資料數：{len(rows)}")
    print(
        f"已標註："
        f"{count_labeled(rows)}"
    )
    print("")
    print("這一版只判斷『手部 landmarks 品質』")
    print("不包含 Pose。")
    print("")
    print("判斷原則：")
    print("  1 = 手部骨架完整、關節位置合理")
    print("  0 = 手部骨架明顯跑掉、扭曲、錯位")
    print("  2 = 不確定")
    print("")
    print("注意：")
    print("  手語比錯 ≠ 品質差")
    print("  單手手語只有一隻手 ≠ 品質差")
    print("")
    print("按鍵：")
    print("  1 = 品質好")
    print("  0 = 品質差")
    print("  2 = 不確定")
    print("  A = 上一筆")
    print("  D = 下一筆")
    print("  N = 下一筆尚未標註")
    print("  C = 清除目前 quality")
    print("  S = 手動儲存")
    print("  Q / ESC = 儲存並離開")
    print("")

    cv2.namedWindow(
        WINDOW_NAME,
        cv2.WINDOW_NORMAL
    )

    while True:
        row = rows[idx]

        frame, error = load_source_frame(
            row
        )

        if frame is None:
            frame = np.zeros(
                (
                    600,
                    1000,
                    3
                ),
                dtype=np.uint8
            )

            cv2.putText(
                frame,
                "SOURCE LOAD FAILED",
                (30, 80),
                cv2.FONT_HERSHEY_SIMPLEX,
                1.2,
                (255, 255, 255),
                2,
                cv2.LINE_AA,
            )

            cv2.putText(
                frame,
                error[:120],
                (30, 130),
                cv2.FONT_HERSHEY_SIMPLEX,
                0.55,
                (255, 255, 255),
                1,
                cv2.LINE_AA,
            )

        display = frame.copy()

        left_pts, left_present = get_hand_xyz(
            row,
            "left"
        )

        right_pts, right_present = get_hand_xyz(
            row,
            "right"
        )

        draw_hand(
            display,
            left_pts,
            left_present,
            "left"
        )

        draw_hand(
            display,
            right_pts,
            right_present,
            "right"
        )

        overlap_distance = None

        if (
            left_present
            and right_present
        ):
            overlap_distance = mean_hand_distance(
                left_pts,
                right_pts
            )

        labeled = count_labeled(
            rows
        )

        display = add_info_panel(
            display,
            row,
            idx,
            len(rows),
            labeled,
            left_present,
            right_present,
            overlap_distance,
        )

        display = fit_to_screen(
            display
        )

        cv2.imshow(
            WINDOW_NAME,
            display
        )

        key = cv2.waitKeyEx(0)

        # Q / q / ESC
        if key in (
            27,
            ord("q"),
            ord("Q"),
        ):
            save_csv(
                CSV_PATH,
                fieldnames,
                rows
            )

            print(
                "已儲存並離開。"
            )

            break

        # 0 / 1 / 2
        if key in (
            ord("0"),
            ord("1"),
            ord("2"),
        ):
            value = chr(key)

            rows[idx]["quality"] = value

            save_csv(
                CSV_PATH,
                fieldnames,
                rows
            )

            print(
                f"[{idx + 1}/{len(rows)}] "
                f"quality={value} | "
                f"{row.get('source_path', '')} | "
                f"frame={row.get('frame_idx', '')}"
            )

            next_idx = find_next_unlabeled(
                rows,
                idx
            )

            if next_idx is None:
                save_csv(
                    CSV_PATH,
                    fieldnames,
                    rows
                )

                print("")
                print("==============================")
                print("全部資料已標註完成！")
                print(f"總資料數：{len(rows)}")
                print(f"已標註：{count_labeled(rows)}")
                print("CSV 已儲存。")
                print("==============================")

                break

            idx = next_idx
            continue

        # C = 清除
        if key in (
            ord("c"),
            ord("C"),
        ):
            rows[idx]["quality"] = ""

            save_csv(
                CSV_PATH,
                fieldnames,
                rows
            )

            print(
                f"[{idx + 1}/{len(rows)}] "
                "quality 已清除"
            )

            continue

        # S = 手動儲存
        if key in (
            ord("s"),
            ord("S"),
        ):
            save_csv(
                CSV_PATH,
                fieldnames,
                rows
            )

            print(
                "已儲存。"
            )

            continue

        # N = 下一筆未標註
        if key in (
            ord("n"),
            ord("N"),
        ):
            idx = find_next_unlabeled(
                rows,
                idx
            )

            continue

        # A / 左方向鍵 = 上一筆
        if key in (
            ord("a"),
            ord("A"),
            2424832,
        ):
            idx = max(
                0,
                idx - 1
            )

            continue

        # D / 右方向鍵 = 下一筆
        if key in (
            ord("d"),
            ord("D"),
            2555904,
        ):
            idx = min(
                len(rows) - 1,
                idx + 1
            )

            continue

    cv2.destroyAllWindows()


if __name__ == "__main__":
    main()
