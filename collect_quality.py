# collect_quality.py
# 純手部版 Quality Dataset 收集程式
#
# 功能：
# 1. 讀取 data/<label>/ 底下的圖片與影片
# 2. 使用 MediaPipe Hand Landmarker，最多偵測 2 隻手
# 3. 保存左右手各 21 個 landmarks (x, y, z)
# 4. 計算每隻手的 bbox_area / bbox_w / bbox_h / mean_dist
# 5. quality 欄位故意留空，之後用 label_quality.py 人工標 0 / 1 / 2
#
# 注意：
# - 這一版完全不使用 Pose Landmarker
# - 不需要 pose_landmarker.task
# - 不使用 BBOX_AREA_MIN / MEAN_DIST_MIN 固定門檻過濾壞資料
# - 只要 MediaPipe 能找到手，就盡量保存，讓 Quality Model 之後自己學

import os
import csv
import shutil
from datetime import datetime

import cv2
import numpy as np
import mediapipe as mp

from mediapipe.tasks.python import vision
from mediapipe.tasks.python import BaseOptions


# =====================
# Config
# =====================
BASE_DIR = os.path.dirname(os.path.abspath(__file__))

DATA_DIR = os.path.join(BASE_DIR, "data")
OUTPUT_CSV = os.path.join(BASE_DIR, "quality_dataset.csv")
HAND_TASK_PATH = os.path.join(BASE_DIR, "hand_landmarker.task")

# 影片每幾幀取一次
# 2 = frame 0,2,4,6...
FRAME_INTERVAL = 2

IMAGE_EXT = (".jpg", ".jpeg", ".png", ".bmp", ".webp")
VIDEO_EXT = (".mp4", ".avi", ".mov", ".mkv", ".wmv")

# 最多偵測幾隻手
NUM_HANDS = 2

# MediaPipe 偵測設定
MIN_HAND_DETECTION_CONFIDENCE = 0.3
MIN_HAND_PRESENCE_CONFIDENCE = 0.3
MIN_TRACKING_CONFIDENCE = 0.3

# 如果你的素材全部有做鏡像翻轉，
# 而你發現 Left / Right 全部相反，可以改成 True。
SWAP_LEFT_RIGHT = False

# 如果 MediaPipe 對同一隻實體手重複偵測兩次，
# 以 landmarks 平均 2D 距離判斷是否是重複偵測。
# 你的資料若本來有兩隻非常靠近的手，可把這個值調小。
DUPLICATE_HAND_MEAN_DIST = 0.035


# =====================
# Basic checks
# =====================
def check_required_files():
    if not os.path.isdir(DATA_DIR):
        raise FileNotFoundError(
            f"找不到資料夾：{DATA_DIR}\n"
            "請確認專案底下有 data/ 資料夾。"
        )

    if not os.path.exists(HAND_TASK_PATH):
        raise FileNotFoundError(
            f"找不到 Hand Landmarker 模型：{HAND_TASK_PATH}\n"
            "請確認 hand_landmarker.task 與 collect_quality.py 放在同一層。"
        )


# =====================
# Backup
# =====================
def backup_old_csv_if_needed():
    """
    若 quality_dataset.csv 已存在，先自動備份，
    避免重新收集時直接把舊資料覆蓋掉。
    """
    if not os.path.exists(OUTPUT_CSV):
        return

    stamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    backup_path = os.path.join(
        BASE_DIR,
        f"quality_dataset_backup_{stamp}.csv"
    )

    shutil.copy2(OUTPUT_CSV, backup_path)
    print(f"偵測到舊 CSV，已先備份：{backup_path}")


# =====================
# MediaPipe
# =====================
def create_hand_landmarker():
    options = vision.HandLandmarkerOptions(
        base_options=BaseOptions(
            model_asset_path=HAND_TASK_PATH
        ),
        running_mode=vision.RunningMode.IMAGE,
        num_hands=NUM_HANDS,
        min_hand_detection_confidence=MIN_HAND_DETECTION_CONFIDENCE,
        min_hand_presence_confidence=MIN_HAND_PRESENCE_CONFIDENCE,
        min_tracking_confidence=MIN_TRACKING_CONFIDENCE,
    )

    return vision.HandLandmarker.create_from_options(options)


# =====================
# Utils
# =====================
def bgr_to_mp_image(img_bgr):
    if img_bgr is None:
        return None

    rgb = cv2.cvtColor(img_bgr, cv2.COLOR_BGR2RGB)

    return mp.Image(
        image_format=mp.ImageFormat.SRGB,
        data=rgb
    )


def safe_category_name(category):
    name = getattr(category, "category_name", None)

    if not name:
        name = getattr(category, "display_name", None)

    return str(name or "").strip().lower()


def safe_category_score(category):
    try:
        return float(category.score)
    except Exception:
        return 0.0


def mean_landmark_distance_2d(a, b):
    """
    比較兩組 21 landmarks 是否幾乎重疊。
    """
    if a is None or b is None:
        return float("inf")

    if a.shape != (21, 3) or b.shape != (21, 3):
        return float("inf")

    diff = a[:, :2] - b[:, :2]
    dist = np.sqrt((diff ** 2).sum(axis=1))

    return float(dist.mean())


def detect_hands_bgr(img_bgr, hand_landmarker):
    """
    回傳：
    {
        "left":  np.ndarray shape (21,3) or None,
        "right": np.ndarray shape (21,3) or None,
    }

    注意：
    - 使用 MediaPipe handedness 區分左右手
    - 若同一隻手被重複偵測兩次，會盡量去除重複結果
    """
    output = {
        "left": None,
        "right": None,
    }

    mp_image = bgr_to_mp_image(img_bgr)

    if mp_image is None:
        return output

    result = hand_landmarker.detect(mp_image)

    if not result.hand_landmarks:
        return output

    handedness_list = getattr(result, "handedness", [])

    candidates = []

    for i, lm in enumerate(result.hand_landmarks):
        if len(lm) != 21:
            continue

        data = np.array(
            [[p.x, p.y, p.z] for p in lm],
            dtype=np.float32
        )

        if data.shape != (21, 3):
            continue

        if not np.all(np.isfinite(data)):
            continue

        side = None
        score = 0.0

        if i < len(handedness_list) and handedness_list[i]:
            cat = handedness_list[i][0]

            name = safe_category_name(cat)
            score = safe_category_score(cat)

            if "left" in name:
                side = "left"
            elif "right" in name:
                side = "right"

        candidates.append({
            "data": data,
            "side": side,
            "score": score,
        })

    if not candidates:
        return output

    # handedness 信心較高的先處理
    candidates.sort(
        key=lambda x: x["score"],
        reverse=True
    )

    accepted = []

    for candidate in candidates:
        data = candidate["data"]

        # 避免同一隻實體手被偵測成兩份幾乎重疊的 landmarks
        duplicate = False

        for old in accepted:
            d = mean_landmark_distance_2d(
                data,
                old["data"]
            )

            if d < DUPLICATE_HAND_MEAN_DIST:
                duplicate = True
                break

        if duplicate:
            continue

        accepted.append(candidate)

    # 將候選結果放入 Left / Right
    for candidate in accepted:
        data = candidate["data"]
        side = candidate["side"]

        if SWAP_LEFT_RIGHT and side in ("left", "right"):
            side = "right" if side == "left" else "left"

        # handedness 沒回來時，依空位放入
        if side not in ("left", "right"):
            if output["left"] is None:
                side = "left"
            elif output["right"] is None:
                side = "right"
            else:
                continue

        # 同一側已有資料時，只保留第一個（前面已依 score 排序）
        if output[side] is None:
            output[side] = data
        else:
            other_side = "right" if side == "left" else "left"

            # 如果另一側還是空的，而且不是重複手，
            # 則保留資料到另一側，避免 handedness 分類失敗造成資料遺失。
            if output[other_side] is None:
                output[other_side] = data

    return output


# =====================
# Quality features
# =====================
def mean_pairwise_dist_2d(data):
    """
    data: (21,3)

    計算 21 個 landmark 的平均 2D pairwise distance。
    這裡只計算特徵，不做固定門檻判斷。
    """
    if data is None or data.shape != (21, 3):
        return 0.0

    xy = data[:, :2]

    if not np.all(np.isfinite(xy)):
        return 0.0

    diff = xy[:, None, :] - xy[None, :, :]

    dist = np.sqrt(
        (diff ** 2).sum(axis=2)
    )

    tri = np.triu_indices(21, k=1)

    return float(
        dist[tri].mean()
    )


def hand_quality_features(data):
    """
    只計算 Quality Model 會使用的基本統計特徵。

    回傳：
        bbox_area
        bbox_w
        bbox_h
        mean_dist

    不做：
        area > 0.002
        mean_dist > 0.045
    這類固定 threshold 過濾。
    """
    if data is None or data.shape != (21, 3):
        return 0.0, 0.0, 0.0, 0.0

    xs = data[:, 0].astype(np.float32)
    ys = data[:, 1].astype(np.float32)

    if not np.all(np.isfinite(xs)):
        return 0.0, 0.0, 0.0, 0.0

    if not np.all(np.isfinite(ys)):
        return 0.0, 0.0, 0.0, 0.0

    xmin = float(xs.min())
    xmax = float(xs.max())

    ymin = float(ys.min())
    ymax = float(ys.max())

    width = max(0.0, xmax - xmin)
    height = max(0.0, ymax - ymin)

    area = width * height

    mean_dist = mean_pairwise_dist_2d(data)

    return (
        float(area),
        float(width),
        float(height),
        float(mean_dist),
    )


# =====================
# CSV
# =====================
def build_csv_header():
    """
    純手部版欄位：

    基本資料              4
    Left Hand            63
    Right Hand           63
    Quality Features      8
    quality               1
    -----------------------
    總共                139
    """
    cols = [
        "sequence_id",
        "frame_idx",
        "source_path",
        "label",
    ]

    # Left Hand: 21 x (x,y,z)
    for i in range(21):
        cols.extend([
            f"left_x{i}",
            f"left_y{i}",
            f"left_z{i}",
        ])

    # Right Hand: 21 x (x,y,z)
    for i in range(21):
        cols.extend([
            f"right_x{i}",
            f"right_y{i}",
            f"right_z{i}",
        ])

    cols.extend([
        "left_bbox_area",
        "left_bbox_w",
        "left_bbox_h",
        "left_mean_dist",

        "right_bbox_area",
        "right_bbox_w",
        "right_bbox_h",
        "right_mean_dist",

        # 人工標：
        # 1 = 品質好
        # 0 = 品質差
        # 2 = 不確定
        "quality",
    ])

    return cols


def empty_row(header):
    row = {
        col: 0
        for col in header
    }

    row["source_path"] = ""
    row["quality"] = ""

    return row


def put_hand_into_row(row, side, data):
    if data is None:
        return

    for i in range(21):
        row[f"{side}_x{i}"] = float(data[i, 0])
        row[f"{side}_y{i}"] = float(data[i, 1])
        row[f"{side}_z{i}"] = float(data[i, 2])

    area, width, height, mean_dist = hand_quality_features(data)

    row[f"{side}_bbox_area"] = area
    row[f"{side}_bbox_w"] = width
    row[f"{side}_bbox_h"] = height
    row[f"{side}_mean_dist"] = mean_dist


def make_source_path(path):
    """
    CSV 裡盡量保存相對路徑，例如：
    data/0/example.png

    這樣之後移動整個專案比較方便。
    """
    try:
        rel = os.path.relpath(
            path,
            BASE_DIR
        )
    except Exception:
        rel = path

    return rel.replace("\\", "/")


def build_frame_row(
    img_bgr,
    sequence_id,
    frame_idx,
    source_path,
    label,
    header,
    hand_landmarker,
):
    hands = detect_hands_bgr(
        img_bgr,
        hand_landmarker
    )

    # Quality Model 是在「MediaPipe 已找到手」之後，
    # 判斷 landmarks 品質。
    #
    # 完全沒有任何 hand landmarks 時，
    # 沒有輸入可以交給 Quality Model，所以直接跳過。
    if (
        hands["left"] is None
        and hands["right"] is None
    ):
        return None

    row = empty_row(header)

    row["sequence_id"] = int(sequence_id)
    row["frame_idx"] = int(frame_idx)
    row["source_path"] = make_source_path(source_path)

    if str(label).isdigit():
        row["label"] = int(label)
    else:
        row["label"] = str(label)

    put_hand_into_row(
        row,
        "left",
        hands["left"]
    )

    put_hand_into_row(
        row,
        "right",
        hands["right"]
    )

    # 重要：
    # 不要用 bbox / mean_dist 自動決定 quality。
    # 後續由人工標註。
    row["quality"] = ""

    return row


# =====================
# Image / Video
# =====================
def process_image(
    path,
    label,
    sequence_id,
    header,
    writer,
    hand_landmarker,
):
    img = cv2.imread(path)

    if img is None:
        print(f"  讀取失敗：{path}")
        return 0

    row = build_frame_row(
        img_bgr=img,
        sequence_id=sequence_id,
        frame_idx=0,
        source_path=path,
        label=label,
        header=header,
        hand_landmarker=hand_landmarker,
    )

    if row is None:
        print(
            f"  無 hand landmarks，跳過："
            f"{os.path.basename(path)}"
        )
        return 0

    writer.writerow(row)

    return 1


def process_video(
    path,
    label,
    sequence_id,
    header,
    writer,
    hand_landmarker,
):
    cap = cv2.VideoCapture(path)

    if not cap.isOpened():
        print(f"  無法開啟影片：{path}")
        return 0

    saved = 0
    source_frame_idx = 0

    while True:
        ok, frame = cap.read()

        if not ok:
            break

        if source_frame_idx % FRAME_INTERVAL == 0:
            row = build_frame_row(
                img_bgr=frame,
                sequence_id=sequence_id,
                frame_idx=source_frame_idx,
                source_path=path,
                label=label,
                header=header,
                hand_landmarker=hand_landmarker,
            )

            if row is not None:
                writer.writerow(row)
                saved += 1

        source_frame_idx += 1

    cap.release()

    return saved


# =====================
# Labels
# =====================
def get_label_folders():
    """
    讀取 data/ 底下的子資料夾。

    支援：
      data/0
      data/1
      data/2

    也支援未來：
      data/你好
      data/謝謝
    """
    folders = []

    for name in os.listdir(DATA_DIR):
        path = os.path.join(
            DATA_DIR,
            name
        )

        if os.path.isdir(path):
            folders.append(name)

    def sort_key(name):
        if str(name).isdigit():
            return (0, int(name))

        return (1, str(name))

    folders.sort(
        key=sort_key
    )

    return folders


# =====================
# Main
# =====================
def main():
    check_required_files()
    backup_old_csv_if_needed()

    header = build_csv_header()

    label_folders = get_label_folders()

    if not label_folders:
        raise ValueError(
            "data/ 底下找不到 label 資料夾。\n"
            "例如：data/0、data/1、data/2"
        )

    hand_landmarker = create_hand_landmarker()

    total_rows = 0
    total_sources = 0
    sequence_id = 0

    try:
        with open(
            OUTPUT_CSV,
            "w",
            newline="",
            encoding="utf-8-sig",
        ) as f:

            writer = csv.DictWriter(
                f,
                fieldnames=header
            )

            writer.writeheader()

            for label in label_folders:
                label_dir = os.path.join(
                    DATA_DIR,
                    label
                )

                files = sorted(
                    os.listdir(label_dir)
                )

                print(
                    f"\n處理 label {label}: "
                    f"{label_dir}"
                )

                for file_name in files:
                    path = os.path.join(
                        label_dir,
                        file_name
                    )

                    if not os.path.isfile(path):
                        continue

                    ext = os.path.splitext(
                        file_name
                    )[1].lower()

                    if (
                        ext not in IMAGE_EXT
                        and ext not in VIDEO_EXT
                    ):
                        continue

                    current_sequence_id = sequence_id
                    saved = 0

                    if ext in IMAGE_EXT:
                        saved = process_image(
                            path=path,
                            label=label,
                            sequence_id=current_sequence_id,
                            header=header,
                            writer=writer,
                            hand_landmarker=hand_landmarker,
                        )

                    elif ext in VIDEO_EXT:
                        saved = process_video(
                            path=path,
                            label=label,
                            sequence_id=current_sequence_id,
                            header=header,
                            writer=writer,
                            hand_landmarker=hand_landmarker,
                        )

                    if saved > 0:
                        print(
                            f"  OK：{file_name} | "
                            f"sequence_id={current_sequence_id} | "
                            f"saved_rows={saved}"
                        )

                        total_rows += saved
                        total_sources += 1
                        sequence_id += 1

                    else:
                        print(
                            f"  沒有可用 frame："
                            f"{file_name}"
                        )

    finally:
        hand_landmarker.close()

    print("")
    print("==============================")
    print("Hand Quality Dataset 收集完成")
    print("==============================")
    print(f"輸出：{OUTPUT_CSV}")
    print(f"來源檔案數：{total_sources}")
    print(f"CSV 資料列數：{total_rows}")
    print(f"CSV 欄位數：{len(header)}")
    print("")
    print("這一版只包含 Hand Landmarks，沒有 Pose。")
    print("不需要 pose_landmarker.task。")
    print("")
    print("quality 欄目前故意留空：")
    print("  1 = 手部 landmarks 品質好")
    print("  0 = 手部 landmarks 品質差")
    print("  2 = 不確定")
    print("")
    print("下一步再使用 label_quality.py 人工標註。")


if __name__ == "__main__":
    main()
