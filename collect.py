# collect.py
import os
import json
import cv2
import numpy as np
import mediapipe as mp

from mediapipe.tasks.python import vision
from mediapipe.tasks.python import BaseOptions

# =====================
# Config
# =====================
DATA_DIR = "data"
OUTPUT_X = "hand_sequences.npy"
OUTPUT_Y = "hand_labels.npy"
OUTPUT_META = "hand_sequences_meta.json"

MP_TASK_PATH = "hand_landmarker.task"

SEQ_LEN = 30
FRAME_INTERVAL = 2

IMAGE_EXT = (".jpg", ".jpeg", ".png")
VIDEO_EXT = (".mp4", ".avi", ".mov", ".mkv")

BBOX_AREA_MIN = 0.002
MEAN_DIST_MIN = 0.045

# =====================
# MediaPipe
# =====================
options = vision.HandLandmarkerOptions(
    base_options=BaseOptions(model_asset_path=MP_TASK_PATH),
    running_mode=vision.RunningMode.IMAGE,
    num_hands=1,
    min_hand_detection_confidence=0.3,
    min_hand_presence_confidence=0.3,
    min_tracking_confidence=0.3,
)

landmarker = vision.HandLandmarker.create_from_options(options)


# =====================
# Utils
# =====================
def detect_landmarks_bgr(img_bgr):
    if img_bgr is None:
        return None

    rgb = cv2.cvtColor(img_bgr, cv2.COLOR_BGR2RGB)
    mp_image = mp.Image(image_format=mp.ImageFormat.SRGB, data=rgb)

    result = landmarker.detect(mp_image)

    if not result.hand_landmarks:
        return None

    lm = result.hand_landmarks[0]

    if len(lm) != 21:
        return None

    data = np.array([[p.x, p.y, p.z] for p in lm], dtype=np.float32)
    return data


def mean_pairwise_dist_2d(data):
    xy = data[:, :2]

    if not np.all(np.isfinite(xy)):
        return 0.0

    diff = xy[:, None, :] - xy[None, :, :]
    dist = np.sqrt((diff ** 2).sum(axis=2))

    tri = np.triu_indices(21, k=1)
    return float(dist[tri].mean())


def bbox_normalize_landmarks(data, eps=1e-6):
    if data is None or data.shape != (21, 3):
        return None

    xs = data[:, 0].astype(np.float32)
    ys = data[:, 1].astype(np.float32)
    zs = data[:, 2].astype(np.float32)

    xmin, xmax = float(xs.min()), float(xs.max())
    ymin, ymax = float(ys.min()), float(ys.max())

    w = xmax - xmin
    h = ymax - ymin
    area = w * h

    if not np.isfinite(area) or w <= eps or h <= eps:
        return None

    if area <= BBOX_AREA_MIN:
        return None

    mean_dist = mean_pairwise_dist_2d(data)
    if mean_dist <= MEAN_DIST_MIN:
        return None

    xs_n = (xs - xmin) / (w + eps)
    ys_n = (ys - ymin) / (h + eps)

    s = max(w, h) + eps
    zs_n = zs / s

    data_n = np.stack([xs_n, ys_n, zs_n], axis=1).astype(np.float32)

    if not np.all(np.isfinite(data_n)):
        return None

    return data_n


def pad_or_sample_sequence(seq, seq_len=SEQ_LEN):
    """
    seq: list of (21,3)
    return: (SEQ_LEN,21,3)
    """
    if len(seq) == 0:
        return None

    seq = np.array(seq, dtype=np.float32)

    if len(seq) == seq_len:
        return seq

    if len(seq) > seq_len:
        idx = np.linspace(0, len(seq) - 1, seq_len).astype(int)
        return seq[idx]

    # 不足 SEQ_LEN：用最後一幀補滿
    pad_count = seq_len - len(seq)
    last = seq[-1:]
    pad = np.repeat(last, pad_count, axis=0)

    return np.concatenate([seq, pad], axis=0)


def process_image(path):
    img = cv2.imread(path)

    lm = detect_landmarks_bgr(img)
    lm_n = bbox_normalize_landmarks(lm)

    if lm_n is None:
        return None

    # 靜態圖片：同一幀重複 SEQ_LEN 次
    seq = np.repeat(lm_n[None, :, :], SEQ_LEN, axis=0)
    return seq.astype(np.float32)


def process_video(path):
    cap = cv2.VideoCapture(path)

    if not cap.isOpened():
        return None

    seq = []
    frame_idx = 0

    while True:
        ret, frame = cap.read()

        if not ret:
            break

        if frame_idx % FRAME_INTERVAL == 0:
            lm = detect_landmarks_bgr(frame)
            lm_n = bbox_normalize_landmarks(lm)

            if lm_n is not None:
                seq.append(lm_n)

        frame_idx += 1

    cap.release()

    if len(seq) == 0:
        return None

    return pad_or_sample_sequence(seq, SEQ_LEN).astype(np.float32)


def save_label_json(label_map):
    with open("label.json", "w", encoding="utf-8") as f:
        json.dump(label_map, f, ensure_ascii=False, indent=2)


# =====================
# Main
# =====================
def main():
    if not os.path.isdir(DATA_DIR):
        raise FileNotFoundError(f"找不到資料夾：{DATA_DIR}")

    X = []
    y = []
    meta = []

    label_map = {}

    labels = sorted([
        d for d in os.listdir(DATA_DIR)
        if os.path.isdir(os.path.join(DATA_DIR, d)) and d.isdigit()
    ], key=lambda x: int(x))

    if len(labels) == 0:
        raise ValueError("data/ 底下找不到數字 label 資料夾，例如 data/0, data/1")

    for label in labels:
        label_int = int(label)
        label_dir = os.path.join(DATA_DIR, label)

        label_map[str(label_int)] = label

        print(f"\n處理 label {label_int}: {label_dir}")

        files = sorted(os.listdir(label_dir))

        for file_name in files:
            path = os.path.join(label_dir, file_name)
            ext = os.path.splitext(file_name)[1].lower()

            seq = None
            sample_type = None

            if ext in IMAGE_EXT:
                seq = process_image(path)
                sample_type = "image"

            elif ext in VIDEO_EXT:
                seq = process_video(path)
                sample_type = "video"

            else:
                continue

            if seq is None:
                print(f"  跳過：{file_name}")
                continue

            if seq.shape != (SEQ_LEN, 21, 3):
                print(f"  shape 錯誤，跳過：{file_name}, shape={seq.shape}")
                continue

            X.append(seq)
            y.append(label_int)

            meta.append({
                "label": label_int,
                "file": path.replace("\\", "/"),
                "type": sample_type,
                "seq_len": SEQ_LEN,
            })

            print(f"  OK：{file_name} -> {seq.shape}")

    if len(X) == 0:
        raise ValueError("沒有成功收集到任何資料")

    X = np.array(X, dtype=np.float32)
    y = np.array(y, dtype=np.int64)

    np.save(OUTPUT_X, X)
    np.save(OUTPUT_Y, y)

    with open(OUTPUT_META, "w", encoding="utf-8") as f:
        json.dump(meta, f, ensure_ascii=False, indent=2)

    save_label_json(label_map)

    print("\n完成")
    print(f"X saved: {OUTPUT_X}, shape={X.shape}")
    print(f"y saved: {OUTPUT_Y}, shape={y.shape}")
    print(f"meta saved: {OUTPUT_META}")
    print(f"label.json saved")
    print(f"SEQ_LEN={SEQ_LEN}")
    print(f"FRAME_INTERVAL={FRAME_INTERVAL}")


if __name__ == "__main__":
    try:
        main()
    finally:
        landmarker.close()