# app.py
import os
import json
import base64
import cv2
import torch
import numpy as np
import mediapipe as mp
from flask import Flask, request, jsonify, render_template

from mediapipe.tasks.python import vision
from mediapipe.tasks.python import BaseOptions
from model import HandTransformer


# =====================
# Config
# =====================
BASE_DIR = os.path.dirname(os.path.abspath(__file__))

MODEL_PATH = os.path.join(BASE_DIR, "hand_sequence_model.pth")
LABEL_PATH = os.path.join(BASE_DIR, "label.json")
MP_TASK_PATH = os.path.join(BASE_DIR, "hand_landmarker.task")

SEQ_LEN = 30
DEVICE = "cuda" if torch.cuda.is_available() else "cpu"

BBOX_AREA_MIN = 0.002
MEAN_DIST_MIN = 0.045

app = Flask(__name__)


# =====================
# Load label
# =====================
with open(LABEL_PATH, "r", encoding="utf-8") as f:
    label_map = json.load(f)

keys = [int(k) for k in label_map.keys() if str(k).isdigit()]
num_classes = max(keys) + 1 if keys else len(label_map)


# =====================
# Load model
# =====================
model = HandTransformer(num_classes=num_classes).to(DEVICE)
state = torch.load(MODEL_PATH, map_location=DEVICE)
model.load_state_dict(state, strict=True)
model.eval()


# =====================
# MediaPipe
# =====================
landmarker = None


def get_landmarker():
    global landmarker

    if landmarker is None:
        options = vision.HandLandmarkerOptions(
            base_options=BaseOptions(model_asset_path=MP_TASK_PATH),
            running_mode=vision.RunningMode.IMAGE,
            num_hands=1,
            min_hand_detection_confidence=0.3,
            min_hand_presence_confidence=0.3,
            min_tracking_confidence=0.3,
        )
        landmarker = vision.HandLandmarker.create_from_options(options)

    return landmarker


# =====================
# Preprocess
# =====================
def decode_base64_image(data_url):
    if "," in data_url:
        data_url = data_url.split(",", 1)[1]

    img_bytes = base64.b64decode(data_url)
    np_arr = np.frombuffer(img_bytes, dtype=np.uint8)
    img_bgr = cv2.imdecode(np_arr, cv2.IMREAD_COLOR)

    return img_bgr


def detect_landmarks_bgr(img_bgr):
    if img_bgr is None:
        return None

    rgb = cv2.cvtColor(img_bgr, cv2.COLOR_BGR2RGB)
    mp_image = mp.Image(image_format=mp.ImageFormat.SRGB, data=rgb)

    result = get_landmarker().detect(mp_image)

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
    if len(seq) == 0:
        return None

    seq = np.array(seq, dtype=np.float32)

    if len(seq) == seq_len:
        return seq

    if len(seq) > seq_len:
        idx = np.linspace(0, len(seq) - 1, seq_len).astype(int)
        return seq[idx]

    pad_count = seq_len - len(seq)
    last = seq[-1:]
    pad = np.repeat(last, pad_count, axis=0)

    return np.concatenate([seq, pad], axis=0)


def predict_sequence(seq):
    """
    seq: [30, 21, 3]
    """
    if seq is None:
        return "沒有可用的手部序列"

    if seq.shape != (SEQ_LEN, 21, 3):
        return f"序列 shape 錯誤：{seq.shape}"

    x = torch.tensor(seq, dtype=torch.float32).unsqueeze(0).to(DEVICE)

    with torch.no_grad():
        logits = model(x)
        prob_all = torch.softmax(logits, dim=1)[0]
        pred = int(torch.argmax(prob_all).item())
        prob = float(prob_all[pred].item())

    pred_name = label_map.get(str(pred), f"未知類別({pred})")
    return f"手語意思：{pred_name}"


def predict_from_single_image(img_bgr):
    lm = detect_landmarks_bgr(img_bgr)
    lm_n = bbox_normalize_landmarks(lm)

    if lm_n is None:
        return "沒有偵測到穩定的手，請靠近鏡頭或改善光線"

    seq = np.repeat(lm_n[None, :, :], SEQ_LEN, axis=0)
    return predict_sequence(seq)


def predict_from_frames_base64(frames):
    seq = []

    for frame_data in frames:
        img_bgr = decode_base64_image(frame_data)

        lm = detect_landmarks_bgr(img_bgr)
        lm_n = bbox_normalize_landmarks(lm)

        if lm_n is not None:
            seq.append(lm_n)

    if len(seq) == 0:
        return "30 幀內沒有偵測到可用的手部資料"

    seq = pad_or_sample_sequence(seq, SEQ_LEN)

    return predict_sequence(seq)





# =====================
# Routes
# =====================
@app.route("/", methods=["GET"])
def index():
    return render_template("index.html")


@app.route("/predict_image", methods=["POST"])
def predict_image_route():
    try:
        payload = request.get_json(silent=True) or {}
        image_data = payload.get("image", "")

        if not image_data:
            return jsonify({"ok": False, "result": "沒有收到圖片資料"}), 400

        img_bgr = decode_base64_image(image_data)
        result = predict_from_single_image(img_bgr)

        return jsonify({"ok": True, "result": result})

    except Exception as e:
        return jsonify({"ok": False, "result": f"圖片辨識失敗：{e}"}), 500


@app.route("/predict_sequence", methods=["POST"])
def predict_sequence_route():
    try:
        payload = request.get_json(silent=True) or {}
        frames = payload.get("frames", [])

        if not frames:
            return jsonify({"ok": False, "result": "沒有收到序列影像"}), 400

        result = predict_from_frames_base64(frames)

        return jsonify({"ok": True, "result": result})

    except Exception as e:
        return jsonify({"ok": False, "result": f"序列辨識失敗：{e}"}), 500


if __name__ == "__main__":
    app.run(debug=True, use_reloader=False)