# eval.py
import os
import json
import numpy as np
import torch
from torch.utils.data import Dataset, DataLoader

from model import HandTransformer


# =====================
# Config
# =====================
X_PATH = "hand_sequences.npy"
Y_PATH = "hand_labels.npy"
MODEL_PATH = "hand_sequence_model.pth"
LABEL_JSON_PATH = "label.json"

BATCH_SIZE = 256
DEVICE = "cuda" if torch.cuda.is_available() else "cpu"


# =====================
# Dataset
# =====================
class HandSequenceDataset(Dataset):
    def __init__(self, x_path, y_path):
        if not os.path.exists(x_path):
            raise FileNotFoundError(f"找不到資料檔：{x_path}")

        if not os.path.exists(y_path):
            raise FileNotFoundError(f"找不到標籤檔：{y_path}")

        self.X = np.load(x_path).astype(np.float32)
        self.y = np.load(y_path).astype(np.int64)

        if len(self.X) != len(self.y):
            raise ValueError(f"X 和 y 數量不一致：X={len(self.X)}, y={len(self.y)}")

        if self.X.ndim != 4:
            raise ValueError(f"X 應該是 [N,T,21,3]，但得到 shape={self.X.shape}")

        if self.X.shape[2] != 21 or self.X.shape[3] != 3:
            raise ValueError(f"X 應該是 [N,T,21,3]，但得到 shape={self.X.shape}")

    def __len__(self):
        return len(self.y)

    def __getitem__(self, idx):
        x = torch.tensor(self.X[idx], dtype=torch.float32)
        y = torch.tensor(self.y[idx], dtype=torch.long)
        return x, y


# =====================
# Utils
# =====================
def load_label_map(path):
    if not os.path.exists(path):
        return None

    with open(path, "r", encoding="utf-8") as f:
        return json.load(f)


def unwrap_state_dict(ckpt):
    """
    支援：
    1. torch.save(model.state_dict(), path)
    2. torch.save({"model": model.state_dict(), ...}, path)
    3. torch.save({"state_dict": model.state_dict(), ...}, path)
    4. torch.save({"model_state_dict": model.state_dict(), ...}, path)
    """
    if isinstance(ckpt, dict):
        if any(k.endswith(".weight") or k.endswith(".bias") for k in ckpt.keys()):
            return ckpt

        for key in ["model", "state_dict", "model_state_dict"]:
            if key in ckpt and isinstance(ckpt[key], dict):
                return ckpt[key]

    raise ValueError("Unsupported checkpoint format: cannot find state_dict.")


def precision_recall_from_cm(cm):
    row_sum = cm.sum(axis=1)
    col_sum = cm.sum(axis=0)

    recall = np.divide(
        np.diag(cm),
        row_sum,
        out=np.zeros_like(row_sum, dtype=float),
        where=row_sum != 0
    )

    precision = np.divide(
        np.diag(cm),
        col_sum,
        out=np.zeros_like(col_sum, dtype=float),
        where=col_sum != 0
    )

    return precision, recall


@torch.no_grad()
def evaluate(model, loader, num_classes, device):
    model.eval()

    cm = np.zeros((num_classes, num_classes), dtype=np.int64)

    correct = 0
    total = 0

    for x, y in loader:
        x = x.to(device, non_blocking=True)
        y = y.to(device, non_blocking=True)

        logits = model(x)
        pred = torch.argmax(logits, dim=1)

        correct += (pred == y).sum().item()
        total += y.numel()

        y_np = y.detach().cpu().numpy()
        pred_np = pred.detach().cpu().numpy()

        for true_label, pred_label in zip(y_np, pred_np):
            cm[true_label, pred_label] += 1

    acc = correct / total if total > 0 else 0.0
    return acc, cm


def main():
    dataset = HandSequenceDataset(X_PATH, Y_PATH)

    labels = dataset.y
    unique_labels = sorted(np.unique(labels))

    if min(unique_labels) != 0:
        raise ValueError("label 必須從 0 開始")

    if max(unique_labels) != len(unique_labels) - 1:
        raise ValueError("label 必須連續，例如 0,1,2,3")

    num_classes = len(unique_labels)

    loader = DataLoader(
        dataset,
        batch_size=BATCH_SIZE,
        shuffle=False,
        num_workers=0,
        pin_memory=torch.cuda.is_available()
    )

    if not os.path.exists(MODEL_PATH):
        raise FileNotFoundError(f"找不到模型權重：{MODEL_PATH}")

    model = HandTransformer(num_classes=num_classes).to(DEVICE)

    ckpt = torch.load(MODEL_PATH, map_location=DEVICE)
    state_dict = unwrap_state_dict(ckpt)
    model.load_state_dict(state_dict, strict=True)
    model.eval()

    label_map = load_label_map(LABEL_JSON_PATH)

    print("Eval start")
    print(f"X shape: {dataset.X.shape}")
    print(f"y shape: {dataset.y.shape}")
    print(f"資料數: {len(dataset)}")
    print(f"類別數: {num_classes}")
    print(f"DEVICE: {DEVICE}")
    print(f"MODEL: {MODEL_PATH}")

    acc, cm = evaluate(
        model=model,
        loader=loader,
        num_classes=num_classes,
        device=DEVICE
    )

    precision, recall = precision_recall_from_cm(cm)

    print("\nConfusion Matrix:")
    print(cm)

    print(
        f"\n整體準確率: {acc * 100:.2f}% "
        f"(total={cm.sum()}, correct={np.trace(cm)})"
    )

    print("\n各類別指標：")
    for i in range(num_classes):
        name = ""

        if label_map and str(i) in label_map:
            name = f" ({label_map[str(i)]})"

        print(
            f"class {i}{name}: "
            f"precision={precision[i] * 100:.2f}% | "
            f"recall={recall[i] * 100:.2f}% | "
            f"total={cm[i].sum()}"
        )


if __name__ == "__main__":
    main()