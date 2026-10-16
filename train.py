# train.py
import os
import json
import random
import numpy as np
import torch
import torch.nn as nn
from torch.utils.data import Dataset, DataLoader, Subset

from model import HandTransformer


# =====================
# Config
# =====================
SEED = 42

X_PATH = "hand_sequences.npy"
Y_PATH = "hand_labels.npy"
LABEL_JSON_PATH = "label.json"

MODEL_SAVE_PATH = "hand_sequence_model.pth"
TRAIN_INFO_PATH = "train_info.json"

EPOCHS = 50
BATCH_SIZE = 16
LR = 1e-3
WEIGHT_DECAY = 1e-4
EARLY_STOPPING_PATIENCE = 10

TRAIN_RATIO = 0.8

DEVICE = "cuda" if torch.cuda.is_available() else "cpu"


# =====================
# Reproducibility
# =====================
random.seed(SEED)
np.random.seed(SEED)
torch.manual_seed(SEED)

if torch.cuda.is_available():
    torch.cuda.manual_seed_all(SEED)

torch.backends.cudnn.deterministic = True
torch.backends.cudnn.benchmark = False


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

        if not np.all(np.isfinite(self.X)):
            raise ValueError("X 裡面有 NaN 或 inf")

        if not np.all(np.isfinite(self.y)):
            raise ValueError("y 裡面有 NaN 或 inf")

    def __len__(self):
        return len(self.y)

    def __getitem__(self, idx):
        x = torch.tensor(self.X[idx], dtype=torch.float32)
        y = torch.tensor(self.y[idx], dtype=torch.long)
        return x, y


# =====================
# Split
# =====================
def stratified_split_indices(labels, train_ratio=0.8, seed=42):
    rng = np.random.default_rng(seed)

    labels = np.asarray(labels)
    train_indices = []
    test_indices = []

    for label in sorted(np.unique(labels)):
        idx = np.where(labels == label)[0]
        rng.shuffle(idx)

        n_train = int(len(idx) * train_ratio)

        if len(idx) >= 2:
            n_train = max(1, min(n_train, len(idx) - 1))
        else:
            n_train = len(idx)

        train_indices.extend(idx[:n_train])
        test_indices.extend(idx[n_train:])

    rng.shuffle(train_indices)
    rng.shuffle(test_indices)

    return train_indices, test_indices


# =====================
# Metrics
# =====================
@torch.no_grad()
def evaluate(model, loader, criterion, num_classes, device):
    model.eval()

    total_loss = 0.0
    total_count = 0

    cm = np.zeros((num_classes, num_classes), dtype=np.int64)

    for x, y in loader:
        x = x.to(device, non_blocking=True)
        y = y.to(device, non_blocking=True)

        logits = model(x)
        loss = criterion(logits, y)

        batch_size = y.size(0)
        total_loss += loss.item() * batch_size
        total_count += batch_size

        pred = torch.argmax(logits, dim=1)

        y_np = y.detach().cpu().numpy()
        pred_np = pred.detach().cpu().numpy()

        for true_label, pred_label in zip(y_np, pred_np):
            cm[true_label, pred_label] += 1

    avg_loss = total_loss / total_count if total_count > 0 else 0.0
    acc = np.trace(cm) / cm.sum() if cm.sum() > 0 else 0.0

    return avg_loss, acc, cm


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


def load_label_map(path):
    if not os.path.exists(path):
        return None

    with open(path, "r", encoding="utf-8") as f:
        return json.load(f)


# =====================
# Main
# =====================
def main():
    dataset = HandSequenceDataset(X_PATH, Y_PATH)

    labels = dataset.y
    unique_labels = sorted(np.unique(labels))

    if min(unique_labels) != 0:
        raise ValueError("label 必須從 0 開始")

    if max(unique_labels) != len(unique_labels) - 1:
        raise ValueError("label 必須連續，例如 0,1,2,3")

    num_classes = len(unique_labels)

    train_indices, test_indices = stratified_split_indices(
        labels,
        train_ratio=TRAIN_RATIO,
        seed=SEED
    )

    train_dataset = Subset(dataset, train_indices)
    test_dataset = Subset(dataset, test_indices)

    train_loader = DataLoader(
        train_dataset,
        batch_size=BATCH_SIZE,
        shuffle=True,
        num_workers=0,
        pin_memory=torch.cuda.is_available()
    )

    test_loader = DataLoader(
        test_dataset,
        batch_size=256,
        shuffle=False,
        num_workers=0,
        pin_memory=torch.cuda.is_available()
    )

    all_loader = DataLoader(
        dataset,
        batch_size=256,
        shuffle=False,
        num_workers=0,
        pin_memory=torch.cuda.is_available()
    )

    print("資料載入完成")
    print(f"X shape: {dataset.X.shape}")
    print(f"y shape: {dataset.y.shape}")
    print(f"總資料數: {len(dataset)}")
    print(f"訓練資料數: {len(train_dataset)}")
    print(f"測試資料數: {len(test_dataset)}")
    print(f"類別數: {num_classes}")
    print(f"DEVICE: {DEVICE}")

    label_map = load_label_map(LABEL_JSON_PATH)
    if label_map:
        print("Label map:")
        for k in sorted(label_map.keys(), key=lambda x: int(x)):
            print(f"  {k}: {label_map[k]}")

    model = HandTransformer(num_classes=num_classes).to(DEVICE)

    criterion = nn.CrossEntropyLoss()

    optimizer = torch.optim.Adam(
        model.parameters(),
        lr=LR,
        weight_decay=WEIGHT_DECAY
    )

    scheduler = torch.optim.lr_scheduler.ReduceLROnPlateau(
        optimizer,
        mode="min",
        factor=0.5,
        patience=3
    )

    best_acc = -1.0
    best_loss = float("inf")
    best_epoch = -1
    no_improve = 0

    history = []

    for epoch in range(1, EPOCHS + 1):
        model.train()

        train_loss_sum = 0.0
        train_count = 0

        for x, y in train_loader:
            x = x.to(DEVICE, non_blocking=True)
            y = y.to(DEVICE, non_blocking=True)

            optimizer.zero_grad()

            logits = model(x)
            loss = criterion(logits, y)

            loss.backward()
            optimizer.step()

            batch_size = y.size(0)
            train_loss_sum += loss.item() * batch_size
            train_count += batch_size

        train_loss = train_loss_sum / train_count if train_count > 0 else 0.0

        test_loss, test_acc, _ = evaluate(
            model,
            test_loader,
            criterion,
            num_classes,
            DEVICE
        )

        scheduler.step(test_loss)

        current_lr = optimizer.param_groups[0]["lr"]

        history.append({
            "epoch": epoch,
            "train_loss": float(train_loss),
            "test_loss": float(test_loss),
            "test_acc": float(test_acc),
            "lr": float(current_lr),
        })

        print(
            f"Epoch {epoch:02d} | "
            f"LR: {current_lr:.6f} | "
            f"Train Loss: {train_loss:.4f} | "
            f"Test Loss: {test_loss:.4f} | "
            f"Test Acc: {test_acc * 100:.2f}%"
        )

        improved = False

        if test_acc > best_acc:
            improved = True
        elif test_acc == best_acc and test_loss < best_loss:
            improved = True

        if improved:
            best_acc = test_acc
            best_loss = test_loss
            best_epoch = epoch
            no_improve = 0

            torch.save(model.state_dict(), MODEL_SAVE_PATH)
        else:
            no_improve += 1

        if no_improve >= EARLY_STOPPING_PATIENCE:
            print(f"\nEarly stopping at epoch {epoch}")
            break

    print("\n訓練完成")
    print(f"最佳 Epoch: {best_epoch}")
    print(f"最佳測試準確率: {best_acc * 100:.2f}%")
    print(f"模型已儲存: {MODEL_SAVE_PATH}")

    state = torch.load(MODEL_SAVE_PATH, map_location=DEVICE)
    model.load_state_dict(state)

    test_loss, test_acc, cm_test = evaluate(
        model,
        test_loader,
        criterion,
        num_classes,
        DEVICE
    )

    precision, recall = precision_recall_from_cm(cm_test)

    print("\n[Test] Confusion Matrix:")
    print(cm_test)

    print(
        f"\n[Test] Accuracy: {test_acc * 100:.2f}% "
        f"(total={cm_test.sum()}, correct={np.trace(cm_test)}, loss={test_loss:.4f})"
    )

    print("\n[Test] 每類別 Precision / Recall:")
    for i in range(num_classes):
        name = ""
        if label_map and str(i) in label_map:
            name = f" ({label_map[str(i)]})"

        print(
            f"class {i}{name}: "
            f"precision={precision[i] * 100:.2f}% | "
            f"recall={recall[i] * 100:.2f}% | "
            f"total={cm_test[i].sum()}"
        )

    all_loss, all_acc, cm_all = evaluate(
        model,
        all_loader,
        criterion,
        num_classes,
        DEVICE
    )

    print("\n[All] Confusion Matrix:")
    print(cm_all)

    print(
        f"\n[All] Accuracy: {all_acc * 100:.2f}% "
        f"(total={cm_all.sum()}, correct={np.trace(cm_all)}, loss={all_loss:.4f})"
    )

    train_info = {
        "seed": SEED,
        "x_path": X_PATH,
        "y_path": Y_PATH,
        "model_save_path": MODEL_SAVE_PATH,
        "epochs_config": EPOCHS,
        "batch_size": BATCH_SIZE,
        "lr": LR,
        "weight_decay": WEIGHT_DECAY,
        "early_stopping_patience": EARLY_STOPPING_PATIENCE,
        "train_ratio": TRAIN_RATIO,
        "device": DEVICE,
        "x_shape": list(dataset.X.shape),
        "y_shape": list(dataset.y.shape),
        "num_samples": int(len(dataset)),
        "num_classes": int(num_classes),
        "best_epoch": int(best_epoch),
        "best_test_acc": float(best_acc),
        "best_test_loss": float(best_loss),
        "history": history,
    }

    with open(TRAIN_INFO_PATH, "w", encoding="utf-8") as f:
        json.dump(train_info, f, ensure_ascii=False, indent=2)

    print(f"\n訓練資訊已儲存: {TRAIN_INFO_PATH}")


if __name__ == "__main__":
    main()