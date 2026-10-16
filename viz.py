# viz.py
import os
import json
import argparse
import random

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt


def get_hand_connections():
    """
    優先用 mediapipe 的 HAND_CONNECTIONS；沒有就用內建的連線表。
    """
    try:
        import mediapipe as mp
        return list(mp.solutions.hands.HAND_CONNECTIONS)
    except Exception:
        return [
            (0, 1), (1, 2), (2, 3), (3, 4),         # thumb
            (0, 5), (5, 6), (6, 7), (7, 8),         # index
            (0, 9), (9, 10), (10, 11), (11, 12),    # middle
            (0, 13), (13, 14), (14, 15), (15, 16),  # ring
            (0, 17), (17, 18), (18, 19), (19, 20),  # pinky
            (5, 9), (9, 13), (13, 17),              # palm
        ]


def clean_columns(df: pd.DataFrame) -> pd.DataFrame:
    df = df.copy()
    df.columns = (
        df.columns.astype(str)
        .str.replace("\ufeff", "", regex=False)
        .str.replace("\r", "", regex=False)
        .str.strip()
    )
    return df


def load_label_map(path: str):
    if not path or not os.path.exists(path):
        return None
    with open(path, "r", encoding="utf-8") as f:
        return json.load(f)


def row_to_xyz(row):
    pts = []
    for i in range(21):
        pts.append([row[f"x{i}"], row[f"y{i}"], row[f"z{i}"]])
    return np.array(pts, dtype=np.float32)


def plot_one(
    ax,
    pts,
    title="",
    annotate=True,
    connections=None,
    invert_y=True,
):
    x = pts[:, 0]
    y = pts[:, 1]
    _z = pts[:, 2]

    if invert_y:
        # 讓視覺上更像圖片座標（上方是小 y）
        y = 1.0 - y

    ax.scatter(x, y)

    if connections:
        for a, b in connections:
            ax.plot([x[a], x[b]], [y[a], y[b]])

    if annotate:
        for i in range(21):
            ax.text(x[i], y[i], str(i), fontsize=8)

    ax.set_title(title, fontsize=10)
    ax.set_aspect("equal", adjustable="box")
    ax.set_xlim(-0.1, 1.1)
    ax.set_ylim(-0.1, 1.1)
    ax.grid(True)


def build_title(row, ridx, label_map=None, show_meta=False):
    label = int(row["label"]) if "label" in row else -1
    label_name = ""
    if label_map is not None and str(label) in label_map:
        label_name = f" ({label_map[str(label)]})"

    parts = [f"row={ridx}", f"label={label}{label_name}"]

    if show_meta:
        if "source_path" in row and pd.notna(row["source_path"]):
            parts.append(f"src={row['source_path']}")
        if "frame_idx" in row and pd.notna(row["frame_idx"]):
            parts.append(f"frame={row['frame_idx']}")

    return " | ".join(parts)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--csv", default="hand_landmarks.csv", help="CSV path")
    ap.add_argument("--meta_csv", default="", help="optional meta csv path")
    ap.add_argument("--label_json", default="label.json", help="optional label.json path")

    ap.add_argument("--label", type=int, default=None, help="which class to visualize, e.g. 8")
    ap.add_argument("--rows", default="", help='specific rows, e.g. "12,35,88"')

    ap.add_argument("--n", type=int, default=20, help="how many samples to draw")
    ap.add_argument("--seed", type=int, default=0, help="random seed")

    ap.add_argument("--outdir", default="", help="if set, save images to this folder")
    ap.add_argument("--show", action="store_true", help="show plots on screen")
    ap.add_argument("--cols", type=int, default=5, help="grid columns when showing")

    ap.add_argument("--annotate", action="store_true", help="show landmark index text")
    ap.add_argument("--show_meta", action="store_true", help="show source_path/frame_idx in title if meta csv exists")
    ap.add_argument("--invert_y", action="store_true", help="invert y axis to look like image coordinates")
    args = ap.parse_args()

    if not os.path.exists(args.csv):
        raise FileNotFoundError(f"CSV not found: {args.csv}")

    df = pd.read_csv(args.csv, encoding="utf-8-sig")
    df = clean_columns(df)

    if "label" not in df.columns:
        raise ValueError("CSV must contain 'label' column")

    for i in range(21):
        for axis in ("x", "y", "z"):
            c = f"{axis}{i}"
            if c not in df.columns:
                raise ValueError(f"CSV 缺少欄位: {c}")

    label_map = load_label_map(args.label_json)
    meta_df = None

    if args.meta_csv:
        if not os.path.exists(args.meta_csv):
            raise FileNotFoundError(f"meta CSV not found: {args.meta_csv}")
        meta_df = pd.read_csv(args.meta_csv, encoding="utf-8-sig")
        meta_df = clean_columns(meta_df)

        if "row_id" not in meta_df.columns:
            raise ValueError("meta CSV 缺少 row_id 欄位")

        # merge 時保留原本訓練 CSV 的 row index 當 row_id
        if "row_id" not in df.columns:
            df = df.copy()
            df.insert(0, "row_id", np.arange(len(df), dtype=int))

        keep_cols = [c for c in ["row_id", "source_path", "frame_idx", "bbox_area", "mean_dist"] if c in meta_df.columns]
        meta_df = meta_df[keep_cols].copy()
        df = df.merge(meta_df, on="row_id", how="left")

    # ===== 選樣本 =====
    selected_indices = []

    if args.rows.strip():
        row_ids = [int(x.strip()) for x in args.rows.split(",") if x.strip()]
        if "row_id" in df.columns:
            sub = df[df["row_id"].isin(row_ids)].copy()
            if len(sub) == 0:
                raise ValueError(f"找不到指定 rows: {row_ids}")
            selected_indices = sub.index.tolist()
        else:
            valid_rows = [r for r in row_ids if 0 <= r < len(df)]
            if len(valid_rows) == 0:
                raise ValueError(f"找不到指定 rows: {row_ids}")
            selected_indices = valid_rows

    elif args.label is not None:
        sub = df[df["label"] == args.label].copy()
        if len(sub) == 0:
            raise ValueError(f"No samples for label={args.label}")

        random.seed(args.seed)
        idxs = list(sub.index)
        random.shuffle(idxs)
        selected_indices = idxs[: min(args.n, len(idxs))]

    else:
        raise ValueError("請至少提供 --label 或 --rows")

    connections = get_hand_connections()

    # ===== 若指定 outdir：每筆存一張 =====
    if args.outdir:
        os.makedirs(args.outdir, exist_ok=True)

        for k, ridx in enumerate(selected_indices):
            row = df.loc[ridx]
            pts = row_to_xyz(row)

            title = build_title(
                row=row,
                ridx=int(row["row_id"]) if "row_id" in row else int(ridx),
                label_map=label_map,
                show_meta=args.show_meta,
            )

            fig, ax = plt.subplots(figsize=(4, 4))
            plot_one(
                ax=ax,
                pts=pts,
                title=title,
                annotate=args.annotate,
                connections=connections,
                invert_y=args.invert_y,
            )

            rid_for_name = int(row["row_id"]) if "row_id" in row else int(ridx)
            label_for_name = int(row["label"]) if "label" in row else -1
            outpath = os.path.join(args.outdir, f"label{label_for_name}_row{rid_for_name}.png")

            fig.tight_layout()
            fig.savefig(outpath, dpi=150)
            plt.close(fig)

        print(f"Saved {len(selected_indices)} images to: {args.outdir}")

    # ===== 若要 show：用網格一次看多筆 =====
    if args.show:
        cols = max(1, args.cols)
        rows = int(np.ceil(len(selected_indices) / cols))

        fig, axes = plt.subplots(rows, cols, figsize=(4 * cols, 4 * rows))
        axes = np.array(axes).reshape(-1)

        for i, ax in enumerate(axes):
            if i >= len(selected_indices):
                ax.axis("off")
                continue

            ridx = selected_indices[i]
            row = df.loc[ridx]
            pts = row_to_xyz(row)

            title = build_title(
                row=row,
                ridx=int(row["row_id"]) if "row_id" in row else int(ridx),
                label_map=label_map,
                show_meta=args.show_meta,
            )

            plot_one(
                ax=ax,
                pts=pts,
                title=title,
                annotate=args.annotate,
                connections=connections,
                invert_y=args.invert_y,
            )

        if args.label is not None:
            label_title = str(args.label)
            if label_map is not None and str(args.label) in label_map:
                label_title += f" ({label_map[str(args.label)]})"
            fig.suptitle(f"Landmarks samples | label={label_title} | n={len(selected_indices)}", fontsize=14)
        else:
            fig.suptitle(f"Landmarks samples | rows={len(selected_indices)}", fontsize=14)

        fig.tight_layout()
        plt.show()


if __name__ == "__main__":
    main()