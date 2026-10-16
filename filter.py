import argparse
import json
import os
import re
import sys
import numpy as np
import pandas as pd


LABEL_JSON_DEFAULT = "label.json"


def load_label_map(label_json_path: str | None):
    if not label_json_path:
        return None
    if not os.path.exists(label_json_path):
        return None
    with open(label_json_path, "r", encoding="utf-8") as f:
        return json.load(f)


def label_name_of(lb, label_map):
    if label_map is None:
        return ""
    key = str(int(lb))
    return label_map.get(key, "")


def find_landmark_cols(df: pd.DataFrame):
    """
    自動找 x0..x20, y0..y20, z0..z20
    即使你之後多加 source/frame/d_* 欄也不會壞
    """
    pat = re.compile(r"^([xyz])(\d+)$")
    cols = {"x": [], "y": [], "z": []}

    for c in df.columns:
        m = pat.match(str(c))
        if not m:
            continue
        axis, idx = m.group(1), int(m.group(2))
        cols[axis].append((idx, c))

    for axis in cols:
        cols[axis].sort(key=lambda t: t[0])
        cols[axis] = [c for _, c in cols[axis]]

    if len(cols["x"]) != 21 or len(cols["y"]) != 21 or len(cols["z"]) != 21:
        raise ValueError(
            f"Expected 21 x / 21 y / 21 z columns, "
            f"got x={len(cols['x'])}, y={len(cols['y'])}, z={len(cols['z'])}. "
            f"Check CSV header."
        )

    return cols


def compute_bbox_area(xy: np.ndarray):
    """
    xy: (N, 21, 2)
    return: area(N,), width(N,), height(N,)
    """
    x = xy[:, :, 0]
    y = xy[:, :, 1]

    xmin = np.nanmin(x, axis=1)
    xmax = np.nanmax(x, axis=1)
    ymin = np.nanmin(y, axis=1)
    ymax = np.nanmax(y, axis=1)

    w = xmax - xmin
    h = ymax - ymin
    area = w * h
    return area, w, h


def compute_mean_pairwise_dist(xy: np.ndarray):
    """
    平均點對點距離（2D）
    xy: (N, 21, 2)
    回傳: (N,)
    """
    n = xy.shape[0]
    out = np.zeros(n, dtype=np.float32)

    tri = np.triu_indices(21, k=1)

    for i in range(n):
        pts = xy[i]
        if np.isnan(pts).any():
            out[i] = np.nan
            continue

        diff = pts[:, None, :] - pts[None, :, :]
        dist = np.sqrt((diff ** 2).sum(axis=2))
        out[i] = float(dist[tri].mean())

    return out


def quantiles_dict(arr: np.ndarray):
    q = [0, 0.01, 0.05, 0.1, 0.25, 0.5, 0.75, 0.9, 0.95, 0.99, 1.0]
    arr = arr[~np.isnan(arr)]
    if len(arr) == 0:
        return None
    vals = np.quantile(arr, q)
    return dict(zip(q, np.round(vals, 6)))


def print_report(df, area, mean_dist, label_map=None, label_col="label"):
    print("\n[Overall quantiles]")

    area_q = quantiles_dict(area)
    dist_q = quantiles_dict(mean_dist)

    if area_q is not None:
        print("bbox_area :", area_q)
    if dist_q is not None:
        print("mean_dist :", dist_q)

    if label_col in df.columns:
        print("\n[Per-label summary]")
        labels = sorted(df[label_col].dropna().unique())

        for lb in labels:
            m = (df[label_col] == lb).to_numpy()
            a = area[m]
            d = mean_dist[m]

            a_med = np.nanmedian(a)
            d_med = np.nanmedian(d)
            a_p10 = np.nanpercentile(a, 10)
            d_p10 = np.nanpercentile(d, 10)

            name = label_name_of(lb, label_map)
            suffix = f" ({name})" if name else ""

            print(
                f"label {int(lb):>2}{suffix}: "
                f"n={m.sum():>4} | "
                f"area_med={a_med:.6f} area_p10={a_p10:.6f} | "
                f"dist_med={d_med:.6f} dist_p10={d_p10:.6f}"
            )


def build_per_label_stats(df, valid, area, mean_dist, label_map=None, label_col="label"):
    rows = []
    labels = sorted(df[label_col].dropna().unique())

    for lb in labels:
        m = (df[label_col] == lb).to_numpy()
        total = int(m.sum())
        kept = int((valid & m).sum())
        removed = total - kept

        a = area[m]
        d = mean_dist[m]

        rows.append({
            "label": int(lb),
            "label_name": label_name_of(lb, label_map),
            "total": total,
            "kept": kept,
            "removed": removed,
            "remove_rate_pct": (removed / total * 100.0) if total > 0 else 0.0,
            "bbox_area_med": float(np.nanmedian(a)) if total > 0 else np.nan,
            "bbox_area_p10": float(np.nanpercentile(a, 10)) if total > 0 else np.nan,
            "mean_dist_med": float(np.nanmedian(d)) if total > 0 else np.nan,
            "mean_dist_p10": float(np.nanpercentile(d, 10)) if total > 0 else np.nan,
        })

    return pd.DataFrame(rows)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--csv", default="hand_landmarks.csv", help="input CSV path")
    ap.add_argument("--out", default="hand_landmarks.filtered.csv", help="output CSV path")
    ap.add_argument("--removed_out", default="hand_landmarks.removed.csv", help="removed rows CSV path")
    ap.add_argument("--summary_csv", default="", help="optional per-label summary CSV path")
    ap.add_argument("--label_json", default=LABEL_JSON_DEFAULT, help="label.json path")
    ap.add_argument("--area_min", type=float, default=0.002, help="min bbox area to keep (default 0.002)")
    ap.add_argument("--dist_min", type=float, default=-1.0, help="min mean pairwise distance to keep (<=0 to disable)")
    ap.add_argument("--keep_metrics", action="store_true", help="keep bbox_area / mean_dist / bbox_w / bbox_h columns in output")
    ap.add_argument("--save_removed", action="store_true", help="save removed rows to --removed_out")
    ap.add_argument("--report", action="store_true", help="print quantiles to help tune thresholds")
    args = ap.parse_args()

    if not os.path.exists(args.csv):
        raise FileNotFoundError(f"CSV not found: {args.csv}")

    df = pd.read_csv(args.csv, encoding="utf-8-sig")
    df.columns = (
        df.columns.astype(str)
        .str.replace("\ufeff", "", regex=False)
        .str.replace("\r", "", regex=False)
        .str.strip()
    )

    if "label" not in df.columns:
        raise ValueError("CSV must contain 'label' column")

    label_map = load_label_map(args.label_json)

    cols = find_landmark_cols(df)
    x = df[cols["x"]].to_numpy(dtype=np.float32)
    y = df[cols["y"]].to_numpy(dtype=np.float32)

    xy = np.stack([x, y], axis=2)  # (N,21,2)

    area, w, h = compute_bbox_area(xy)
    mean_dist = compute_mean_pairwise_dist(xy)

    if args.report:
        print_report(df, area, mean_dist, label_map=label_map, label_col="label")

    # 基本有效性條件
    valid = (
        ~np.isnan(area) &
        ~np.isnan(w) &
        ~np.isnan(h) &
        (w > 0) &
        (h > 0) &
        (area > args.area_min)
    )

    # 第二道過濾：平均距離
    if args.dist_min is not None and args.dist_min > 0:
        valid = valid & (~np.isnan(mean_dist)) & (mean_dist > args.dist_min)

    removed = int((~valid).sum())
    kept = int(valid.sum())
    total = int(len(df))

    out_df = df.copy()

    # 若原本沒有 row_id，就幫你加
    if "row_id" not in out_df.columns:
        out_df.insert(0, "row_id", np.arange(total, dtype=int))

    if args.keep_metrics:
        out_df["bbox_area"] = area
        out_df["bbox_w"] = w
        out_df["bbox_h"] = h
        out_df["mean_dist"] = mean_dist

    kept_df = out_df[valid].reset_index(drop=True)
    kept_df.to_csv(args.out, index=False, encoding="utf-8-sig")

    if args.save_removed:
        removed_df = out_df[~valid].reset_index(drop=True)
        removed_df.to_csv(args.removed_out, index=False, encoding="utf-8-sig")

    print(f"\nDone. total={total}, kept={kept}, removed={removed}")
    print(f"Saved kept CSV: {args.out}")
    if args.save_removed:
        print(f"Saved removed CSV: {args.removed_out}")

    stat = build_per_label_stats(
        df=df,
        valid=valid,
        area=area,
        mean_dist=mean_dist,
        label_map=label_map,
        label_col="label"
    )

    print("\nRemoved/Kept by label:")
    display_cols = ["label", "label_name", "total", "kept", "removed", "remove_rate_pct"]
    print(stat[display_cols].to_string(index=False))

    if args.summary_csv:
        stat.to_csv(args.summary_csv, index=False, encoding="utf-8-sig")
        print(f"\nSaved summary CSV: {args.summary_csv}")


if __name__ == "__main__":
    try:
        main()
    except Exception as e:
        print("ERROR:", e)
        sys.exit(1)