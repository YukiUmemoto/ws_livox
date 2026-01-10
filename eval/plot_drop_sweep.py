#!/usr/bin/env python3
# -*- coding: utf-8 -*-

import glob
import os
import re
import sys

import pandas as pd
import numpy as np
import matplotlib.pyplot as plt

ROOT = os.environ.get("OUT", "").strip()
if not ROOT:
    raise SystemExit("Set env OUT=<eval_root>  (e.g. export OUT=~/ws_livox/eval/drop_sweep)")

WARMUP_SEC = 5.0  # EMA安定のため最初5秒は捨てる（必要なら調整）

# ------------------------------------------------------------
# find directories: support both "q_0.10" and "snapshot_q_0.10"
# ------------------------------------------------------------
dir_candidates = []
dir_candidates += glob.glob(os.path.join(ROOT, "q_*"))
dir_candidates += glob.glob(os.path.join(ROOT, "snapshot_q_*"))

# 重複排除＆ソート
dir_candidates = sorted(set(dir_candidates))

rows = []
used = 0

for d in dir_candidates:
    base = os.path.basename(os.path.normpath(d))

    # q値の抽出
    # q_0.10 / snapshot_q_0.10 の両方に対応
    m = re.search(r"^(?:snapshot_)?q_([0-9]+(?:\.[0-9]+)?)$", base)
    if not m:
        continue
    q = float(m.group(1))

    # CSV候補（どちらでもOK）
    csv_candidates = [
        os.path.join(d, "frames.csv"),
        os.path.join(d, "frames_snapshot.csv"),
    ]
    csv_path = next((p for p in csv_candidates if os.path.exists(p)), None)
    if csv_path is None:
        print(f"[WARN] missing frames csv under: {d}")
        continue

    df = pd.read_csv(csv_path)
    used += 1

    # 必須列チェック
    required = ["frame_rel", "alert_ratio"]
    missing = [c for c in required if c not in df.columns]
    if missing:
        print(f"[WARN] {csv_path}: missing columns {missing} -> skip")
        continue

    # ---- warmup cut ----
    # stamp_sec + stamp_nanosec があるなら高精度で時刻化
    if "stamp_sec" in df.columns:
        t = df["stamp_sec"].astype(float).to_numpy()
        if "stamp_nanosec" in df.columns:
            t = t + df["stamp_nanosec"].astype(float).to_numpy() * 1e-9
        t0 = float(np.nanmin(t)) if np.isfinite(t).any() else 0.0
        df = df[(t - t0) >= WARMUP_SEC]

    # ---- metrics ----
    fr = df["frame_rel"].astype(float).to_numpy()
    ar = df["alert_ratio"].astype(float).to_numpy()

    # 任意列
    drop_mean = np.nan
    if "drop_ratio" in df.columns:
        drop_mean = float(np.nanmean(df["drop_ratio"].astype(float).to_numpy()))

    rows.append({
        "q": q,
        "frame_rel_mean": float(np.nanmean(fr)) if fr.size else np.nan,
        "frame_rel_std": float(np.nanstd(fr)) if fr.size else np.nan,
        "alert_ratio_mean": float(np.nanmean(ar)) if ar.size else np.nan,
        "alert_ratio_std": float(np.nanstd(ar)) if ar.size else np.nan,
        "n_frames": int(len(df)),
        "drop_ratio_mean": drop_mean,
        "csv_path": csv_path,   # デバッグ用に残す
    })

# ------------------------------------------------------------
# guard: nothing collected
# ------------------------------------------------------------
if not rows:
    msg = [
        "[ERROR] No valid rows were collected.",
        f"ROOT(OUT)={ROOT}",
        f"found dirs={len(dir_candidates)} (matched name pattern but usable rows=0)",
        "Tips:",
        "  - Check directory names: should be q_0.10 or snapshot_q_0.10",
        "  - Check CSV existence: frames.csv or frames_snapshot.csv",
        "  - Check required columns: frame_rel, alert_ratio",
    ]
    raise SystemExit("\n".join(msg))

summary = pd.DataFrame(rows).sort_values("q").reset_index(drop=True)

# 保存
out_summary = os.path.join(ROOT, "summary.csv")
summary.to_csv(out_summary, index=False)
print(summary)
print("saved:", out_summary)

# ------------------------------------------------------------
# plot (left graph): dual axis
# ------------------------------------------------------------
fig = plt.figure()
ax1 = plt.gca()
ax2 = ax1.twinx()

# 1本目（Frame reliability）
e1 = ax1.errorbar(
    summary["q"], summary["frame_rel_mean"],
    yerr=summary["frame_rel_std"],
    marker="o", linestyle="-",
    label="Frame reliability"
)
c1 = e1[0].get_color()  # Line2D の色

# 2本目（Alert ratio） ※こちらは明示的に別色を指定
e2 = ax2.errorbar(
    summary["q"], summary["alert_ratio_mean"],
    yerr=summary["alert_ratio_std"],
    marker="s", linestyle="--",
    color="C1",                 # ←ここで色を分ける（例：tab:orange）
    ecolor="C1",                # 誤差棒の色も揃える
    capsize=3,
    label="Alert ratio"
)
c2 = e2[0].get_color()

ax1.set_xlabel("Packet drop probability q")
ax1.set_ylabel("Frame reliability (mean)", color=c1)
ax2.set_ylabel("AlertRatio (mean)", color=c2)

# 目盛りの色も揃える（見やすさ改善）
ax1.tick_params(axis="y", colors=c1)
ax2.tick_params(axis="y", colors=c2)

ax1.grid(True, which="both")
fig.tight_layout()

out_pdf = os.path.join(ROOT, "left_drop_graph.pdf")
out_png = os.path.join(ROOT, "left_drop_graph.png")
fig.savefig(out_pdf)
fig.savefig(out_png, dpi=200)

print("saved:", out_pdf)
print("saved:", out_png)

