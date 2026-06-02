"""
Compare two plant_presence.csv runs side by side.

Usage:
    uv run python compare.py <yolo_csv> <yoloe_csv> [--out compare.png]

Prints empty-row counts, summary stats, mean processing time. Saves a
histogram of PlantPresence and a scatter plot of YOLO vs YOLOE scores.
"""

from __future__ import annotations

import argparse
import sys
from pathlib import Path

import numpy as np
import pandas as pd
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt


def load(p: Path, label: str) -> pd.DataFrame:
    df = pd.read_csv(p)
    df["__label__"] = label
    return df


def summarize(df: pd.DataFrame, label: str) -> None:
    empty_score = (df.PlantPresence == 0).sum()
    empty_cov   = (df.PlantCoverageFraction == 0).sum()
    empty_det   = (df.NumPlantDetections == 0).sum()
    print(f"== {label} ==")
    print(f"  rows: {len(df)}")
    print(f"  empty (PlantPresence==0):         {empty_score}  ({empty_score/len(df)*100:.1f}%)")
    print(f"  empty (PlantCoverageFraction==0): {empty_cov}  ({empty_cov/len(df)*100:.1f}%)")
    print(f"  empty (NumPlantDetections==0):    {empty_det}  ({empty_det/len(df)*100:.1f}%)")
    print(f"  PlantPresence  mean={df.PlantPresence.mean():.3f}  "
          f"median={df.PlantPresence.median():.3f}  max={df.PlantPresence.max():.2f}")
    if "ProcessSeconds" in df.columns:
        ps = df.ProcessSeconds
        print(f"  ProcessSeconds mean={ps.mean():.3f}  median={ps.median():.3f}  "
              f"p95={ps.quantile(0.95):.3f}  total={ps.sum():.1f}s")
    print()


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("yolo_csv")
    ap.add_argument("yoloe_csv")
    ap.add_argument("--out", default="compare.png")
    args = ap.parse_args()

    a = load(Path(args.yolo_csv),  "YOLO (oiv7, Plant)")
    b = load(Path(args.yoloe_csv), "YOLOE (prompt: plant)")

    summarize(a, "YOLO (oiv7, Plant)")
    summarize(b, "YOLOE (prompt: plant)")

    # ---- histogram of PlantPresence ----
    fig, axes = plt.subplots(1, 2, figsize=(12, 4.5))

    bins = np.linspace(0, 10, 21)
    axes[0].hist(a.PlantPresence, bins=bins, alpha=0.6, label="YOLO (oiv7, Plant)", color="#1f77b4")
    axes[0].hist(b.PlantPresence, bins=bins, alpha=0.6, label="YOLOE (prompt: plant)", color="#d62728")
    axes[0].set_xlabel("PlantPresence (0–10)")
    axes[0].set_ylabel("# images")
    axes[0].set_title("PlantPresence histogram")
    axes[0].legend()
    axes[0].set_yscale("log")  # mostly-zero distribution; log helps see the tail

    # ---- scatter YOLO vs YOLOE per image ----
    merged = a.merge(b, on=["PictureId", "Folder"], suffixes=("_yolo", "_yoloe"))
    if len(merged):
        axes[1].scatter(merged.PlantPresence_yolo, merged.PlantPresence_yoloe,
                        s=8, alpha=0.4)
        axes[1].plot([0, 10], [0, 10], "k--", lw=0.8)
        axes[1].set_xlabel("YOLO PlantPresence")
        axes[1].set_ylabel("YOLOE PlantPresence")
        axes[1].set_title(f"Per-image agreement (n={len(merged)})")
        axes[1].set_xlim(-0.2, 10.2); axes[1].set_ylim(-0.2, 10.2)
        r = merged[["PlantPresence_yolo", "PlantPresence_yoloe"]].corr().iloc[0, 1]
        axes[1].text(0.5, 9.2, f"Pearson r = {r:.3f}", fontsize=10)

    plt.tight_layout()
    plt.savefig(args.out, dpi=120)
    print(f"Wrote {args.out}")

    # Per-cell agreement table
    if len(merged):
        a_empty = merged.PlantPresence_yolo == 0
        b_empty = merged.PlantPresence_yoloe == 0
        print("Agreement on emptiness:")
        print(f"  both empty:           {(a_empty & b_empty).sum()}")
        print(f"  only YOLO empty:      {(a_empty & ~b_empty).sum()}")
        print(f"  only YOLOE empty:     {(~a_empty & b_empty).sum()}")
        print(f"  neither empty:        {(~a_empty & ~b_empty).sum()}")


if __name__ == "__main__":
    sys.exit(main())
