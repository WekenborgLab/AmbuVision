"""
3-way compare across:
  A: YOLO+oiv7 multi-class  (Plant, Tree, Flower, Houseplant, Flowerpot + species)
  B: YOLO+oiv7 single-class (Plant only)
  C: YOLOE prompt 'plant'   (segmentation mask)

Usage: uv run python compare3.py <A.csv> <B.csv> <C.csv> --out compare3.png
"""

from __future__ import annotations

import argparse
from pathlib import Path

import numpy as np
import pandas as pd
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt


def load(p, label):
    df = pd.read_csv(p)
    df["__label__"] = label
    return df


def summarize(df, label):
    n = len(df)
    empty = (df.PlantPresence == 0).sum()
    nz = df[df.PlantPresence > 0]
    print(f"== {label} ==")
    print(f"  rows {n}  empty {empty} ({empty/n*100:.1f}%)  non-zero {len(nz)}")
    print(f"  PlantPresence  mean={df.PlantPresence.mean():.3f}  median={df.PlantPresence.median():.3f}  "
          f"p90={df.PlantPresence.quantile(0.90):.2f}  p95={df.PlantPresence.quantile(0.95):.2f}  max={df.PlantPresence.max():.2f}")
    print(f"  non-zero mean={nz.PlantPresence.mean():.3f}  median={nz.PlantPresence.median():.3f}")
    print()


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("a"); ap.add_argument("b"); ap.add_argument("c")
    ap.add_argument("--labels", nargs=3,
                    default=["YOLO multi-class", "YOLO single-class (Plant)", "YOLOE (prompt: plant)"])
    ap.add_argument("--out", default="compare3.png")
    args = ap.parse_args()

    a = load(args.a, args.labels[0])
    b = load(args.b, args.labels[1])
    c = load(args.c, args.labels[2])

    for df, lbl in [(a,args.labels[0]), (b,args.labels[1]), (c,args.labels[2])]:
        summarize(df, lbl)

    fig, axes = plt.subplots(1, 2, figsize=(13, 5))
    bins = np.linspace(0, 10, 21)
    colors = ["#1f77b4", "#2ca02c", "#d62728"]
    for df, lbl, col in zip([a, b, c], args.labels, colors):
        axes[0].hist(df.PlantPresence, bins=bins, alpha=0.5, label=lbl, color=col)
    axes[0].set_xlabel("PlantPresence (0–10)")
    axes[0].set_ylabel("# images")
    axes[0].set_title("PlantPresence histogram (log y)")
    axes[0].legend()
    axes[0].set_yscale("log")

    # Stacked bar: empty / low (0,2] / med (2,5] / high (5,10]
    cats = [("=0", lambda s: s==0),
            ("(0,2]", lambda s: (s>0)&(s<=2)),
            ("(2,5]", lambda s: (s>2)&(s<=5)),
            ("(5,10]", lambda s: s>5)]
    x = np.arange(3)
    bottoms = np.zeros(3)
    for cat_lbl, cat_fn in cats:
        vals = [cat_fn(d.PlantPresence).sum() for d in [a,b,c]]
        axes[1].bar(x, vals, bottom=bottoms, label=cat_lbl)
        for xi, v, bot in zip(x, vals, bottoms):
            if v > 0:
                axes[1].text(xi, bot + v/2, str(v), ha="center", va="center", fontsize=9, color="white")
        bottoms += np.array(vals)
    axes[1].set_xticks(x); axes[1].set_xticklabels(args.labels, rotation=10, ha="right", fontsize=9)
    axes[1].set_ylabel("# images")
    axes[1].set_title("Score-band counts")
    axes[1].legend(title="PlantPresence")

    plt.tight_layout()
    plt.savefig(args.out, dpi=120)
    print(f"Wrote {args.out}")


if __name__ == "__main__":
    main()
