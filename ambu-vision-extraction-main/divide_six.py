#!/usr/bin/env python3
"""
Split rows into files based on exact (case-sensitive) combinations of:
  Psychological_variable ∈ {"positive affect", "negative affect", "stress"}
  direction              ∈ {"increase", "decrease"}

Anything that doesn't match these exact strings goes to 'artifacts/visual_factors/mixed.csv'.

Input:
  artifacts/visual_factors/removed_none.csv

Output (all under artifacts/visual_factors/), e.g.:
  positive_affect_increase.csv
  positive_affect_decrease.csv
  negative_affect_increase.csv
  negative_affect_decrease.csv
  stress_increase.csv
  stress_decrease.csv
  mixed.csv
"""

import os
import pandas as pd
from pathlib import Path

# ---- Config ----
INPUT_FILE = Path("artifacts/visual_factors/removed_none.csv")  
OUT_DIR = Path("artifacts/visual_factors")
OUT_DIR.mkdir(parents=True, exist_ok=True)

# ---- Checks ----
if not INPUT_FILE.exists():
    raise FileNotFoundError(f"Input file not found: {INPUT_FILE}")

df = pd.read_csv(INPUT_FILE)

required_cols = {"Psychological_variable", "direction"}
missing = required_cols - set(df.columns)
if missing:
    raise KeyError(f"Missing required columns: {sorted(missing)}")

# ---- Exact-match (case-sensitive) definitions ----
outcomes = [
    "positive affect",
    "negative affect",
    "stress",
]
directions = ["increase", "decrease"]

# Map outcome/direction to safe filename token
def safe_token(s: str) -> str:
    return (
        s.strip()
         .lower()
         .replace(" ", "_")
         .replace("-", "_")
    )

# Track which rows have been assigned to one of the 6 buckets
assigned_mask = pd.Series(False, index=df.index)

# ---- Create 6 files (3 outcomes * 2 directions) ----
for outcome in outcomes:
    for direction in directions:
        mask = (df["Psychological_variable"] == outcome) & (df["direction"] == direction)
        bucket = df[mask]
        assigned_mask |= mask

        out_name = OUT_DIR / f"{safe_token(outcome)}_{safe_token(direction)}.csv"
        # Ensure file is created even if empty, preserving headers
        bucket.to_csv(out_name, index=False)
        print(f"Saved {len(bucket):>6} rows to {out_name}")

# ---- Mixed (everything else) ----
mixed = df[~assigned_mask]
mixed_path = OUT_DIR / "mixed.csv"
mixed.to_csv(mixed_path, index=False)
print(f"Saved {len(mixed):>6} rows to {mixed_path}")
