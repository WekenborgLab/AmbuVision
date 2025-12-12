#!/usr/bin/env python3
import pandas as pd
from pathlib import Path

# ---- Input files ----
files = [
    "artifacts/visual_factors/output/positive_affect_increase.csv",
    "artifacts/visual_factors/output/positive_affect_decrease.csv",
    "artifacts/visual_factors/output/negative_affect_increase.csv",
    "artifacts/visual_factors/output/negative_affect_decrease.csv",
    "artifacts/visual_factors/output/stress_increase.csv",
    "artifacts/visual_factors/output/stress_decrease.csv",
]

# ---- Output file ----
output_txt = Path("artifacts/visual_factors/output/lists/unique_group_names.txt")

unique_groups = set()

# ---- Collect unique values ----
for file in files:
    path = Path(file)
    if not path.exists():
        print(f"File not found: {path}")
        continue

    df = pd.read_csv(path)

    required_cols = {"Group Name", "Exposure_type"}
    missing = required_cols - set(df.columns)
    if missing:
        print(f"Missing columns in {path.name}: {', '.join(sorted(missing))}")
        continue

    # Filter to only PASSIVE exposure type
    df = df[df["Exposure_type"].astype(str).str.strip().str.upper() == "PASSIVE"]

    # Drop NaN and empty strings, strip whitespace
    groups = df["Group Name"].dropna().astype(str).str.strip()
    groups = groups[groups != ""]
    unique_groups.update(groups.unique())

# ---- Save to text file ----
output_txt.parent.mkdir(parents=True, exist_ok=True)
with open(output_txt, "w", encoding="utf-8") as f:
    for name in sorted(unique_groups):
        f.write(name + "\n")

print(f"Saved {len(unique_groups)} unique PASSIVE group names to {output_txt}")
