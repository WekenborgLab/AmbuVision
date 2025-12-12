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

# ---- Output folder ----
out_dir = Path("artifacts/visual_factors/output/lists")
out_dir.mkdir(parents=True, exist_ok=True)

for file in files:
    path = Path(file)
    if not path.exists():
        print(f"File not found: {path}")
        continue

    df = pd.read_csv(path)

    required = {"Group Name", "Psychological_variable", "direction"}
    missing = required - set(df.columns)
    if missing:
        print(f"Missing columns in {path.name}: {', '.join(sorted(missing))}")
        continue

    df["Group Name"] = df["Group Name"].astype(str).str.strip()
    df["Psychological_variable"] = df["Psychological_variable"].astype(str).str.strip()
    df["direction"] = df["direction"].astype(str).str.strip()

    # Drop rows with empty/NaN Group Name since we want unique group names
    df = df[df["Group Name"].notna() & (df["Group Name"] != "")]

    if df.empty:
        print(f"No rows after filtering in {path.name}")
        # Still write an empty summary for completeness
        out_path = out_dir / f"{path.stem}_group_psych_summary.csv"
        pd.DataFrame(columns=[
            "Group Name", "Psychological_variable", "direction", "Frequency of Groups"
        ]).to_csv(out_path, index=False)
        print(f"Wrote empty summary: {out_path}")
        continue

    # Group and count occurrences
    summary = (
        df.groupby(["Group Name", "Psychological_variable", "direction"], dropna=False)
          .size()
          .reset_index(name="Frequency of Groups")
    )

    # Sort for readability
    summary = summary.sort_values(
        by=["Group Name", "Psychological_variable", "direction"],
        kind="stable"
    )

    # Save per-file summary
    out_path = out_dir / f"{path.stem}_group_psych_summary.csv"
    summary.to_csv(out_path, index=False)
    print(f"Wrote summary for {path.name} → {out_path}")
