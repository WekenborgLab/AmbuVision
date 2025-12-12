#!/usr/bin/env python3
import os
import re
import asyncio
import pandas as pd
from openai import AsyncOpenAI

# ---------------- Config ----------------
MODEL_NAME = os.environ.get("MODEL_NAME", "GPT-OSS-120B")
BASE_URL   = os.environ.get("OPENAI_BASE_URL")
API_KEY    = os.environ.get("VIRTUAL_API_KEY")
if not API_KEY:
    raise ValueError("Missing VIRTUAL_API_KEY environment variable.")

CATEGORY_COL   = "Category"
SEED_SIZE      = 200
MAX_CONCURRENT = int(os.environ.get("MAX_CONCURRENT", "200"))

# Input files
INPUT_FILES = [
    "artifacts/visual_factors/positive_affect_increase.csv",
    "artifacts/visual_factors/positive_affect_decrease.csv",
    "artifacts/visual_factors/negative_affect_increase.csv",
    "artifacts/visual_factors/negative_affect_decrease.csv",
    "artifacts/visual_factors/stress_increase.csv",
    "artifacts/visual_factors/stress_decrease.csv",
    "mixed.csv",
]

# Output folder
OUTPUT_DIR = "artifacts/visual_factors/output"
os.makedirs(OUTPUT_DIR, exist_ok=True)

# ---------------- Utilities ----------------
def parse_groups_text(groups_text: str):
    groups = []
    pat = re.compile(r"^Group\s+(\d+)\s+(.+?):\s*(.+?)\s*$")
    for raw_line in groups_text.splitlines():
        line = raw_line.strip()
        if not line:
            continue
        m = pat.match(line)
        if not m:
            continue
        gnum = int(m.group(1))
        gname = m.group(2).strip()
        labels = [lab.strip() for lab in m.group(3).split(",") if lab.strip()]
        groups.append({"group": gnum, "name": gname, "labels": labels})
    groups = sorted(groups, key=lambda x: x["group"])
    for i, g in enumerate(groups, start=1):
        g["group"] = i
    return groups

def groups_summary_for_prompt(groups, max_examples_per_group=6):
    lines = []
    for g in groups:
        ex = g["labels"][:max_examples_per_group]
        example_str = ", ".join(ex) if ex else "(none yet)"
        lines.append(f"Group {g['group']} {g['name']}: {example_str}")
    return "\n".join(lines)

def pick_and_pop_row(df: pd.DataFrame, category_col: str, label: str):
    matches = df.index[df[category_col] == label]
    if len(matches) == 0:
        matches = df.index[df[category_col].astype(str).str.strip().str.lower() == str(label).strip().lower()]
    if len(matches) == 0:
        return None, df
    idx = matches[0]
    row = df.loc[idx].copy()
    df_new = df.drop(index=idx)
    return row, df_new

# ---------------- Async LLM calls ----------------
async def chat_call(client: AsyncOpenAI, system: str, user: str, max_tokens=4096) -> str:
    print("Calling local model...")
    resp = await client.chat.completions.create(
        model=MODEL_NAME,
        messages=[{"role": "system", "content": system},
                  {"role": "user",   "content": user}],
        temperature=0.1,
        max_completion_tokens=max_tokens,
        reasoning_effort="medium",
        extra_body={"allowed_openai_params": ["reasoning_effort"]},
    )
    print("Response received from model.")
    return (resp.choices[0].message.content or "").strip()

async def seed_groups_async(client: AsyncOpenAI, labels):
    comma_list = ", ".join([str(c).strip() for c in labels])
    system = (
        "You are a precise taxonomy normalizer. "
        "Group near-duplicate or synonymous labels together based on meaning. "
        "Number groups sequentially (Group 1, Group 2, ...). "
        "If a label has no synonyms, put it alone in its own group. "
        "Do not group opposite categories together. "
        "Return the groups between the exact markers, one group per line. "
        "Use a concise CategoryName for each group."
    )
    user = f"""Cluster these labels by meaning. Use the format:

<<GROUPS-BEGIN>>
Group 1 CategoryName: labelA, labelB
Group 2 CategoryName: labelC
...
<<GROUPS-END>>

Rules:
- Use the original labels verbatim.
- If the same label appears multiple times in the input, include it only once in your output.
- Keep first-seen order as much as possible.
- Do NOT add commentary or tables.
- Every distinct label must appear exactly once.

Labels (comma-separated, may include duplicates):
{comma_list}
"""
    print(f"Creating seed groups with {len(labels)} labels (no de-dup).")
    content = await chat_call(client, system, user, max_tokens=12000)
    start, end = content.find("<<GROUPS-BEGIN>>"), content.find("<<GROUPS-END>>")
    extracted = content[start + len("<<GROUPS-BEGIN>>"):end].strip() if (start != -1 and end != -1 and end > start) else content.strip()
    print("🔎 Seed groups preview:")
    print("\n".join(extracted.splitlines()[:10]))
    return extracted

async def assign_one_label(client: AsyncOpenAI, sem: asyncio.Semaphore, label: str, groups_summary: str) -> tuple[str, str]:
    system = (
        "You are a precise taxonomy normalizer. "
        "You will assign ONE label to the best-fitting existing group, or propose a NEW group if none fits."
    )
    user = f"""We already have these groups:

{groups_summary}

Now assign this label to ONE existing group by number, or propose a new group.

Label: {label}

Return ONLY one line between the markers, no explanations:
<<ASSIGN-BEGIN>>
Group N
<<ASSIGN-END>>
OR
<<ASSIGN-BEGIN>>
NEW: GroupName
<<ASSIGN-END>>"""
    async with sem:
        print(f"Assigning label: {label}")
        content = await chat_call(client, system, user, max_tokens=256)
    start, end = content.find("<<ASSIGN-BEGIN>>"), content.find("<<ASSIGN-END>>")
    extracted = content[start + len("<<ASSIGN-BEGIN>>"):end].strip() if (start != -1 and end != -1 and end > start) else content.strip()
    decision = (extracted.splitlines()[0].strip() if extracted else "")
    print(f"Decision for '{label}': {decision}")
    return label, decision

# ---------------- Per-file Processing ----------------
async def process_file(input_csv: str, output_csv: str):
    client = AsyncOpenAI(base_url=BASE_URL, api_key=API_KEY)

    df_all = pd.read_csv(input_csv)
    print(f"\nLoaded {len(df_all)} rows from {input_csv}")
    if CATEGORY_COL not in df_all.columns:
        raise KeyError(f"'{CATEGORY_COL}' column not found in {input_csv}. Columns: {list(df_all.columns)}")

    total_limit = len(df_all)
    df_work = df_all.head(total_limit).copy()
    all_labels = df_work[CATEGORY_COL].astype(str).tolist()

    seed_labels = all_labels[:SEED_SIZE]
    remaining   = all_labels[SEED_SIZE:total_limit]
    print(f"Seeding with {len(seed_labels)} labels; then placing {len(remaining)} more (total {total_limit}).")

    seed_text = await seed_groups_async(client, seed_labels)
    groups = parse_groups_text(seed_text)
    if not groups:
        print("No seed groups parsed; falling back to 1 label per group from seed.")
        for i, lab in enumerate(seed_labels, start=1):
            groups.append({"group": i, "name": lab, "labels": [lab]})
    print(f"Parsed {len(groups)} seed groups.")

    frozen_summary = groups_summary_for_prompt(groups, max_examples_per_group=6)
    sem = asyncio.Semaphore(MAX_CONCURRENT)
    tasks = [assign_one_label(client, sem, lab, frozen_summary) for lab in remaining]

    if tasks:
        results = await asyncio.gather(*tasks, return_exceptions=False)
        for lab, decision in results:
            if decision.lower().startswith("group"):
                m = re.search(r"\d+", decision)
                if m:
                    num = int(m.group(0))
                    if 1 <= num <= len(groups):
                        groups[num-1]["labels"].append(lab)
                        continue
            if decision.lower().startswith("new:"):
                gname = decision.split(":", 1)[1].strip() or lab
            else:
                gname = lab
            new_idx = len(groups) + 1
            groups.append({"group": new_idx, "name": gname, "labels": [lab]})

    picked_rows = []
    working_df = df_work.copy()
    for g in groups:
        gnum, gname = g["group"], g["name"]
        for lab in g["labels"]:
            row, working_df = pick_and_pop_row(working_df, CATEGORY_COL, lab)
            if row is None:
                row = pd.Series(index=df_work.columns, dtype="object")
                row[CATEGORY_COL] = lab
            row_out = row.to_dict()
            row_out["Group"] = gnum
            row_out["Group Name"] = gname
            picked_rows.append(row_out)

    out_df = pd.DataFrame(picked_rows)
    if CATEGORY_COL in out_df.columns:
        cols = list(out_df.columns)
        ci = cols.index(CATEGORY_COL)
        new_order = cols[:ci+1] + ["Group", "Group Name"] + [c for c in cols if c not in [CATEGORY_COL, "Group", "Group Name"]]
        out_df = out_df[new_order]

    os.makedirs(os.path.dirname(output_csv), exist_ok=True)
    out_df.to_csv(output_csv, index=False)
    print(f"Saved {len(out_df):6d} rows to {output_csv}")

# ---------------- Entrypoint ----------------
async def main():
    for path in INPUT_FILES:
        base_name = os.path.basename(path)
        output_path = os.path.join(OUTPUT_DIR, base_name)
        try:
            await process_file(path, output_path)
        except Exception as e:
            print(f"Error processing {path}: {e}")

if __name__ == "__main__":
    asyncio.run(main())
