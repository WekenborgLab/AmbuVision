# AmbuVision — Categorization Pipeline

A reproducible pipeline to **retrieve papers**, **extract visually detectable factors linked to emotions**, **clean & categorize factors**, and **produce deduplicated group lists** for downstream VLM scoring.

---

## Table of contents
- [Overview](#overview)
- [Directory layout](#directory-layout)
- [Prerequisites](#prerequisites)
- [Quick start](#quick-start)
- [Step-by-step](#step-by-step)
  - [Step 1: Retrieve files from EPMC](#step-1-retrieve-files-from-epmc)
  - [Step 2: LLM extraction of factor-outcome associations](#step-2-llm-extraction-of-factor-outcome-associations)
  - [Step 3: Clean and filter factors](#step-3-clean-and-filter-factors)
  - [Step 4: Split by psychological outcome and direction](#step-4-split-by-psychological-outcome-and-direction)
  - [Step 5: Deduplicate categories with LLM grouping](#step-5-deduplicate-categories-with-llm-grouping)
  - [Step 6: Build group lists](#step-6-build-group-lists)
- [Outputs](#outputs)

---

## Overview

**Goal:** identify **visually observable contextual factors** reported in psychology papers that are associated with **affect, emotion, stress, safety, and well-being**, then normalize and deduplicate these factors into clean lists.

**High-level pipeline**
1) Retrieve full texts/records from Europe PMC  
2) Use an LLM to extract factor → outcome associations  
3) Filter out invalid/`None` categories  
4) Split into six bins by outcome × direction  
5) LLM-based deduplication & grouping  
6) Export per-group CSVs and a master unique list

---

## Directory layout

```
AmbuVision/
├─ artifacts/
│  ├─ epmc_fulltext/                 # raw EPMC dumps 
│  └─ visual_factors/
│     ├─ all_visual_associations.csv # output of Step 2
│     ├─ output/
│     │  ├─ lists/                   # final grouped lists (Step 6)
│     │  └─ ...                      # intermediate CSVs
├─ retrieve.py                       # Step 1
├─ llm_extraction.py                 # Step 2
├─ factor_categorization.py          # Step 3 (filtered factors count)
├─ remove_none.py                    # Step 3 (drop None categories)
├─ divide_six.py                     # Step 4 (split by outcome×direction)
├─ deduplicator.py                   # Step 5 (LLM grouping)
├─ group_lists.py                    # Step 6 (per-group CSVs)
└─ unique_group_lists.py             # Step 6 (master unique list)
```

---

## Prerequisites

- Python 3.10+  
- A local or remote LLM endpoint (OpenAI-compatible)  
- Disk space for ~65k EPMC records

Recommended:
```bash
python -m venv .venv
source .venv/bin/activate
pip install -r requirements.txt
```

---

## Quick start

```bash
# 1) Retrieve papers from EPMC
python retrieve.py

# 2) Extract factor→outcome associations with LLM
python llm_extraction.py

# 3) Clean and filter categories
python factor_categorization.py
python remove_none.py

# 4) Split into six datasets (outcome × direction)
python divide_six.py

# 5) Deduplicate/group categories with LLM
python deduplicator.py

# 6) Build group lists and a unique master list
python group_lists.py
python unique_group_lists.py
```

---

## Step-by-step

### Step 1: Retrieve files from EPMC
**Script:** `retrieve.py`  
**Input:** EPMC API  
**Output dir:** `artifacts/epmc_fulltext/`  
**Expected scale:** **Total files from EPMC = 64,597**

Typical run:
```bash
python retrieve.py
```

> Configure any query filters or paging inside `retrieve.py` as needed.

---

### Step 2: LLM extraction of factor-outcome associations
**Script:** `llm_extraction.py`  
**Input dir:** `artifacts/epmc_fulltext/`  
**Output:** `artifacts/visual_factors/all_visual_associations.csv`

This step prompts an LLM to parse each paper and emit JSON rows with:
- `factor` (visually detectable)
- `outcome_raw`, `outcome LLM`
- `direction` (positive/negative)
- metadata (title, citation, DOI, notes)

Run:
```bash
python llm_extraction.py
```

---

### Step 3: Clean and filter factors
**Scripts:**  
- `factor_categorization.py` — normalization & filtering  
- `remove_none.py` — drop rows where category is `None`

**Output scale:** **Filtered Factors = 43,257** (after cleaning)  

Run:
```bash
python factor_categorization.py
python remove_none.py
```

---

### Step 4: Split by psychological outcome and direction
**Script:** `divide_six.py`  

Splits the master CSV into six exact bins (case-sensitive matches):
- `positive affect` × `positive` / `negative`
- `negative affect` × `positive` / `negative`
- `stress` × `positive` / `negative`
- *(If present in your data, you can extend to `Well-being`, `perceived level of safety` using this same pattern.)*

Run:
```bash
python divide_six.py
```

---

### Step 5: Deduplicate categories with LLM grouping
**Script:** `deduplicator.py`  

Uses an LLM to cluster near-duplicates (e.g., “GreenSpace”, “Green area”, “Park”) into canonical groups.  
Also **excludes active/seed categories** from being re-used as inputs on subsequent passes.

Run:
```bash
python deduplicator.py
```

---

### Step 6: Build group lists
**Scripts:**  
- `group_lists.py` — emits **per-group CSVs**  
- `unique_group_lists.py` — emits a **single long unique list**

**Outputs dir:** `artifacts/visual_factors/output/` and `artifacts/visual_factors/output/lists/`

Run:
```bash
python group_lists.py
python unique_group_lists.py
```

---

## Outputs

- `artifacts/visual_factors/all_visual_associations.csv` — raw LLM-extracted associations (Step 2)
- `artifacts/visual_factors/output/*.csv` — split, cleaned, and deduplicated tables (Steps 3–5)
- `artifacts/visual_factors/output/lists/*.csv` — final **group lists** per category (Step 6)
- `artifacts/visual_factors/output/unique_group_list.csv` — master **unique** list (Step 6)

