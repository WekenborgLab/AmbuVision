#!/usr/bin/env python3
from pathlib import Path
import os
import csv
import json
import requests
import asyncio
from openai import AsyncOpenAI

# ----------------------
# Config
# ----------------------
INPUT_DIR = Path("artifacts/epmc_fulltext")
OUTPUT_DIR = Path("artifacts/visual_factors")
OUTPUT_CSV = OUTPUT_DIR / "all_visual_associations.csv"
TRUNCATE_CHARS = int(os.environ.get("TRUNCATE_CHARS", "350000"))

MODEL_NAME = os.environ.get("MODEL_NAME", "GPT-OSS-120B")
BASE_URL = os.environ.get("OPENAI_BASE_URL", "http://pluto/v1/")
API_KEY = os.environ.get("VIRTUAL_API_KEY", "VIRTUAL_API_KEY")

EPMC_SEARCH_URL = "https://www.ebi.ac.uk/europepmc/webservices/rest/search"

SYSTEM_MSG = (
    "You are an expert research assistant for evidence extraction. "
    "Return ONLY valid JSON with no extra commentary."
)

# --- Prompt (UPDATED) ---
PROMPT_HEADER_TEMPLATE = """Task:
From the paper below, identify every reported association between visually observable contextual factors
(i.e., elements that could be recognized in a single smartphone photo by a vision–language model)
and Psychological variable namely positive affect, negative affect and stress in humans. Do not report associations which have been found in studies relying solely on animals or cell/molecular lab contexts, cages/mesh platforms, Morris water maze, gene/protein markers. 
Does the factor increase or decrease the level of the psychological variable? The answer to this question is to be stored as ‘direction’. 
Only include factors that are immediately and visually observable in an image. Do not include things that would require any contextual or background knowledge (like whether someone is family, a therapist, or a friend) which is not obvious in the picture.
Also, report if the factor is active or passive. Use the following definition for the same. 
ACTIVE: Factor involves active engagement by the person who feels the psychological variable(e.g., eating, using).
PASSIVE: Factor is present or perceived in the environment (e.g., seeing, being near).

STRICT SCENE SCOPE (must be visible in a single photo of human everyday life):
- Valid factors: for example proportion of green/blue (vegetation/water), proportion of grey/built surfaces, trees/water features, animals/pets,
  number of people/crowding, companions (alone/with others), litter/clutter, indoor/outdoor, location type
  (home/office/park/street/transport/shop), daylight/lighting, uniforms/badges/signage, food/alcohol/cigarettes, screens/devices, clear room type cues.
- Exclude if the factor requires multi-step events or specialized contexts not evident in a photo
- Exclude VR/AR-only stimuli as the "factor" (unless the factor is the visible physical viewing setup/room).

SIMPLE OUTCOME VOCAB ONLY:
Set "outcome" to EXACTLY ONE of: ["positive affect", "negative affect", "stress"].
Use your own reasoning to map the paper’s outcome wording to the closest one of these. Do not output any other label.

For each distinct association, return objects with keys:
- "factor" (short, human-readable, concrete and visible)
- "Exposure_type"  (ACTIVE | PASSIVE)
- "outcome_raw" (the Psychological variable wording used in the paper, verbatim/near-verbatim)
- "Psychological_variable" (exactly one of the three allowed labels above)
- "direction" ("increase" | "decrease" | "mixed/none")
- "vlm_detectability" ("likely" | "possible" | "unclear")
- "notes" (≤20 words, optional)

Validation rules:
1) If the factor is not directly visible in a single photo, discard it.
2) Only keep items whose "outcome" is one of the three allowed labels above.
3) Prefer Abstract/Results/Discussion; avoid Introduction/speculation.

Output: ONLY a JSON array of objects with exactly the keys above.

Paper ID: {paper_id}

---BEGIN FULL TEXT---
"""
PROMPT_FOOTER = """
---END FULL TEXT---
"""

# Post-filters for obviously non-photo or lab contexts
BANNED_FACTOR_KEYWORDS = (
    "travel", "passport", "visa",
    "virtual", "vr", "ar", "online", "internet", "screening video",
    "mouse", "mice", "rat", "rodent", "cage", "mesh", "maze", "morris", "platform",
    "lab", "laboratory", "gene", "mrna", "immuno", "iba-1", "qpcr",
    "mri", "eeg", "ecg"
)

ALLOWED_NORMALIZED = {
    "positive affect", "negative affect",  "stress"
}

ALLOWED_DIRECTION = {"increase", "decrease", "mixed/none"}
ALLOWED_VLM = {"likely", "possible", "unclear"}
ALLOWED_EXPOSURE = {"ACTIVE", "PASSIVE"}

def factor_is_visual(factor: str) -> bool:
    t = (factor or "").lower()
    return bool(t) and not any(k in t for k in BANNED_FACTOR_KEYWORDS)

def sanitize(s: str) -> str:
    return (s or "").replace("\n", " ").replace("\r", " ").strip()

def norm_label(s: str) -> str:
    s = (s or "").strip()
    if not s:
        return ""
    low = s.lower()
    # normalize spacing/case
    if low == "positive affect":
        return "positive affect"
    if low == "negative affect":
        return "negative affect"
    if low == "stress":
        return "stress"
    return s 

def build_prompt(paper_id: str, text: str) -> str:
    if len(text) > TRUNCATE_CHARS:
        text = text[:TRUNCATE_CHARS]
    header = PROMPT_HEADER_TEMPLATE.format(paper_id=paper_id)  
    return f"{header}{text}{PROMPT_FOOTER}"

async def ask_llm_extract(client: AsyncOpenAI, sem: asyncio.Semaphore, paper_id: str, text: str):
    prompt = build_prompt(paper_id, text)
    async with sem:
        resp = await client.chat.completions.create(
            model=MODEL_NAME,
            messages=[{"role":"system","content":SYSTEM_MSG},
                      {"role":"user","content":prompt}],
            temperature=0.1, top_p=1.0,
            max_completion_tokens=2048, 
        )
    content = resp.choices[0].message.content
    try:
        data = json.loads(content)
        return data if isinstance(data, list) else []
    except Exception:
        return []

def doi_to_url(doi: str) -> str:
    """Normalize various DOI strings to a clickable https://doi.org/... URL."""
    doi = (doi or "").strip()
    if not doi:
        return ""
    if doi.lower().startswith("doi:"):
        doi = doi[4:].strip()
    if doi.startswith("http://") or doi.startswith("https://"):
        return doi.replace("dx.doi.org", "doi.org")
    return f"https://doi.org/{doi}"

def epmc_metadata_for_pmcid(pmcid: str):
    """Return (title, citation, doi_url) for a PMCID using Europe PMC."""
    params = {"query": f"PMCID:{pmcid}", "format": "json", "pageSize": 1, "resultType": "lite"}
    r = requests.get(EPMC_SEARCH_URL, params=params, timeout=20)
    r.raise_for_status()
    data = r.json()
    item = data["resultList"]["result"][0]
    title = (item.get("title") or "").strip()
    year = (item.get("pubYear") or "").strip()
    first_chunk = (item.get("authorString") or "").split(";")[0].strip()
    surname = (first_chunk.split(",")[0] if "," in first_chunk else first_chunk.split()[0]).strip()
    citation = f"{surname} et al. {year}"
    doi_raw = (item.get("doi") or "").strip()
    doi_url = doi_to_url(doi_raw)
    return (title, citation, doi_url)

# ----------------------
# Async pipeline
# ----------------------
MAX_CONCURRENT = int(os.environ.get("MAX_CONCURRENT", "200"))  
N_FILES = 64597

async def process_file(idx: int, total: int, fpath: Path, client: AsyncOpenAI,
                       sem: asyncio.Semaphore, writer, writer_lock: asyncio.Lock) -> int:
    try:
        text = fpath.read_text(encoding="utf-8", errors="ignore")
        pmcid = fpath.stem

        # 1) Extract
        items = await ask_llm_extract(client, sem, pmcid, text)

        prepared = []
        for it in items:
            if not isinstance(it, dict):
                continue

            factor = sanitize(it.get("factor", ""))
            if not factor or not factor_is_visual(factor):
                continue

            exposure = sanitize(it.get("Exposure_type", "")).upper()
            if exposure not in ALLOWED_EXPOSURE:
                exposure = ""  

            outcome_raw = sanitize(
                it.get("outcome_raw", "") or it.get("outcome", "") or it.get("Psychological_variable", "")
            )
            if not outcome_raw:
                continue

            pv = norm_label(it.get("Psychological_variable", "") or it.get("outcome", ""))
            if pv not in ALLOWED_NORMALIZED:
                continue

            direction = sanitize(it.get("direction", "")).lower()
            if direction not in ALLOWED_DIRECTION:
                # allow some common variants just in case
                if direction in {"increase", "increased"}:
                    direction = "increase"
                elif direction in {"decrease", "decreased"}:
                    direction = "decrease"
                elif direction in {"mixed", "none", "mixed/none"}:
                    direction = "mixed/none"
                else:
                    direction = ""

            vlm = sanitize(it.get("vlm_detectability", "")).lower()
            if vlm not in ALLOWED_VLM:
                vlm = "possible"

            notes = sanitize(it.get("notes", ""))[:200]

            prepared.append({
                "factor": factor,
                "Exposure_type": exposure,
                "outcome_raw": outcome_raw,
                "Psychological_variable": pv,
                "direction": direction,
                "vlm_detectability": vlm,
                "notes": notes,
            })

        if not prepared:
            print(f"[{idx}/{total}] {fpath.name}: 0 items")
            return (0, 0, 0, 0)

        # 2) Metadata (run blocking requests in a thread)
        title, citation, doi_url = await asyncio.to_thread(epmc_metadata_for_pmcid, pmcid)
        if not title:
            title = sanitize(text.splitlines()[0] if text.splitlines() else "")

        # 3) Prepare rows and write 
        seen = set()
        rows = []
        written = 0
        for rec in prepared:
            rowkey = (
                rec["factor"].lower(),
                rec["outcome_raw"].lower(),
                rec["Psychological_variable"].lower(),
                rec["direction"].lower(),
                rec["Exposure_type"].upper(),
            )
            if rowkey in seen:
                continue
            seen.add(rowkey)
            rows.append([
                rec["factor"],
                rec["Exposure_type"],
                rec["outcome_raw"],
                rec["Psychological_variable"],
                rec["direction"],
                rec["notes"],
                title,
                citation,
                doi_url,
                rec["vlm_detectability"],
            ])

            written += 1

        if rows:
            async with writer_lock:
                writer.writerows(rows)
        active_in_rows = sum(1 for r in rows if r[1] == "ACTIVE")
        passive_in_rows = sum(1 for r in rows if r[1] == "PASSIVE")
        had_any = 1 if written > 0 else 0

        print(f"[{idx}/{total}] {fpath.name}: {written} rows")
        return (written, active_in_rows, passive_in_rows, had_any)
    
    except Exception as e:
        print(f"[{idx}/{total}] {fpath.name}: ERROR ({e})")
        return (0, 0, 0, 0)

async def main_async():
    OUTPUT_DIR.mkdir(parents=True, exist_ok=True)
    client = AsyncOpenAI(api_key=API_KEY, base_url=BASE_URL)

    all_files = sorted(INPUT_DIR.glob("*.txt"))
    files = all_files[:N_FILES]  # first 100 files
    total = len(files)
    print(f"Processing {total} files from {INPUT_DIR} ...")

    sem = asyncio.Semaphore(MAX_CONCURRENT)
    writer_lock = asyncio.Lock()
    total_rows = 0

    # Open the CSV once; write header up front
    with OUTPUT_CSV.open("w", encoding="utf-8", newline="") as fcsv:
        writer = csv.writer(fcsv)
        writer.writerow([
            "factor",
            "Exposure_type",         
            "outcome_raw",
            "Psychological_variable", 
            "direction",              # increase | decrease | mixed/none
            "notes",
            "study_title",
            "citation",
            "DOI",
            "vlm_detectability",
        ])

        tasks = [
            asyncio.create_task(process_file(i, total, f, client, sem, writer, writer_lock))
            for i, f in enumerate(files, 1)
        ]
        results = await asyncio.gather(*tasks, return_exceptions=False)

        total_rows = sum(r[0] for r in results)
        active_total = sum(r[1] for r in results)
        passive_total = sum(r[2] for r in results)
        papers_with_any = sum(r[3] for r in results)


    print(f"Done. Wrote rows: {total_rows}, Output: {OUTPUT_CSV}")
    print(f"Papers with ≥1 factor written: {papers_with_any}")
    print(f"Exposure rows — ACTIVE: {active_total} | PASSIVE: {passive_total}")


if __name__ == "__main__":
    asyncio.run(main_async())
