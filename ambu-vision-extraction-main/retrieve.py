#!/usr/bin/env python3
import argparse
from pathlib import Path
import requests
import xml.etree.ElementTree as ET

SEARCH_URL = "https://www.ebi.ac.uk/europepmc/webservices/rest/search"
FULLTEXT_URL = "https://www.ebi.ac.uk/europepmc/webservices/rest/{pmcid}/fullTextXML"

# ------------------------------------------------------------------------------------
# Expanded optional concept terms: Environment, Social, Biological, Affective, Methods
# ------------------------------------------------------------------------------------
OPTIONAL_TERMS = [

    # --- Environment / Context / Digital ---
    "environment", "context", "setting", "surroundings", "indoor", "outdoor",
    "physical environment", "social context", "built environment", "natural environment",
    "urban environment", "green space", "nature exposure", "daylight", "light", "lighting",
    "noise", "weather", "temperature", "time of day", "location", "room", "scene",
    "visual environment", "physical context", "smartphone", "screen", "digital", "phone",
    "computer", "device", "mobile phone", "social media", "digital media", "technology use",
    "digital environment", "online", "screen exposure", "digital context",

    # --- Social context / presence ---
    "social context", "people", "others", "interaction", "companionship", "presence",
    "alone", "together", "crowd", "group", "social interaction", "partner", "family",
    "friends", "social company", "social presence", "social environment", "being alone",
    "with others", "social isolation", "social contact", "social proximity",

    # --- Biological / living elements ---
    "animal", "pets", "wildlife", "dogs", "cats", "nature exposure",
    "animal-assisted", "pet ownership", "contact with animals", "animal presence",

    # --- Affective / emotional / well-being terms ---
    "affect", "emotion", "mood", "stress", "wellbeing", "well-being", "health",
    "positive affect", "negative affect", "happiness", "anxiety", "depression",
    "valence", "mental health", "emotional experience", "psychological stress",
    "subjective well-being", "Stress, Psychological",

    # --- Methodological context (study type / approach) ---
    "laboratory", "experimental study", "ecological momentary assessment",
    "experience sampling", "ambulatory assessment", "daily diary", "field study",
    "naturalistic study", "momentary assessment", "mobile sensing", "in situ",
    "real-world study", "smartphone", "wearable",
]
# ------------------------------------------------------------------------------------

def _quote(term: str) -> str:
    return f'"{term}"' if any(ch in term for ch in " -/") else term

def build_query(restrict: bool = False, kw_optional: bool = False) -> str:
    """
    required: psychology (keyword or title/abstract)
    restrict: require at least one optional term (in title/abstract OR keywords)
    kw_optional: require at least one optional term specifically in keywords
                 (takes precedence over restrict if both are set)
    """
    required = '(KW:psychology OR TITLE_ABS:psycholog*)'

    if kw_optional:
        parts = [f"KW:{_quote(t)}" for t in OPTIONAL_TERMS]
        optional_group = "(" + " OR ".join(parts) + ")"
        query = f"({required}) AND {optional_group}"
    elif restrict:
        parts = []
        for t in OPTIONAL_TERMS:
            qt = _quote(t)
            parts.append(f"TITLE_ABS:{qt}")
            parts.append(f"KW:{qt}")
        optional_group = "(" + " OR ".join(parts) + ")"
        query = f"({required}) AND {optional_group}"
    else:
        query = required

    return f"({query}) AND OPEN_ACCESS:y"

def epmc_iter_all(query: str, batch: int = 1000, max_records: int | None = None):
    """
    Iterate over ALL matching results using cursor-based paging.
    Yields result dicts. Stops when exhausted or when max_records reached.
    """
    cursor = "*"
    fetched = 0
    while True:
        params = {
            "query": query,
            "format": "json",
            "pageSize": batch,
            "resultType": "lite",
            "cursorMark": cursor,
        }
        r = requests.get(SEARCH_URL, params=params, timeout=60)
        r.raise_for_status()
        data = r.json()
        results = (data.get("resultList") or {}).get("result", []) or []
        next_cursor = data.get("nextCursorMark")

        for rec in results:
            yield rec
            fetched += 1
            if max_records is not None and fetched >= max_records:
                return

        if not results or not next_cursor or next_cursor == cursor:
            return
        cursor = next_cursor

def fetch_fulltext_xml(pmcid: str) -> bytes:
    url = FULLTEXT_URL.format(pmcid=pmcid)
    r = requests.get(url, timeout=120)
    r.raise_for_status()
    return r.content

def _itertext_clean(elem: ET.Element) -> str:
    parts = []
    for t in elem.itertext():
        s = " ".join(t.split())
        if s:
            parts.append(s)
    return " ".join(parts)

def jats_xml_to_text(xml_bytes: bytes) -> str:
    """
    Minimal text extraction:
    - Title (article-title)
    - Abstract(s)
    - Body paragraphs (p under body/sections)
    """
    root = ET.fromstring(xml_bytes)

    titles = root.findall(".//{*}article-title")
    title = _itertext_clean(titles[0]) if titles else ""

    abstracts = root.findall(".//{*}abstract")
    abstract_txt = "\n\n".join(_itertext_clean(a) for a in abstracts) if abstracts else ""

    body_paras = []
    for p in root.findall(".//{*}body//{*}p"):
        txt = _itertext_clean(p)
        if txt:
            body_paras.append(txt)
    body_txt = "\n\n".join(body_paras)

    blocks = []
    if title:
        blocks.append(title)
    if abstracts:
        blocks.append("ABSTRACT\n" + abstract_txt)
    if body_txt:
        blocks.append("MAIN TEXT\n" + body_txt)

    return "\n\n".join(blocks).strip()

def save_text(outdir: Path, pmcid: str, title: str, text: str) -> Path:
    outdir.mkdir(parents=True, exist_ok=True)
    path = outdir / f"{pmcid}.txt"
    with path.open("w", encoding="utf-8") as f:
        if title:
            f.write(title + "\n")
            f.write("-" * 80 + "\n")
        f.write(text)
        f.write("\n")
    return path

def main():
    ap = argparse.ArgumentParser(
        description="Stream ALL Europe PMC OA psychological studies and save full text."
    )
    ap.add_argument("--restrict", action="store_true",
                    help="Require at least one optional term (in title/abstract OR keywords).")
    ap.add_argument("--require-kw-optional", action="store_true",
                    help="Require at least one optional term specifically in keywords (KW:).")
    ap.add_argument("--batch", type=int, default=1000,
                    help="Batch size per API call (default 1000).")
    ap.add_argument("--max", type=int, default=None,
                    help="Optional cap on total records to process (default unlimited).")
    ap.add_argument("--outdir", default="artifacts/epmc_fulltext",
                    help="Where to save .txt files (default: artifacts/epmc_fulltext)")
    args = ap.parse_args()

    q = build_query(restrict=args.restrict, kw_optional=args.require_kw_optional)
    print("Europe PMC query:\n", q)
    print(f"\nFetching in batches of {args.batch} (max: {'∞' if args.max is None else args.max})...\n")

    outdir = Path(args.outdir)
    n_listed = 0
    n_saved = 0

    try:
        for rec in epmc_iter_all(q, batch=args.batch, max_records=args.max):
            n_listed += 1
            pmcid = rec.get("pmcid")
            title = (rec.get("title") or "(no title)").strip()
            print(f"{n_listed:>7}. {pmcid or '-'}\t{title[:140]}")

            if not pmcid:
                continue
            try:
                xml_bytes = fetch_fulltext_xml(pmcid)
                txt = jats_xml_to_text(xml_bytes)
                if not txt:
                    print(f"         Skipping {pmcid}: empty parsed text")
                    continue
                p = save_text(outdir, pmcid, title, txt)
                print(f"         Saved: {p}")
                n_saved += 1
            except requests.HTTPError as e:
                code = e.response.status_code if e.response is not None else "ERR"
                print(f"         Skipping {pmcid}: HTTP {code}")
            except Exception as e:
                print(f"         Skipping {pmcid}: {e}")

    except KeyboardInterrupt:
        print("\nInterrupted by user.")

    print(f"\nDone. Listed: {n_listed} | Saved: {n_saved}")

if __name__ == "__main__":
    main()
