import os
import re
import asyncio
import logging
from logging.handlers import RotatingFileHandler
from datetime import datetime

from dotenv import load_dotenv
from Middle import Middle
from Imager import Imager

# Try to import pandas for Excel/CSV loading
try:
    import pandas as pd  # type: ignore
except Exception:
    pd = None

# ---------------------------
# Load .env early
# ---------------------------
load_dotenv()

# ---------------------------
# Tunables (env overrides)
# ---------------------------
MAX_CONCURRENCY = int(os.getenv("MAX_CONCURRENCY", 16))                 # worker count
PER_IMAGE_JOBS_CONCURRENCY = int(os.getenv("PER_IMAGE_JOBS_CONCURRENCY", 1))  # prompts in parallel per image (1 safest)

# longest-side cap (0 = no resize)
MAX_IMAGE_SIZE = int(os.getenv("MAX_IMAGE_SIZE", 1280))
JPEG_QUALITY = int(os.getenv("JPEG_QUALITY", 85))

DELETE_ORIGINALS = os.getenv("DELETE_ORIGINALS", "False").lower() == "true"
PROGRESS_REFRESH_INTERVAL = float(os.getenv("PROGRESS_REFRESH_INTERVAL", 0.5))
CONVERT_TO_EXCEL = os.getenv("CONVERT_TO_EXCEL", "True").lower() == "true"  # if False, keep CSV only

# Optional cost estimation (USD) per 1K tokens; leave unset to skip cost calc
PROMPT_COST_PER_1K = os.getenv("PROMPT_COST_PER_1K")
COMPLETION_COST_PER_1K = os.getenv("COMPLETION_COST_PER_1K")
PROMPT_COST_PER_1K = float(PROMPT_COST_PER_1K) if PROMPT_COST_PER_1K else None
COMPLETION_COST_PER_1K = float(COMPLETION_COST_PER_1K) if COMPLETION_COST_PER_1K else None

# Optional experiment metadata outputs
WRITE_EXPERIMENT_METADATA = os.getenv("WRITE_EXPERIMENT_METADATA", "False").lower() == "true"

def _init_logging(log_dir: str):
    os.makedirs(log_dir, exist_ok=True)
    log_path = os.path.join(log_dir, "run.log")

    logger = logging.getLogger("app")
    logger.setLevel(logging.INFO)

    ch = logging.StreamHandler()
    ch.setLevel(logging.INFO)
    ch.setFormatter(logging.Formatter("[%(levelname)s] %(message)s"))

    fh = RotatingFileHandler(log_path, maxBytes=5_000_000, backupCount=3)
    fh.setLevel(logging.INFO)
    fh.setFormatter(logging.Formatter("%(asctime)s %(levelname)s %(name)s - %(message)s"))

    if not logger.handlers:
        logger.addHandler(ch)
        logger.addHandler(fh)

    return logger, log_path

def to_snake(name: str) -> str:
    """CamelCase / spaced → snake_case for file names."""
    s = re.sub(r"[\s\-]+", "_", name.strip())
    s = re.sub(r"([a-z0-9])([A-Z])", r"\1_\2", s)
    s = re.sub(r"[^A-Za-z0-9_]+", "", s)
    s = re.sub(r"_+", "_", s)
    return s.lower().strip("_")

def read_categories_from_excel(path: str, col_label: str | None = None) -> list[str]:
    if pd is None:
        raise RuntimeError(
            "pandas is required to read Excel/CSV categories. Install with: pip install pandas openpyxl"
        )

    if path.lower().endswith((".xlsx", ".xls")):
        df = pd.read_excel(path)
    else:
        df = pd.read_csv(path)

    if col_label and col_label in df.columns:
        series = df[col_label]
    else:
        # default: first column
        series = df.iloc[:, 0]

    cats = [str(x).strip() for x in series.dropna().astype(str).tolist()]
    # filter out accidental non-categories
    cats = [
        c for c in cats
        if c
        and c.lower() != "imagedescription"
        and not c.lower().endswith("confidence")
    ]
    return cats

def load_categories_from_env_or_excel() -> list[str]:
    """
    Precedence:
    1) CATEGORIES (comma-separated)
    2) CATEGORY (single)
    3) CATEGORIES_EXCEL_PATH (+ optional CATEGORIES_COLUMN_LABEL)
    4) fallback to original 5 categories
    """
    env_multi = os.getenv("CATEGORIES")
    env_single = os.getenv("CATEGORY")
    excel_path = os.getenv("CATEGORIES_EXCEL_PATH") or os.getenv("CATEGORY_EXCEL_PATH")
    excel_col = os.getenv("CATEGORIES_COLUMN_LABEL") or os.getenv("CATEGORY_COLUMN_LABEL")

    if env_multi:
        raw = [c.strip() for c in env_multi.split(",")]
    elif env_single:
        raw = [env_single.strip()]
    elif excel_path:
        raw = read_categories_from_excel(excel_path, excel_col)
    else:
        raw = ["NatureScore", "PlantPresence", "NaturalLightExposure", "Greenness", "InsideOutside"]

    # de-duplicate but keep order
    seen = set()
    out = []
    for c in raw:
        key = c.lower()
        if key and key not in seen:
            seen.add(key)
            out.append(c)
    return out

# def create_prompt(category_name, confidence_name, is_inside_outside=False):
#     return f"""
#             Analyze the image regarding the presence of the following category: "{category_name}",
#             return a json with three fields:
#                 - "presence": 0 or 1 value indicating the presence of the category where 0 means absence and 1 means presence,
#                 - "confidence": a value from 1 to 10 indicating your confidence in the presence evaluation,
#                 - "description": a brief description of what is seen in the image.
#     """

def create_prompt(category_name, confidence_name, is_inside_outside=False):
    if is_inside_outside:
        scoring_text = "For InsideOutside category use 1 for Inside and 2 for Outside."
    else:
        scoring_text = ("Analyze the image and evaluate the presence of the category, "
                        "whereas 0 is absolute absence and 1 is abundance of the category, mentioned in criteria.")

    return f"""
            Here are the given categories:
            "{category_name}", "{confidence_name}",
            "ImageDescription".

            {scoring_text} For the category with Confidence in it give out a value from 1 to 10 of how confident you are about the evaluation of the category. For image description 
            describe what is to be seen in the picture. For output use following 
            format Categoryname: value, Category2name: value2.... Do not say anything else
            """

def build_evaluation_jobs(categories: list[str]) -> list[dict]:
    jobs = []
    for cat in categories:
        # handle InsideOutside special rules by name (be forgiving about underscores)
        norm = to_snake(cat)
        is_io = norm in {"insideoutside", "inside_outside", "inside-outside"}

        confidence = f"{cat}Confidence"
        jobs.append({
            "name": cat,
            "prompt": create_prompt(cat, confidence, is_inside_outside=is_io),
            "csv_file": f"{norm}.csv",
            "xlsx_file": f"{norm}.xlsx",
            "columns": [cat, confidence, "ImageDescription"],
        })
    return jobs

async def main():
    input_folder = os.getenv("FOLDERPATH")
    if not input_folder:
        print("FOLDERPATH env var not set. Exiting.")
        return

    base_output_dir = os.getenv("OUTPUT_DIR", ".")  # where the run folder is created

    logging.basicConfig(level=logging.INFO, format='[%(levelname)s] %(message)s')

    middle = Middle()
    imager = Imager(
        input_path=input_folder,
        base_output_dir=base_output_dir,
        max_image_size=MAX_IMAGE_SIZE,
        jpeg_quality=JPEG_QUALITY,
        delete_originals=DELETE_ORIGINALS,
        progress_refresh_interval=PROGRESS_REFRESH_INTERVAL,
        per_image_jobs_concurrency=PER_IMAGE_JOBS_CONCURRENCY,
        convert_to_excel=CONVERT_TO_EXCEL,
        prompt_cost_per_1k=PROMPT_COST_PER_1K,
        completion_cost_per_1k=COMPLETION_COST_PER_1K,
        write_experiment_metadata=WRITE_EXPERIMENT_METADATA,
        experiment_name=os.getenv("EXPERIMENT_NAME"),
        experiment_notes=os.getenv("EXPERIMENT_NOTES"),
    )

    # connectivity check
    try:
        middle.test_connection()
    except Exception:
        print("Exiting application due to connection failure.")
        return

    # Create a unique output folder for this run under OUTPUT_DIR
    run_stamp = datetime.now().strftime("%Y%m%d%H%M%S")
    imager.createFolder(title=run_stamp)
    logger, log_path = _init_logging(imager.getOutputFolder())
    logger.info(f"Logs will be written to {log_path}")
    print(f"Output will be saved to: {imager.getOutputFolder()}")

    print("Finding all available pictures...")
    all_pictures = imager.get_all_picture_paths()
    if not all_pictures:
        print("No pictures found in the input path. Exiting.")
        return
    print(f"Found {len(all_pictures)} total pictures.")

    # --- NEW: load categories & build jobs dynamically ---
    try:
        categories = load_categories_from_env_or_excel()
    except Exception as e:
        logging.getLogger("app").exception(f"Failed to load categories: {e}")
        print("Failed to load categories. Exiting.")
        return

    if not categories:
        print("No categories resolved from env or Excel. Exiting.")
        return

    logger.info(f"Using categories: {categories}")
    evaluation_jobs = build_evaluation_jobs(categories)

    try:
        await imager.run_all(
            middle=middle,
            jobs=evaluation_jobs,
            picture_paths=all_pictures,
            max_concurrency=MAX_CONCURRENCY,
        )
    except Exception as e:
        logging.getLogger("app").exception(f"Unhandled error during run: {e}")
        print("Run aborted due to an unhandled error.")
        return

    print("\nRun complete. See per-job CSV/Excel, failures.csv, and tokens.json/tokens.csv in the output folder.")

if __name__ == "__main__":
    asyncio.run(main())
