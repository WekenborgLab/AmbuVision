import os
import asyncio
import logging
from logging.handlers import RotatingFileHandler
from datetime import datetime

from dotenv import load_dotenv
from Middle import Middle
from Imager import Imager

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

def create_prompt(category_name, confidence_name, is_inside_outside=False):
    if is_inside_outside:
        scoring_text = "For InsideOutside category use 1 for Inside and 2 for Outside."
    else:
        scoring_text = ("Analyze the image and evaluate accordingly from 1 to 10, "
                        "whereas 1 is absolute absence and 10 is abundance of the subject, mentioned in criteria.")

    return f"""
            Here are the given categories:
            "{category_name}", "{confidence_name}",
            "ImageDescription".

            {scoring_text} For the category with Confidence in it give out a value from 1 to 10 of how confident you are about the evaluation of the category. For image description 
            describe what is to be seen in the picture. For output use following 
            format Categoryname: value, Category2name: value2.... Do not say anything else
            """

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

    # Five separate prompts (kept intact)
    evaluation_jobs = [
        {
            "name": "NatureScore",
            "prompt": create_prompt("NatureScore", "NatureScoreConfidence"),
            "csv_file": "nature_score.csv",
            "xlsx_file": "nature_score.xlsx",
            "columns": ["NatureScore", "NatureScoreConfidence", "ImageDescription"]
        },
        {
            "name": "PlantPresence",
            "prompt": create_prompt("PlantPresence", "PlantPresenceConfidence"),
            "csv_file": "plant_presence.csv",
            "xlsx_file": "plant_presence.xlsx",
            "columns": ["PlantPresence", "PlantPresenceConfidence", "ImageDescription"]
        },
        {
            "name": "NaturalLightExposure",
            "prompt": create_prompt("NaturalLightExposure", "NaturalLightExposureConfidence"),
            "csv_file": "natural_light.csv",
            "xlsx_file": "natural_light.xlsx",
            "columns": ["NaturalLightExposure", "NaturalLightExposureConfidence", "ImageDescription"]
        },
        {
            "name": "Greenness",
            "prompt": create_prompt("Greenness", "GreennessConfidence"),
            "csv_file": "greenness.csv",
            "xlsx_file": "greenness.xlsx",
            "columns": ["Greenness", "GreennessConfidence", "ImageDescription"]
        },
        {
            "name": "InsideOutside",
            "prompt": create_prompt("InsideOutside", "InsideOutsideConfidence", is_inside_outside=True),
            "csv_file": "insideoutside.csv",
            "xlsx_file": "insideoutside.xlsx",
            "columns": ["InsideOutside", "InsideOutsideConfidence", "ImageDescription"]
        }
    ]

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
