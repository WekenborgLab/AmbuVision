import os
import asyncio
import logging
import random
import re
from datetime import datetime
from dotenv import load_dotenv
import pandas as pd
from Middle import Middle
from Imager import Imager
from typing import List, Dict, Any  # Added typing


# --- NEW: Chunking Helper ---
def chunk_list(items: List[Any], chunk_size: int) -> List[List[Any]]:
    """Splits a list into smaller lists of a max size."""
    if chunk_size <= 0:  # Handle 0 or negative
        chunk_size = 1

    chunks = []
    for i in range(0, len(items), chunk_size):
        # This handles the last "remainder" chunk correctly
        chunks.append(items[i:i + chunk_size])
    return chunks


# --- PROMPT ENGINEERING (MODIFIED) ---

def create_single_prompt(category_name, confidence_name, is_inside_outside=False):
    """
    Creates the original prompt for a single category.
    The response format is key: value, key: value
    """
    if is_inside_outside:
        scoring_text = "For InsideOutside category use 1 for Inside and 2 for Outside."
    else:
        scoring_text = "Analyze the image and evaluate accordingly from 1 to 10, whereas 1 is absolute absence and 10 is abundance of the subject, mentioned in criteria."

    return f"""
            Here are the given categories:
            "{category_name}", "{confidence_name}",
            "ImageDescription".

            {scoring_text} For the category with Confidence in it give out a value from 1 to 10 of how confident you are about the evaluation of the category. For image description 
            describe what is to be seen in the picture. For output use following 
            format Categoryname: value, Category2name: value2.... Do not say anything else
            """


def create_chunked_prompt(categories: List[str]) -> tuple[str, List[str]]:
    """
    Creates a new, complex prompt for a *chunk* of categories.
    The response format is **JSON**.
    Returns (prompt_text, list_of_all_columns)
    """

    prompt_categories = []
    confidence_categories = []
    all_columns = []

    # Check for the one special case
    is_inside_outside_in_chunk = "InsideOutside" in categories

    for cat in categories:
        confidence_name = f"{cat}Confidence"
        prompt_categories.append(f'"{cat}"')
        confidence_categories.append(f'"{confidence_name}"')

        all_columns.append(cat)
        all_columns.append(confidence_name)

    # Add the shared description column
    all_columns.append("ImageDescription")

    # Build the scoring rules
    scoring_rules = [
        "Analyze the image and evaluate each category from 1 to 10, where 1 is absolute absence and 10 is abundance.",
        "Also provide a confidence score (1-10) for each evaluation."
    ]
    if is_inside_outside_in_chunk:
        scoring_rules.append('SPECIAL RULE: For the "InsideOutside" category, use 1 for Inside and 2 for Outside.')

    scoring_text = "\n".join(scoring_rules)

    # Build the list of JSON keys the AI must return
    json_keys_list = ",\n".join(all_columns)

    # The prompt is now much stricter.
    prompt = f"""
Analyze the image based on the following categories:
{', '.join(prompt_categories)}

**Instructions:**
1. {scoring_text}
2. For "ImageDescription", briefly describe the image.
3. You MUST return ONLY a single, valid JSON object.
4. Do NOT include markdown ```json or any text before or after the JSON.
5. The JSON object MUST contain *only* these keys:
---
{json_keys_list}
---
"""
    return prompt, all_columns


# --- LOGGING SETUP ---
def _init_logging(log_folder: str, log_filename: str = "run.log"):
    """Sets up logging to both console and a file."""
    log_path = os.path.join(log_folder, log_filename)

    logging.basicConfig(
        level=logging.INFO,
        format="%(asctime)s [%(levelname)s] %(message)s",
        handlers=[
            logging.FileHandler(log_path),
            logging.StreamHandler()
        ]
    )
    logging.info(f"Logging initialized. Log file: {log_path}")


# --- EXCEL CATEGORY LOADER ---
def load_categories_from_excel(path: str) -> list[str]:
    """
    Loads categories from an Excel file.
    Assumes a column named 'categories'.
    """
    logging.info(f"Loading categories from Excel file: {path}")
    try:
        df = pd.read_excel(path)

        if 'categories' not in df.columns:
            logging.error(f"Excel file missing required 'categories' column in {path}")
            raise ValueError(f"No 'categories' column found in {path}")

        # Get unique, non-empty categories
        categories = df['categories'].dropna().unique().tolist()
        categories = [str(cat).strip() for cat in categories if str(cat).strip()]

        if not categories:
            logging.error(f"No categories found in the 'categories' column.")
            raise ValueError("Category list is empty.")

        logging.info(f"Successfully loaded {len(categories)} unique categories.")
        return categories

    except FileNotFoundError:
        logging.error(f"Category Excel file not found at path: {path}")
        raise
    except Exception as e:
        logging.error(f"Failed to read Excel file: {e}")
        raise


# (Sanitize function removed, as chunk names are simpler)

# --- ASYNC MAIN FUNCTION ---
async def main():
    load_dotenv()

    # --- 1. Load Configuration ---
    try:
        # Paths
        input_path = os.getenv('INPUT_FOLDER_PATH')
        output_path = os.getenv('OUTPUT_FOLDER_PATH')
        category_excel_path = os.getenv('CATEGORY_EXCEL_PATH')

        # Concurrency
        max_concurrency = int(os.getenv('MAX_CONCURRENCY', 16))
        per_image_jobs_concurrency = int(os.getenv('PER_IMAGE_JOBS_CONCURRENCY', 8))

        # Image Processing
        max_image_size = int(os.getenv('MAX_IMAGE_SIZE', 1024))
        jpeg_quality = int(os.getenv('JPEG_QUALITY', 75))
        delete_originals = os.getenv('DELETE_ORIGINALS', 'False').lower() == 'true'

        # --- NEW CHUNKING CONFIG ---
        category_chunk_size = int(os.getenv('CATEGORY_CHUNK_SIZE', 1))
        # --- END NEW ---

        # Costs (for reporting)
        costs = {
            "prompt_1k": float(os.getenv('PROMPT_COST_PER_1K', 0)),
            "completion_1k": float(os.getenv('COMPLETION_COST_PER_1K', 0))
        }

        # Admin
        save_metadata = os.getenv('SAVE_EXPERIMENT_METADATA', 'True').lower() == 'true'
        convert_to_excel = os.getenv('CONVERT_TO_EXCEL', 'True').lower() == 'true'
        num_pictures_per_set = int(os.getenv('NUM_PICTURES_PER_SET', 2))

        if not input_path or not output_path:
            raise ValueError("INPUT_FOLDER_PATH and OUTPUT_FOLDER_PATH must be set in .env")
        if not category_excel_path:
            raise ValueError("CATEGORY_EXCEL_PATH must be set in .env")

    except Exception as e:
        print(f"Error reading .env configuration: {e}")
        print("Please ensure your .env file is correct. Exiting.")
        exit()

    # --- 2. Initialize Clients ---
    middle = Middle()
    imager = Imager(
        max_image_size=max_image_size,
        jpeg_quality=jpeg_quality,
        delete_originals=delete_originals,
        costs=costs,
        save_metadata=save_metadata,
        convert_to_excel=convert_to_excel,
        per_image_jobs_concurrency=per_image_jobs_concurrency
    )

    # --- 3. Test Connection (Fail-Fast) ---
    try:
        middle.test_connection()
    except Exception as e:
        print(f"Failed to connect to LLM: {e}")
        print("Exiting application.")
        exit()

    # --- 4. Get User Input for Sets ---
    try:
        start_set = int(input("Enter the START set number (e.g., 1): "))
        end_set = int(input("Enter the END set number (e.g., 3): "))
        if end_set < start_set:
            print("End set must be greater than or equal to start set. Exiting.")
            exit()
    except ValueError:
        print("Invalid number. Exiting.")
        exit()

    # --- 5. Setup Output Folder & Logging ---
    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    run_output_folder = os.path.join(output_path, f"run_{timestamp}")
    os.makedirs(run_output_folder, exist_ok=True)

    imager.set_output_folder(run_output_folder)
    _init_logging(run_output_folder)

    logging.info(f"Run started. Output will be saved to: {run_output_folder}")
    logging.info(f"Input folder: {input_path}")

    # --- 6. Load Categories & Find Pictures ---
    try:
        all_categories = load_categories_from_excel(category_excel_path)
    except Exception as e:
        logging.critical(f"Failed to start: {e}. Exiting.")
        exit()

    logging.info("Finding all available pictures...")
    all_pictures_pool = imager.get_all_picture_paths(input_path)
    if not all_pictures_pool:
        logging.error("No pictures found in the input path. Exiting.")
        return
    logging.info(f"Found {len(all_pictures_pool)} total pictures.")

    # --- 7. Run Batch Loop ---
    for set_number in range(start_set, end_set + 1):
        batch_suffix = f"part{set_number}"
        logging.info(f"\n--- STARTING SET {set_number}/{end_set} ({batch_suffix}) ---")

        if not all_pictures_pool:
            logging.warning("No more pictures left in the pool. Stopping all remaining sets.")
            break

        # Select pictures for this batch
        num_to_process_this_set = min(num_pictures_per_set, len(all_pictures_pool))
        selected_pictures = random.sample(all_pictures_pool, num_to_process_this_set)
        all_pictures_pool = [p for p in all_pictures_pool if p not in selected_pictures]  # Remove selected
        logging.info(
            f"Processing {len(selected_pictures)} pictures for this set. {len(all_pictures_pool)} pictures remaining.")

        # 1. Split all categories into chunks
        category_chunks = chunk_list(all_categories, category_chunk_size)

        if category_chunk_size > 1:
            logging.info(
                f"Chunking enabled. Splitting {len(all_categories)} categories into {len(category_chunks)} jobs (size={category_chunk_size}).")
        else:
            logging.info(f"Chunking disabled. Generating {len(all_categories)} jobs (size=1).")

        evaluation_jobs = []

        # 2. Loop over the CHUNKS, not the individual categories
        for i, category_chunk in enumerate(category_chunks):

            if category_chunk_size == 1:
                # --- SINGLE CATEGORY (Original) LOGIC ---
                # (This ensures old, safe prompts are used if chunk_size=1)
                category = category_chunk[0]  # Get the only item
                confidence_name = f"{category}Confidence"
                safe_filename = re.sub(r'[\\/:*?"<>|]+', '', category.strip().lower().replace(' ', '_'))[:100]
                is_special_case = (category.lower() == 'insideoutside')

                job = {
                    "name": category,
                    "prompt": create_single_prompt(category, confidence_name, is_inside_outside=is_special_case),
                    "csv_file": f"{safe_filename}_{batch_suffix}.csv",
                    "xlsx_file": f"{safe_filename}_{batch_suffix}.xlsx",
                    "columns": [category, confidence_name, "ImageDescription"]
                }

            else:
                # --- NEW CHUNKED LOGIC ---
                chunk_name = f"Chunk{i + 1}"  # e.g., Chunk1, Chunk2

                # Create the JSON prompt and get the column list
                prompt, columns = create_chunked_prompt(category_chunk)

                job = {
                    "name": chunk_name,  # Job name is now "Chunk1"
                    "prompt": prompt,
                    "csv_file": f"{chunk_name.lower()}_{batch_suffix}.csv",  # chunk1_part1.csv
                    "xlsx_file": f"{chunk_name.lower()}_{batch_suffix}.xlsx",  # chunk1_part1.xlsx
                    "columns": columns  # All columns for this chunk
                }

            evaluation_jobs.append(job)

        # --- END OF KEY CHANGE ---

        if not evaluation_jobs:
            logging.error("No jobs were generated from the category list. Skipping this set.")
            continue

        try:
            # Run the *entire* batch
            await imager.run_one_batch(
                middle,
                evaluation_jobs,
                selected_pictures,
                max_concurrency
            )
            logging.info(f"--- FINISHED SET: {batch_suffix} ---")

        except ConnectionAbortedError as e:
            logging.critical(f"--- APPLICATION HALTED (Set {batch_suffix}) ---")
            logging.critical(f"A fatal connection error occurred: {e}")
            logging.critical("Stopping all remaining sets.")
            break
        except Exception as e:
            logging.critical(f"An unexpected error occurred in set {batch_suffix}: {e}", exc_info=True)
            logging.critical("Stopping all remaining sets.")
            break

    logging.info("\nAll requested sets have been processed.")


# --- RUN THE ASYNC MAIN FUNCTION ---
if __name__ == "__main__":
    asyncio.run(main())