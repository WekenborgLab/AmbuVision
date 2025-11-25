import os
import asyncio
import logging
import json
import re
from pathlib import Path
from datetime import datetime
from io import BytesIO
from typing import List, Dict, Any, Tuple
from itertools import chain
import random

import pandas as pd
from PIL import Image, ExifTags
from tqdm.asyncio import tqdm_asyncio

# Try to import pyvips for resizing, fall back to PIL
try:
    import pyvips

    PYVIPS_AVAILABLE = True
    logging.info("Using pyvips for fast image resizing.")
except ImportError:
    PYVIPS_AVAILABLE = False
    logging.info("pyvips not found. Using PIL for image resizing (slower).")

# Ensure PIL can load large images
Image.MAX_IMAGE_PIXELS = None


class Imager:
    def __init__(self, max_image_size: int, jpeg_quality: int,
                 delete_originals: bool, costs: dict, save_metadata: bool,
                 convert_to_excel: bool, per_image_jobs_concurrency: int):

        # --- Config ---
        self.output_path = ""  # Will be set by main.py
        self.max_image_size = max_image_size
        self.jpeg_quality = jpeg_quality
        self.delete_originals = delete_originals
        self.costs = costs
        self.save_metadata = save_metadata
        self.convert_to_excel = convert_to_excel
        self.per_image_jobs_concurrency = per_image_jobs_concurrency

        # --- Base Data ---
        self.base_columns = [
            "PictureId", "Folder", "TimeCreated",
            "TimeDigitized", "TimeModified"
        ]

        # --- Runtime State (reset for each batch) ---
        self._tokens = {}
        self._failures = []
        self._all_jobs = []
        self._tqdm_progress = None
        self._start_time = None

    def set_output_folder(self, path: str):
        self.output_path = path

    def get_all_picture_paths(self, input_path: str) -> List[str]:
        """
        Finds all pictures in the input folder and returns them as a list.
        Uses the '§' separator to join path and folder name.
        """
        picture_list = []
        for subfolder in Path(input_path).iterdir():
            if subfolder.is_dir():
                image_files = chain(
                    subfolder.glob("*.[jJ][pP][gG]"),
                    subfolder.glob("*.[jJ][pP][eE][gG]"),
                    subfolder.glob("*.[pP][nN][gG]")
                )
                for image_path in image_files:
                    picture_list.append(f'{str(image_path)}§{subfolder.name}')
        return picture_list

    # --- Image Processing ---

    def _pil_resize(self, img: Image.Image) -> Image.Image:
        img.thumbnail((self.max_image_size, self.max_image_size), Image.Resampling.LANCZOS)
        return img

    def _vips_resize(self, img_path: str) -> bytes:
        image = pyvips.Image.new_from_file(img_path)
        image_resized = image.thumbnail_image(self.max_image_size, height=self.max_image_size, crop='none')
        return image_resized.jpegsave_buffer(Q=self.jpeg_quality, strip=True)

    def _read_and_compress(self, image_path: str) -> Tuple[bytes, Dict[str, Any]]:
        with Image.open(image_path) as img:
            metadata = self.readImageTime(img)

        if PYVIPS_AVAILABLE:
            compressed_bytes = self._vips_resize(image_path)
        else:
            with Image.open(image_path) as img:
                if img.mode == 'RGBA' or img.mode == 'P':
                    img = img.convert('RGB')
                img_resized = self._pil_resize(img)
                buffer = BytesIO()
                img_resized.save(buffer, format="JPEG", quality=self.jpeg_quality, optimize=True)
                compressed_bytes = buffer.getvalue()

        return compressed_bytes, metadata

    def readImageTime(self, image: Image.Image) -> Dict[str, str]:
        try:
            exifdata = image.getexif()
        except Exception:
            exifdata = {}

        data = {'TO': '-', 'TD': '-', 'TM': '-'}
        if exifdata:
            for tag_id in exifdata:
                if tag_id == 36867:  # DateTimeOriginal
                    data['TO'] = exifdata[tag_id]
                elif tag_id == 36868:  # DateTimeDigitized
                    data['TD'] = exifdata[tag_id]
                elif tag_id == 306:  # DateTime (Modify)
                    data['TM'] = exifdata[tag_id]
        return data

    # --- Batch Execution ---

    async def run_one_batch(self, middle, jobs: List[Dict], selected_pictures: List[str], max_concurrency: int):
        self._start_time = datetime.now()
        num_to_process = len(selected_pictures)
        if num_to_process == 0:
            logging.warning("No pictures provided for this batch.")
            return

        total_tasks = num_to_process * len(jobs)
        logging.info(f"Creating {len(jobs)} jobs for each of the {num_to_process} pictures.")

        self._tokens = {"total": {"prompt": 0, "completion": 0, "total": 0}}
        self._failures = []
        self._all_jobs = jobs

        # --- Create CSV headers with STRICT SEPARATOR ---
        for job in jobs:
            job["_csv_path"] = os.path.join(self.output_path, job["csv_file"])
            all_columns = self.base_columns + job["columns"]

            # Store the ordered columns list in the job object so we can align data later
            job["_ordered_columns"] = all_columns

            df = pd.DataFrame(columns=all_columns)
            # FIX: Use semicolon separator and utf-8-sig encoding
            df.to_csv(job["_csv_path"], index=False, sep=';', encoding='utf-8-sig')

            self._tokens[job["name"]] = {"prompt": 0, "completion": 0, "total": 0}

        work_q = asyncio.Queue()
        semaphore = asyncio.Semaphore(max_concurrency)

        for pic_path_full in selected_pictures:
            await work_q.put(pic_path_full)

        self._tqdm_progress = tqdm_asyncio(total=total_tasks, desc="Running Jobs", unit="job")

        workers = []
        for _ in range(max_concurrency):
            workers.append(asyncio.create_task(self._worker(middle, work_q, semaphore)))

        await work_q.join()
        for _ in range(max_concurrency):
            await work_q.put(None)
        await asyncio.gather(*workers)

        self._tqdm_progress.close()
        logging.info("\nBatch processing complete.")

        self._save_failures_csv()
        self._save_token_report()
        if self.convert_to_excel:
            self._convert_csvs_to_excel()
        if self.save_metadata:
            self._save_experiment_metadata(num_processed=num_to_process)

    async def _worker(self, middle, work_q: asyncio.Queue, semaphore: asyncio.Semaphore):
        while True:
            pic_path_full = await work_q.get()
            if pic_path_full is None:
                work_q.task_done()
                break

            try:
                await self._process_one_picture(middle, pic_path_full, semaphore)
            except Exception as e:
                logging.error(f"UNHANDLED exception in worker for {pic_path_full}: {e}", exc_info=True)
            finally:
                work_q.task_done()

    async def _process_one_picture(self, middle, image_path_full: str, main_semaphore: asyncio.Semaphore):
        image_path_str = ""
        compressed_bytes = None

        try:
            image_path_str, folder = image_path_full.split('§')
            image_path = Path(image_path_str)
            picture_id = image_path.name
        except Exception as e:
            self._record_failure(job_name="Metadata", picture_id=image_path_full, error=str(e))
            return

        try:
            compressed_bytes, image_time = await asyncio.to_thread(
                self._read_and_compress, image_path_str
            )

            base_data = {
                "PictureId": picture_id, "Folder": folder,
                "TimeCreated": image_time.get('TO', '-'),
                "TimeDigitized": image_time.get('TD', '-'),
                "TimeModified": image_time.get('TM', '-')
            }

            image_job_semaphore = asyncio.Semaphore(self.per_image_jobs_concurrency)

            job_tasks = []
            for job in self._all_jobs:
                job_tasks.append(
                    self._run_job(
                        middle, job, compressed_bytes, base_data,
                        main_semaphore, image_job_semaphore
                    )
                )
            await asyncio.gather(*job_tasks)

        except Exception as e:
            logging.error(f"--- PARENT TASK FAILED: Could not process {picture_id}: {e} ---")
            self._record_failure(job_name="ImageRead", picture_id=picture_id, error=str(e))

        finally:
            del compressed_bytes
            if self.delete_originals and image_path_str:
                try:
                    await asyncio.to_thread(os.remove, image_path_str)
                except Exception as e_del:
                    logging.warning(f"Failed to remove original file {picture_id}: {e_del}")

    async def _run_job(self, middle, job: Dict, image_bytes: bytes, base_data: Dict,
                       main_sem: asyncio.Semaphore, img_sem: asyncio.Semaphore):
        job_name = job["name"]
        picture_id = base_data["PictureId"]
        max_attempts = 6
        retry_delay = 3.0

        async with img_sem:
            async with main_sem:
                for attempt in range(max_attempts):
                    try:
                        eval_str, tokens = await middle.sendImageRequestAsync(image_bytes, job["prompt"])
                        parsed_data = self._parse_llm_output(eval_str, job["columns"])
                        self._accumulate_tokens(job_name, tokens)

                        final_data = {**base_data, **parsed_data}

                        # FIX: Pass the ordered columns list to ensure alignment
                        await self._append_csv_row(job["_csv_path"], final_data, job["_ordered_columns"])

                        self._tqdm_progress.update(1)
                        return

                    except Exception as e:
                        logging.warning(
                            f"[Attempt {attempt + 1}/{max_attempts}] Job '{job_name}' for {picture_id} failed: {e}")
                        if attempt < max_attempts - 1:
                            sleep_time = retry_delay * (2 ** attempt) + random.uniform(0, 1)
                            await asyncio.sleep(sleep_time)
                        else:
                            self._record_failure(job_name, picture_id, str(e))
                            self._tqdm_progress.update(1)
                            return

    # --- Parsing and Saving ---

    def _parse_llm_output(self, llm_output: str, columns: List[str]) -> Dict[str, Any]:
        if not llm_output:
            raise ValueError("LLM returned empty output.")

        llm_output = llm_output.strip()
        data = {}

        # Try JSON
        try:
            if llm_output.startswith("```json"):
                llm_output = llm_output[7:-3].strip()
            elif llm_output.startswith("```"):
                llm_output = llm_output[3:-3].strip()

            data = json.loads(llm_output)

            # Convert numerics
            for key, value in data.items():
                if isinstance(value, str):
                    # Sanitize strings (Remove newlines which break CSVs)
                    data[key] = value.replace('\n', ' ').replace('\r', '')
                    try:
                        data[key] = float(value)
                    except (ValueError, TypeError):
                        pass
            return data

        except json.JSONDecodeError:
            pass

        # Fallback Regex
        cols_to_find = list(columns)
        pattern = re.compile(r"([\w]+):\s*(.*?)(?=\s*[\w]+:|$)", re.DOTALL)
        matches = pattern.finditer(llm_output)

        found_keys = []
        for match in matches:
            key = match.group(1).strip()
            value = match.group(2).strip().rstrip(',')
            # Sanitize
            value = value.replace('\n', ' ').replace('\r', '')

            if key in cols_to_find:
                data[key] = value
                found_keys.append(key)

        if len(found_keys) != len(cols_to_find):
            # Fallback Split
            eval_list = llm_output.split(", ")
            if len(eval_list) == len(cols_to_find):
                for category, value_str in zip(cols_to_find, eval_list):
                    try:
                        value = value_str.split(': ')[1]
                    except IndexError:
                        value = value_str
                    data[category] = value.replace('\n', ' ').replace('\r', '')
            else:
                raise ValueError(f"Fallback parsing failed.")

        # Numeric Conversion
        for key, value in data.items():
            if isinstance(value, str):
                if "Confidence" in key or "Score" in key or "InsideOutside" in key:
                    try:
                        data[key] = float(value)
                    except (ValueError, TypeError):
                        pass
        return data

    async def _append_csv_row(self, csv_path: str, data: Dict, ordered_columns: List[str]):
        """Asynchronously appends a single row to a CSV file."""
        try:
            df = pd.DataFrame([data], columns=ordered_columns)

            # FIX: Use semicolon sep and utf-8-sig
            df.to_csv(csv_path, mode='a', header=False, index=False, sep=';', encoding='utf-8-sig')
        except Exception as e:
            logging.error(f"Failed to append to {csv_path}: {e}")
            self._record_failure("SaveCSV", data.get("PictureId"), str(e))

    # --- Reporting ---

    def _record_failure(self, job_name, picture_id, error):
        logging.error(f"FAILURE: Job='{job_name}' Picture='{picture_id}' Error='{error}'")
        self._failures.append({
            "job_name": job_name,
            "picture_id": picture_id,
            "error": error,
            "timestamp": datetime.now().isoformat()
        })

    def _accumulate_tokens(self, job_name: str, tokens: Dict[str, int]):
        for key in ["prompt", "completion", "total"]:
            self._tokens["total"][key] += tokens.get(key, 0)
            if job_name in self._tokens:
                self._tokens[job_name][key] += tokens.get(key, 0)

    def _save_failures_csv(self):
        if not self._failures:
            logging.info("No failures recorded for this batch.")
            return
        path = os.path.join(self.output_path, f"failures_part{self._start_time.strftime('%Y%m%d')}.csv")
        try:
            df = pd.DataFrame(self._failures)
            is_new_file = not os.path.exists(path)
            # Failures CSV also gets semicolon
            df.to_csv(path, mode='a', header=is_new_file, index=False, sep=';', encoding='utf-8-sig')
            logging.info(f"Saved {len(self._failures)} failures to {path}")
        except Exception as e:
            logging.error(f"Failed to save failures CSV: {e}")

    def _save_token_report(self):
        path = os.path.join(self.output_path, f"token_report_part{self._start_time.strftime('%Y%m%d')}.json")
        report = {"jobs": {}, "total": {}}

        # Calc total costs
        total_p = self._tokens["total"]["prompt"]
        total_c = self._tokens["total"]["completion"]
        cost_p = (total_p / 1000) * self.costs["prompt_1k"]
        cost_c = (total_c / 1000) * self.costs["completion_1k"]
        report["total"] = {**self._tokens["total"], "cost": cost_p + cost_c}

        for job_name, tokens in self._tokens.items():
            if job_name == "total": continue
            p = tokens["prompt"]
            c = tokens["completion"]
            cp = (p / 1000) * self.costs["prompt_1k"]
            cc = (c / 1000) * self.costs["completion_1k"]
            report["jobs"][job_name] = {**tokens, "cost": cp + cc}

        logging.info(
            f"Batch Token Report: Total Tokens={report['total']['total']}, Est. Cost=${report['total']['cost']:.4f}")

        try:
            with open(path, 'a') as f:
                json.dump(report, f, indent=2)
                f.write("\n")
        except Exception as e:
            logging.error(f"Failed to save token report: {e}")

    def _convert_csvs_to_excel(self):
        """Converts all generated CSVs for this batch to Excel files."""
        logging.info("Converting batch CSVs to Excel...")
        for job in self._all_jobs:
            try:
                csv_path = job["_csv_path"]
                xlsx_path = os.path.join(self.output_path, job["xlsx_file"])

                # FIX: Read with strict semicolon separator
                df = pd.read_csv(csv_path, sep=';', encoding='utf-8-sig')
                df.to_excel(xlsx_path, index=False)
            except Exception as e:
                logging.error(f"Failed to convert {csv_path} to Excel: {e}")

    def _save_experiment_metadata(self, num_processed: int):
        path = os.path.join(self.output_path, f"metadata_part{self._start_time.strftime('%Y%m%d')}.json")
        end_time = datetime.now()
        metadata = {
            "run_timestamp": self._start_time.isoformat(),
            "duration_seconds": (end_time - self._start_time).total_seconds(),
            "pictures_processed": num_processed,
            "settings": {
                "max_concurrency": os.getenv('MAX_CONCURRENCY'),
                "max_image_size": self.max_image_size,
                "jpeg_quality": self.jpeg_quality,
            },
            "costs": self.costs,
            "jobs_run": [j["name"] for j in self._all_jobs]
        }
        try:
            with open(path, 'a') as f:
                json.dump(metadata, f, indent=2)
                f.write("\n")
        except Exception as e:
            logging.error(f"Failed to save experiment metadata: {e}")