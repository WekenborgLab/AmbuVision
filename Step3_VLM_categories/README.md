# Step 3: Using 997 categories

## Project Structure

```text
Ambuvision/
├─ Step3_VLM_categories/
│  ├─ Imager.py                                                           # Work with images and jobs
│  ├─ Middle.py                                                           # Middle point between host and server
│  ├─ main.py                                                             # Execution of all scripts combined
│  ├─ allpsychologicalvariables_greater_equal_3_withoutdublicates.xlsx    # excel file with categories
│  ├─ pyproject.toml                                                      # Central configuration file containing project metadata,Python version requirements, dependencies, and build-system settings 
│  ├─ README.md
```

## main.py
### Summary
This script processes all images in a given folder and evaluates each one across a set of 997 categories using LLaMA 4 vision-language model and Qwen3-VL-235B-A22B-Instruct-FP8. Categories can come from environment variables, a single Excel/CSV file, or a built-in fallback list. For each category, the script generates a prompt, runs model inference, and saves results to CSV/Excel.
#### Key features:
Loads settings from .env (paths, categories, concurrency, cost tracking, etc.)
Reads categories from env vars or Excel/CSV
Automatically builds evaluation jobs
Processes all images, optionally resizes/compresses them
Runs model calls in parallel and outputs logs, results, and token usage
#### Run:
python main.py
Requires proper .env configuration and (optionally) pandas/openpyxl for Excel support.

## Imager.py
The Imager class handles all low-level image processing, batching and result writing for the pipeline.
### What it does:
Scans the input folder and collects all jpg/png images (grouped by subfolder).
Robustly opens images (PIL, optionally imageio / pyvips), reads EXIF timestamps, and downsizes/compresses to JPEG.
Runs all jobs per image with bounded async concurrency, including retries with backoff.
Streams results into per-job CSV files and optionally converts them to Excel.
Tracks token usage per job and overall, writing tokens.json and tokens.csv.
Logs all failures to failures.csv.
Optionally writes rich experiment metadata (experiment.json + experiment_summary.csv), including environment, host info and output files.
The main script (main.py) wires this module together with the model client (Middle) and environment configuration.

## Middle module
The Middle class is the API bridge between the image pipeline and the OpenAI-compatible server.
### What it does:
Reads BASE_URL, MODEL and optional generation controls (MAX_COMPLETION_TOKENS, TEMPERATURE, SEED) from environment variables.
Initializes sync and async OpenAI clients and provides a test_connection() helper to verify the server/model.
Encodes images to base64 and sends multimodal chat requests (text + image).
Uses Pydantic (CategoryResponse) for structured responses (presence, confidence, description), with a fallback JSON-parse path.
Returns both the parsed result and optional token usage (prompt_tokens, completion_tokens, total_tokens) for downstream cost tracking.
The main script (main.py) uses Middle in combination with Imager to run all category evaluations over all images.

## Project configuration (pyproject.toml)
The project uses a modern Python packaging setup based on PEP 621 and Hatchling.
Key points:
Defines the project as AmbuVision, version 0.1.0.
Requires Python ≥ 3.13.
Lists core dependencies:
python-dotenv for environment variable loading
openai for the LLM/multimodal API
pillow for image decoding
pandas and openpyxl for CSV/Excel output
Specifies README.md as the long description.
Uses Hatchling as the build backend.
This file enables clean installation, reproducible dependency management, and packaging support for the entire pipeline.

## Categories
The 997 categories are included in the excel sheet "allpsychologicalvariables_greater_3_withoutdublicates.xlsx.
