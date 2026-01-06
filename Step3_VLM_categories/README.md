# Step 3: VLM Multi-Category Evaluation

Evaluates images using Vision Language Models (VLM) across 997 psychological categories loaded from an Excel file. Uses temperature=0 for reproducible results.

## Prerequisites

- **Python 3.13+**
- **uv** (Python package manager) - install with: `curl -LsSf https://astral.sh/uv/install.sh | sh`
- **OpenAI-compatible API endpoint** (e.g., OpenAI, local LLM server, or other provider)

## Setup

1. **Install dependencies:**
   ```bash
   uv sync
   ```

2. **Configure environment:**
   ```bash
   cp .env.example .env
   ```
   Edit `.env` and set the required variables:
   - `FOLDERPATH` - Path to folder containing subfolders with images (JPG/PNG)
   - `BASE_URL` - Your LLM API endpoint (e.g., `https://api.openai.com/v1`)
   - `MODEL` - Model name (e.g., `gpt-4o-mini`)
   - `OPENAI_API_KEY` - Your API key

   **Categories configuration** (choose one):
   - **Option A (Excel/CSV):** Set `CATEGORIES_EXCEL_PATH` to your categories file (e.g., `allpsychologicalvariables_greater_equal_3_withoutdublicates.xlsx`)
   - **Option B (Comma-separated):** Set `CATEGORIES=PlantPresence,Greenness,NaturalLightExposure`
   - **Option C (Single category):** Set `CATEGORY=PlantPresence`
   - **Option D (Fallback):** Uses default 5 categories if none specified

   Optional settings (see `.env.example` for defaults):
   - `OUTPUT_DIR` - Where to save results (default: current directory)
   - `MAX_CONCURRENCY` -
1. **I of parallel workers (default: 12)
   - `MAX_IMAGE_SIZE` - Longest side in pixels, 0 = no resize (default: 1280)
   - `CONVERT_TO_EXCEL` - Generate Excel files alongside CSV (default: True)
   - `TEMPERATURE` - Sampling temperature (default: 0.0 for reproducibility)
   - `SEED` - Random seed for reproducibility (default: 42)
   - `MAX_COMPLETION_TOKENS` - Max tokens in response (default: 500)

## Running the Experiments

```bash
uv run python main.py
```

The scri   - **Option B (Comma-separated):** Set `CATEGORIES=PlantPresence,Greenness,NaturalLightExposure`
   - **Option C (Single category):** Set `scale images to reduce API costs (if `MAX_IMAGE_SIZE` > 0)
4. Send each image to the LLM with evaluation prompts for all categories
5. Stream results to CSV files in real-time
6. Convert CSVs to Excel (if enabled)
7. Generate token usage and failure reports

## Input Structure

```
FOLDERPATH/
├── subfolder1/
│   ├── image1.jpg
│   ├── image2.png
│   └── ...
├── subfolder2/
│   └── ...
└── ...
```

## Output

A timestamped folder is created in `OUTPUT_DIR` containing:

- **Evaluation results** (one CSV/Excel per category):
  - `{category_name}.csv` / `{category_name}.xlsx`
  - Example: `plant_presence.csv`, `greenness.csv`, etc.

- **Metadata and logs**:
  - `tokens.json` / `tokens.csv` - Token usage per category and overall
  - `failures.csv` - Failailed execution log
  - `experiment.json` / `experiment_summary.csv` - Run metadata (if `WRITE_EXPERIMENT_METADATA=True`)

Each CSV contains: `PictureId`, `Folder`, `TimeCreated`, `TimeDigitized`, `TimeModified`, plus the category presence score (0-1), confidence (1-10), and image description.

## Evaluation Categories

The 997 psychological categories are loaded from `allpsychologicalvariables_greater_equal_3_withoutdublicates.xlsx`. Each category returns:

- **Presence**: 0 (absence) to 1 (abundance) - except InsideOutside which uses 1=Inside, 2=Outside
- **Confidence**: 1-10 rating of evaluation confidence
- **ImageD└── ...
```
f description of what's visible in the image

## Reproducibility

This step uses `TEMPERATURE=0` and `SEED=42` by default to ensure reproducible results across multiple runs. These settings can be adjusted in `.env` if needed.
