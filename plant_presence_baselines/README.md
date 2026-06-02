# Plant Presence Baselines

Supplementary AmbuVision pipelines for estimating plant presence and greenness
without uploading study images, generated outputs, model checkpoints, or local
configuration files.

This folder contains:

- `yolo_oiv7/`: YOLOv8x Open Images V7 detection for the class `Plant`.
- `color_baseline/`: cheap RGB/HSV greenness baseline and image-prep helper.

## What is intentionally not tracked

Do not commit raw or processed study data. The `.gitignore` in this folder
excludes image folders, generated CSV/XLSX/JSON/log outputs, local `.env`
files, virtual environments, and model/checkpoint binaries such as `.pt` and
`.ts` files.

If a script needs a model checkpoint, download it locally or set `MODEL` in
your `.env` file to a local path. If a script needs self-report/modelling data,
provide that file locally; it is not included here.

## Input layout

All pipelines expect a folder with one subfolder per participant:

```text
/path/to/participant_image_folders/
  participant_001/
    Foto-001.jpg
    Foto-002.png
  participant_002/
    ...
```

## YOLOv8 Open Images V7 plant detection

```bash
cd plant_presence_baselines/yolo_oiv7
uv sync
cp .env.example .env
```

Edit `.env`:

- `FOLDERPATH`: image root with participant subfolders.
- `OUTPUT_DIR`: where timestamped run outputs should be written.
- `MODEL`: local checkpoint path or an Ultralytics-supported model name.
- `DEVICE`: use `cpu`, `cuda:0`, or another supported PyTorch device.

Run:

```bash
uv run python detect.py
```

The output schema is:

```text
PictureId, Folder, PlantPresence, NumPlantDetections,
PlantCoverageFraction, ProcessSeconds
```

`PlantPresence` is `PlantCoverageFraction * 10`, rounded to two decimals.

## Color baseline

Install Python dependencies in your own environment:

```bash
cd plant_presence_baselines/color_baseline
python -m venv .venv
. .venv/bin/activate
pip install -r requirements.txt
```

Optional: install `pyvips` separately if you want the faster streaming image
resize path in `prep_images.py`. The script falls back to Pillow when `pyvips`
is unavailable.

Prepare resized JPEGs:

```bash
python prep_images.py \
  --input /path/to/raw_or_filtered_images \
  --output /path/to/filtered_prepped \
  --skip-existing
```

Run the cheap color/greenness baseline:

```bash
python color_baseline.py \
  --input /path/to/filtered_prepped \
  --outdir /path/to/color_baseline_outputs \
  --histograms
```

Outputs:

- `color_baseline_per_image.csv`
- `color_baseline_per_participant.csv`
- `greenness.csv`

The generated `greenness.csv` is intended as a drop-in greenness indicator for
the downstream statistical comparison.

## Statistical comparison

The modelling/self-report dataset is not included. Provide it locally and, if
needed, adjust the `CFG` column names near the top of
`color_baseline/compare_baseline_vs_vlm.R`.

```bash
Rscript color_baseline/compare_baseline_vs_vlm.R \
  --greenness /path/to/greenness.csv \
  --modelling /path/to/modelling_dataset.csv \
  --label baseline \
  --outdir /path/to/comparison_outputs
```

Run the same command with `--label vlm` and the VLM `greenness.csv` to append a
side-by-side `comparison_summary.csv`.
