# Ambuvision - Image Evaluation

A customizable programm to allow different levels of setting, chunking and job management

---

## Table of contents
- [Overview](#overview)
- [Directory Layout](#directory-layout)
- [Prerequisite](#prerequisites)
- [Quick start](#quick-start)
- [Step-by-step](#step-by-step)
  - [Step 1: Extraction and downscaling of pictures](#step-1-extraction-and-downscaling-of-pictures)
  - [Step 2: Creating jobs and correspondent requests to LLM-Server](#step-2-creating-jobs-and-correspondent-requests-to-llm-server)
  - [Step 3: Conversion of the output into both Excel and CSV file formats](#step-3-conversion-of-the-output-into-both-excel-and-csv-file-formats)
- [Outputs](#outputs)

---

## Overview
**Goal:** evaluation of provided images, using LLM and given categories, and storing results for further analysis.

**Algorithm**
1) Extraction and downscaling of pictures
2) Creating jobs and correspondent requests to LLM-Server
3) Conversion of the output into both Excel and CSV file formats

---

## Directory Layout

```
Ambuvision/
├─ Step1_VLM_Greeness/
  ├─ Imager.py              # Work with images and jobs
  ├─ Middle.py              # Middle point between host and server
  ├─ main.py                # Execution of all scripts combined
  ├─ exifRecognition.py     # Algorithm, used prior to the main assignment
  ├─ PyCharmMiscProject.iml
  ├─ README.md
```

---

## Prerequisites

- Python 3.13+
- A local or remote LLM endpoint (OpenAI-compatible)
- Storage volume depends on the information
- .env

---

## Quick start

```bash
python main.py
```

---

## Step-by-step

### Step 1: Extraction and downscaling of pictures
**Script:** `main.py`
**Input:** First and final set numbers
**Output:** No

This step defines the further workflow of the program.

---

### Step 2: Creating jobs and correspondent requests to LLM-Server
**Script:** `main.py`
**Input:** No
**Output:** No

This step is automatical and doesn't require interaction.

---

### Step 3: Conversion of the output into both Excel and CSV file formats
**Script:** `main.py`
**Input:** No
**Output:** `filename.csv`, `filename.xlsx`, `filename.json`, `token_report.json`, `run.log`

This step is automatical and also final, producing the sorted out results and logs.

---

## Outputs

- `.csv` -  image evaluation, step 3
- `.xlsx` - image evaluation, step 3
- `.json` - image metadata, step 3
- `token_report.json` - token number report for requests, step 3
- `run.log` - error report, step 3
