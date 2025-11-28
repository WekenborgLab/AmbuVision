# Ambuvision - Image Evaluation

A customizable programm to allow different levels of setting, chunking and job management


## Table of contents
- [Overview](#overview)
- [Directory Layout](#directory-layout)
- [Prerequisite](#prerequisites)
- [Quick start](#quick-start)
- [Step-by-step](#step-by-step)
  - [Step 1: Extraction and downscaling of pictures](#extraction-and-downscaling-of-pictures)
  - [Step 2: Creating jobs and correspondent requests to LLM-Server](#creating-jobs-and-correspondent-requests-to-llm-server)
  - [Step 3: Conversion of the output into both Excel and CSV file formats](#conversion-of-the-output-into-both-excel-and-csv-file-formats)
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
  ├─ Imager.py
  ├─ Middle.py
  ├─ main.py
  ├─ exifRecognition.py
  ├─ PyCharmMiscProject.iml
  ├─ README.md
```

---

## Prerequisites

- Python 3.10+
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
**Script:** main.py
**Input:** First and final set numbers
**Output:** No

---

### Step 2: Creating jobs and correspondent requests to LLM-Server
**Script:** main.py
**Input:** No
**Output:** No

---

### Step 3: Conversion of the output into both Excel and CSV file formats
**Script:** main.py
**Input:** No
**Output:** filename.csv, filename.xlsx, filename.json

---

## Outputs

- `.csv` -  image evaluation, step 3
- `.xlsx` - image evaluation, step 3
- `.json` - image metadata, step 3
