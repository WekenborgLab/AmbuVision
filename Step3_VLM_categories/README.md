# Step 3: Using 997 categories

## Project Structure

```text
Ambuvision/
├─ Step3_VLM_categories/
│  ├─ Imager.py                                                           # Work with images and jobs
│  ├─ Middle.py                                                           # Middle point between host and server
│  ├─ main.py                                                             # Execution of all scripts combined
│  ├─ allpsychologicalvariables_greater_equal_3_withoutdublicates.xlsx    # excel file with categories
│  ├─ categories_1.xlsx                                                   # first half of the excel file
│  ├─ categories_2.xlsx                                                   # second half of the excel file
│  ├─ pyproject.toml                                                      # Central configuration file containing project metadata,Python version requirements, dependencies, and build-system settings 
│  ├─ README.md
```

## main.py
### Summary
This script processes all images in a given folder and evaluates each one across a set of categories using an AI model. Categories can come from environment variables, a single Excel/CSV file, or a built-in fallback list. For each category, the script generates a prompt, runs model inference, and saves results to CSV/Excel.
#### Key features:
Loads settings from .env (paths, categories, concurrency, cost tracking, etc.)
Reads categories from env vars or Excel/CSV
Automatically builds evaluation jobs
Processes all images, optionally resizes/compresses them
Runs model calls in parallel and outputs logs, results, and token usage
#### Run:
python main.py
Requires proper .env configuration and (optionally) pandas/openpyxl for Excel support.
