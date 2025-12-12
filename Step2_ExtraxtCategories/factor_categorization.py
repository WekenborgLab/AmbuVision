import os
import asyncio
import pandas as pd
from openai import AsyncOpenAI

MODEL_NAME = os.environ.get("MODEL_NAME", "GPT-OSS-120B")
BASE_URL = os.environ.get("OPENAI_BASE_URL")
API_KEY = os.environ.get("VIRTUAL_API_KEY")

INPUT_CSV  = "artifacts/visual_factors/all_visual_associations.csv"
OUTPUT_FILE = "artifacts/visual_factors/llm_categorization.csv"

MAX_CONCURRENT = int(os.environ.get("MAX_CONCURRENT", "200"))

SYSTEM_PROMPT = (
    "You are a precise tagging assistant. Your job is to map short input phrases "
    "from psychology papers to a compact 1-2 word output that a vision-language model "
    "could detect in a smartphone photo.\n\n"
    "Rules:\n"
    "• Output must be 1 or 2 words, no spaces (CamelCase is OK).\n"
    "• The category should be visually detectable in a typical photo.\n"
    "• Be general and concise (avoid ultra-specific, study-unique terms).\n"
    "• There is no fixed list of outputs.\n\n"
    "Examples (not a closed set): Input: living alone → Output: LivingAlone, "
    "Input: natural scenery (sunrise, flowers, landscape) → Output: NatureView, "
    "Input: adding color to team spaces (colored walls or décor) → Output: ColoredWalls.\n"
)

PROMPT_TMPL = (
    "Return a compact, photo-detectable output for the input below. If it is not detectable in a photo return 'None'.\n\n"
    "Input: \"{factor}\"\n"
)

# ------------ Data prep (unchanged) ------------
df = pd.read_csv(INPUT_CSV)
n=len(df)
df["Category"] = ""
cols = df.columns.tolist()
# make "Category" the second column
cols.insert(1, cols.pop(cols.index("Category")))
df = df[cols]

# build (index, messages) pairs for the first n factors
rows = list(df.head(n).itertuples(index=True))
work = []
for row in rows:
    messages = [
        {"role": "system", "content": SYSTEM_PROMPT},
        {"role": "user", "content": PROMPT_TMPL.format(factor=row.factor)},
    ]
    work.append((row.Index, messages))

# ------------ Async client + semaphore ------------
async def get_response(client: AsyncOpenAI, sem: asyncio.Semaphore, messages: list, model: str):
    async with sem:
        resp = await client.chat.completions.create(
            model=model,
            messages=messages,
            temperature=0.2,
            max_completion_tokens=1024,
            reasoning_effort="medium", 
            extra_body={"allowed_openai_params": ["reasoning_effort"]},
        )
        result = (resp.choices[0].message.content or "").strip()
        print(f"✅ Task completed. Result: {result}")
        return result

async def main_async():
    client = AsyncOpenAI(base_url=BASE_URL, api_key=API_KEY)
    sem = asyncio.Semaphore(MAX_CONCURRENT)

    # fire off tasks in the same order as 'work'
    tasks = [asyncio.create_task(get_response(client, sem, msgs, MODEL_NAME)) for _, msgs in work]
    results = await asyncio.gather(*tasks, return_exceptions=False)

    # write results back to df using the saved indices
    for (idx, _), label in zip(work, results):
        df.at[idx, "Category"] = label

# run the async batch
asyncio.run(main_async())


df.to_csv(OUTPUT_FILE, index=False, encoding="utf-8")
