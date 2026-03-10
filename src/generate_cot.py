import json
import requests
import time
import os
from concurrent.futures import ThreadPoolExecutor, as_completed
from threading import Lock

from datasets import load_dataset
from huggingface_hub import login
from dotenv import load_dotenv

# 1. Setup Environment
load_dotenv()
OR_KEY = os.getenv("OPEN_ROUTER_KEY")
HF_TOKEN = os.getenv("HF_TOKEN")
MODEL_NAME = "arcee-ai/trinity-large-preview:free"
OUTPUT_FILE = "data/silver_cot/raw_reasoning.jsonl"

# Authenticate Hugging Face
if HF_TOKEN:
    login(token=HF_TOKEN)

# Thread-safe file writing
file_lock = Lock()

def format_prompt(row):
    """format medqa row into a clinical case prompt"""
    options = row['options']
    options_str = "\n".join([f"{k}: {v}" for k, v in options.items()])

    prompt = f"""You are a senior medical educator. Provide a detailed clinical reasoning path for the following case.

Case: {row['question']}
Options:
{options_str}

Requirement: Your final sentence must be: 'Therefore, the correct answer is [LETTER].'"""
    return prompt

def generate_reasoning(row, idx):
    """Calls Trinity API and captures both reasoning and final content."""
    prompt = format_prompt(row)

    headers = {
        "Authorization": f"Bearer {OR_KEY}",
        "Content-Type": "application/json",
        "HTTP-Referer": "https://github.com/your-username/clinical-precision", # Good practice for OpenRouter
    }

    payload = {
        "model": MODEL_NAME,
        "messages": [{"role": "user", "content": prompt}],
        "reasoning": {"enabled": True}
    }

    try:
        # Increased timeout to 120s because 4 parallel 400B requests can be slow
        response = requests.post(
            "https://openrouter.ai/api/v1/chat/completions",
            headers=headers,
            data=json.dumps(payload),
            timeout=120
        )
        response.raise_for_status()
        res_json = response.json()
        message = res_json['choices'][0]['message']

        result = {
            "meta_id": f"medqa_train_{idx}", # Stable unique ID
            "question": row['question'],
            "options": row['options'],
            "ground_truth": row['answer_idx'],
            "content": message.get('content'),
            "reasoning_details": message.get('reasoning_details'),
        }

        # Lock file for writing to prevent JSON corruption
        with file_lock:
            with open(OUTPUT_FILE, "a") as f:
                f.write(json.dumps(result) + "\n")

        return f"Success: {idx}"

    except Exception as e:
        return f"Error on {idx}: {str(e)}"

def main():
    # Ensure directory exists
    os.makedirs(os.path.dirname(OUTPUT_FILE), exist_ok=True)

    print("Loading dataset...")
    ds = load_dataset("GBaker/MedQA-USMLE-4-options", split="train")

    # Target subset
    subset_size = 3000
    subset = ds.select(range(subset_size))

    print(f"Starting parallel generation for {subset_size} samples using 4 threads...")

    start_time = time.time()

    # Use ThreadPoolExecutor for concurrent API calls
    with ThreadPoolExecutor(max_workers=8) as executor:
        # Mapping row and index to the generator function
        futures = [executor.submit(generate_reasoning, row, i) for i, row in enumerate(subset)]

        for i, future in enumerate(as_completed(futures)):
            status = future.result()
            elapsed = time.time() - start_time
            print(f"[{i+1}/{subset_size}] {status} | Elapsed: {elapsed:.2f}s", end="\r")

    print(f"\nFinished! Total time: {(time.time() - start_time)/60:.2f} minutes.")

if __name__ == "__main__":
    main()
