import json
from datasets import load_dataset

def main():
    print("Loading MedQA to create baseline...")
    ds = load_dataset("GBaker/MedQA-USMLE-4-options", split="train")

    # Grab the last 302 samples (far away from the first 3000 used for curation)
    total_samples = len(ds)
    baseline_samples = ds.select(range(total_samples - 302, total_samples))

    output_path = "data/processed/raw_train.jsonl"
    with open(output_path, "w") as f:
        for row in baseline_samples:
            # Baseline provides the answer but NO reasoning logic
            payload = {
                "question": row['question'],
                "content": f"Based on the clinical findings, the correct answer is {row['answer_idx']}."
            }
            f.write(json.dumps(payload) + "\n")

    print(f"✅ Created baseline with {len(baseline_samples)} samples at {output_path}")

if __name__ == "__main__":
    main()
