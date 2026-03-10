import json
import re
import os
import spacy
from collections import Counter

# Load the medical NER model (Make sure you ran: pip install en_core_sci_md)
print("Loading Medical NER (ScispaCy)...")
try:
    nlp = spacy.load("en_core_sci_md")
except:
    print("NER model not found. Run: pip install https://s3-us-west-2.amazonaws.com/ai2-s2-scispacy/releases/v0.5.4/en_core_sci_md-0.5.4.tar.gz")
    exit()

class CurationStats:
    def __init__(self):
        self.total = 0
        self.passed = 0
        self.rejections = Counter()

    def print_report(self):
        print("\n" + "="*40)
        print("🧬 CLINICAL DATA CURATION REPORT")
        print("="*40)
        print(f"Total Samples Processed: {self.total}")
        print(f"Total 'Gold' Samples:    {self.passed} (Yield: {self.passed/self.total*100:.1f}%)")
        print("-" * 40)
        print("REJECTION BREAKDOWN:")
        for reason, count in self.rejections.items():
            percentage = (count / self.total) * 100
            print(f" ❌ {reason:25} | {count:4} samples ({percentage:4.1f}%)")
        print("="*40)

def curate_sample(row, stats):
    stats.total += 1

    # --- 1. Correctness Gate ---
    # We look for Trinity's final answer and compare it to the MedQA ground truth
    content = row.get('content', '')
    match = re.search(r"Therefore, the correct answer is ([A-D])", content)

    if not match:
        stats.rejections['Format Error'] += 1
        return None

    predicted_letter = match.group(1)
    if predicted_letter != row['ground_truth']:
        stats.rejections['Wrong Answer'] += 1
        return None

    # --- 2. Clinical Entity Density ---
    # Reasoning is "shallow" if it doesn't mention enough medical terms
    doc = nlp(content)
    unique_entities = set([ent.text.lower() for ent in doc.ents])
    if len(unique_entities) < 5:
        stats.rejections['Low Clinical Density'] += 1
        return None

    # --- 3. Differential Reasoning Check ---
    # Senior-level logic must rule out distractors
    differential_keywords = ['incorrect', 'rule out', 'distinguish', 'unlike', 'whereas']
    if not any(word in content.lower() for word in differential_keywords):
        stats.rejections['No Differential Logic'] += 1
        return None

    # --- 4. Length / Depth Check ---
    # Brief answers are usually not helpful for fine-tuning reasoning
    word_count = len(content.split())
    if word_count < 150:
        stats.rejections['Too Brief'] += 1
        return None

    stats.passed += 1
    return row

def main():
    input_file = "data/silver_cot/raw_reasoning.jsonl"
    output_file = "data/processed/curated_reasoning.jsonl"
    os.makedirs("data/processed", exist_ok=True)

    stats = CurationStats()
    curated_list = []

    print(f"Opening {input_file}...")
    with open(input_file, 'r') as f:
        for line in f:
            row = json.loads(line)
            clean_row = curate_sample(row, stats)
            if clean_row:
                curated_list.append(clean_row)

    # Save the Gold Set
    with open(output_file, 'w') as f:
        for item in curated_list:
            f.write(json.dumps(item) + "\n")

    stats.print_report()
    print(f"Saved {len(curated_list)} gold samples to {output_file}")

if __name__ == "__main__":
    main()
