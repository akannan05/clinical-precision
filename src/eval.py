import json
import torch
from tqdm import tqdm
from transformers import AutoModelForCausalLM, AutoTokenizer
from peft import PeftModel
from datasets import load_dataset

# Configuration
BASE_MODEL = "Qwen/Qwen2.5-3B-Instruct"
GOLD_PATH = "./models/qwen-3b-clinical-gold"
RAW_PATH = "./models/qwen-3b-clinical-raw"
TEST_SIZE = 50 

def get_model(adapter_path):
    tokenizer = AutoTokenizer.from_pretrained(BASE_MODEL)
    model = AutoModelForCausalLM.from_pretrained(
        BASE_MODEL,
        torch_dtype=torch.bfloat16,
        device_map="auto"
    )
    model = PeftModel.from_pretrained(model, adapter_path)
    return model, tokenizer

def generate_answer(model, tokenizer, question):
    prompt = f"<|im_start|>system\nYou are a clinical reasoning assistant.<|im_end|>\n<|im_start|>user\n{question}<|im_end|>\n<|im_start|>assistant\n"
    inputs = tokenizer(prompt, return_tensors="pt").to("cuda")
    outputs = model.generate(**inputs, max_new_tokens=512, temperature=0.1)
    return tokenizer.decode(outputs[0], skip_special_tokens=True).split("assistant\n")[-1]

def run_eval():
    print("Loading test dataset...")
    test_ds = load_dataset("GBaker/MedQA-USMLE-4-options", split="test").select(range(TEST_SIZE))

    results = []

    print("Evaluating GOLD model...")
    model, tokenizer = get_model(GOLD_PATH)
    for row in tqdm(test_ds):
        ans = generate_answer(model, tokenizer, row['question'])
        results.append({"question": row['question'], "gold_response": ans, "ground_truth": row['answer_idx']})

    # Clean VRAM for the next model
    del model
    torch.cuda.empty_cache()

    print("Evaluating RAW model...")
    model, tokenizer = get_model(RAW_PATH)
    for i, row in enumerate(tqdm(test_ds)):
        ans = generate_answer(model, tokenizer, row['question'])
        results[i]["raw_response"] = ans

    with open("results/eval_comparison.json", "w") as f:
        json.dump(results, f, indent=2)
    print("Inference complete. Results saved to results/eval_comparison.json")

if __name__ == "__main__":
    run_eval()
