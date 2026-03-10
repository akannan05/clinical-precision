import os
import torch
import wandb
from datasets import load_dataset
from transformers import (
    AutoModelForCausalLM,
    AutoTokenizer,
    BitsAndBytesConfig,
    TrainingArguments,
)
from peft import LoraConfig, get_peft_model
from trl import SFTTrainer, SFTConfig
from dotenv import load_dotenv

load_dotenv()

# --- CONFIGURATION ---
MODEL_ID = "Qwen/Qwen2.5-3B-Instruct"
# TOGGLE THESE FOR YOUR TWO RUNS:
# 1. Run Gold: DATA_PATH="data/processed/curated_reasoning.jsonl", RUN_NAME="qwen-3b-gold"
# 2. Run Raw:  DATA_PATH="data/processed/raw_train.jsonl", RUN_NAME="qwen-3b-raw"
DATA_PATH = "data/processed/raw_train.jsonl"
RUN_NAME = "qwen-3b-clinical-raw"
OUTPUT_DIR = f"./models/{RUN_NAME}"

def train():
    # 1. Initialize WandB
    wandb.init(
        project="clinical-precision-3b",
        name=RUN_NAME,
        config={
            "model": MODEL_ID,
            "dataset": DATA_PATH,
            "learning_rate": 2e-4,
            "epochs": 3,
            "batch_size": 8 # Effective (1 bs * 8 grad_acc)
        }
    )

    # 2. BitsAndBytes Config (Crucial for 8GB VRAM)
    bnb_config = BitsAndBytesConfig(
        load_in_4bit=True,
        bnb_4bit_quant_type="nf4",
        bnb_4bit_compute_dtype=torch.bfloat16,
        bnb_4bit_use_double_quant=False,
    )

    # 3. Load Model & Tokenizer
    tokenizer = AutoTokenizer.from_pretrained(MODEL_ID)
    tokenizer.pad_token = tokenizer.eos_token
    tokenizer.padding_side = "right"

    model = AutoModelForCausalLM.from_pretrained(
        MODEL_ID,
        quantization_config=bnb_config,
        device_map="auto",
        torch_dtype=torch.bfloat16,
        trust_remote_code=True
    )

    # 4. LoRA Config
    peft_config = LoraConfig(
        r=16,
        lora_alpha=32,
        target_modules=["q_proj", "k_proj", "v_proj", "o_proj", "gate_proj", "up_proj", "down_proj"],
        lora_dropout=0.05,
        bias="none",
        task_type="CAUSAL_LM",
    )

    # 5. Formatting for Qwen Chat Template
    def format_prompt(sample):
        # Using Qwen's ChatML-style format
        text = f"<|im_start|>system\nYou are a clinical reasoning assistant.<|im_end|>\n"
        text += f"<|im_start|>user\n{sample['question']}<|im_end|>\n"
        text += f"<|im_start|>assistant\n{sample['content']}<|im_end|>"
        return {"text": text}

    dataset = load_dataset("json", data_files=DATA_PATH, split="train")
    dataset = dataset.map(format_prompt)

    # 6. Training Configuration (Using the exact fields from your class definition)
    sft_config = SFTConfig(
        output_dir=OUTPUT_DIR,
        max_length=768,                # Changed from max_seq_length to max_length
        dataset_text_field="text",       # Required
        per_device_train_batch_size=1,
        gradient_accumulation_steps=8,
        learning_rate=1e-4,
        lr_scheduler_type="cosine",
        num_train_epochs=3,
        logging_steps=1,
        save_strategy="epoch",
        bf16=True,
        optim="paged_adamw_8bit",
        report_to="wandb",
        remove_unused_columns=True,
        gradient_checkpointing=True,
    )

    # 7. Start Training
    trainer = SFTTrainer(
        model=model,
        train_dataset=dataset,
        args=sft_config,
        peft_config=peft_config,
    )

    trainer.train()
    trainer.save_model(OUTPUT_DIR)
    print(f"Model saved to {OUTPUT_DIR}")
    wandb.finish()

if __name__ == "__main__":
    train()
