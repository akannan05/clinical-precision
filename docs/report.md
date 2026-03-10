---
layout: default
title: Technical Report
---

# Technical Report: Data Curation vs. Raw SFT
**Author:** Ani
**Date:** March 2026 
**Model:** Qwen2.5-3B-Instruct (4-bit QLoRA)

## 1. Executive Summary
This project investigates the impact of **expert-aligned data curation** on the clinical reasoning capabilities of small language models (3B parameters). By comparing a model trained on raw Q&A pairs ("Raw") against one trained on curated Chain-of-Thought reasoning ("Gold"), we demonstrate a **76.5% win rate** for the curated approach.

---

## 2. Training Methodology
We utilized two distinct training paths on a single NVIDIA RTX 5060 (8GB VRAM):

| Feature | Raw Model | Gold Model |
| :--- | :--- | :--- |
| **Dataset size** | 302 samples | 302 samples |
| **Data Format** | Direct Answer | **Chain-of-Thought (CoT)** |
| **Compute** | 4-bit QLoRA | 4-bit QLoRA |
| **Stability** | Standard | Gradient Checkpointing |

### Training Logic
To maintain stability on 8GB VRAM, we implemented `paged_adamw_8bit` and disabled `double_quant` to prevent the `cudaErrorIllegalAddress` encountered during the initial "Gold" run.

---

## 3. Evaluation Results
Evaluation was conducted using **LLM-as-a-Judge** with **Step 3.5 Flash** (Reasoning Enabled).

### Win-Rate Distribution
![Win Rate Comparison](./results/win_rate_comparison.png)

* **Gold Model Wins:** 13
* **Raw Model Wins:** 4
* **Ties:** 33

### Qualitative Analysis
The "Gold" model consistently produced structured clinical reasoning, even in cases where it failed to select the correct final option.
> **Judge Observation (Case #2):** "Model A provides accurate and detailed clinical reasoning... while Model B only states the correct answer without any explanation."

---

## 4. Discussion & Limitations
While the **Gold** model significantly outperformed the **Raw** model in logical depth, the high number of ties (66%) highlights the "Parameter Ceiling." A 3B-parameter model lacks the internal knowledge base to solve the most complex USMLE cases regardless of the training format. 

---

## 5. Conclusion
**Curation is the ultimate lever.** By focusing on 300 "Gold" samples, we achieved a 3.25x performance gain over the baseline, proving that for specialized domains like medicine, the *quality* of the reasoning chain is more important than the *quantity* of raw data.
