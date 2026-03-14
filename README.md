# Fine-Tuning Jais-6.7B for Arabic Text Generation — Ibn Battuta Style

Fine-tuning the **Jais-family-6.7B-chat** Arabic LLM to generate responses in the classical literary style of **Ibn Battuta**, the 14th-century Moroccan explorer. The project uses **QLoRA** (4-bit quantization + LoRA adapters) and conducts a series of controlled experiments to compare different fine-tuning strategies on a small custom Arabic dataset.

---

## Overview

The model is prompted to respond as Ibn Battuta — answering questions about his travels, observations, and experiences in his distinctive classical Arabic voice. The project explores how effectively a small, domain-specific dataset (29 examples) can shift an Arabic LLM's style through parameter-efficient fine-tuning.

**Base Model:** [`inceptionai/jais-family-6p7b-chat`](https://huggingface.co/inceptionai/jais-family-6p7b-chat)  
**Dataset:** Custom JSONL dataset — 29 Ibn Battuta-style Arabic Q&A examples  
**Hardware:** Google Colab — Tesla T4 GPU (15.6 GB VRAM)

---

## Experiments

Five experiments are run and compared across perplexity and a custom Arabic style score:

| Experiment | Description | Trainable Params |
|---|---|---|
| **A: Few-shot baseline** | No fine-tuning; 2-shot prompting only | — |
| **B: Full LoRA** | LoRA on all linear layers (all 29 examples) | ~35.7M (0.99%) |
| **C1: Top-half LoRA** | LoRA on all layers; lower 16 layers frozen | ~17.8M (0.49%) |
| **C2: Top-quarter LoRA** | LoRA on all layers; lower 24 layers frozen | ~8.9M (0.25%) |
| **D: Attention-only LoRA** | LoRA on attention modules only | ~8.9M (0.25%) |
| **E: Few-shot fine-tuning** | LoRA with k=3, k=5, k=10 training examples | varies |

---

## Evaluation Metrics

Each fine-tuned model is evaluated on:

- **Perplexity (PPL)** — computed on a fixed reference Arabic passage from Ibn Battuta's writings
- **Arabic Style Score** — fraction of classical Ibn Battuta style markers present in generated responses (e.g., travel verbs, religious expressions, descriptive adjectives)

Five Arabic evaluation questions are used for generation:
- *"Describe the markets of Cairo in your era."*
- *"What was the Sultan's court like in Delhi?"*
- *"Describe the sea voyage from India to China."*
- *"What advice would you give a young traveler?"*
- *"Compare the generosity of India with that of the people of Crimea."*

---

## Setup

### Requirements

```bash
pip install transformers==4.44.0 datasets==2.20.0 peft==0.12.0 trl==0.9.6 \
            bitsandbytes>=0.44.0 accelerate==0.33.0 torch huggingface_hub \
            pandas==2.2.2 matplotlib numpy==1.26.4
```

### Hugging Face Authentication

```python
from huggingface_hub import login
login()  # Required to access the Jais model
```

### Dataset

Place the `ibn_battuta_arabic_dataset.jsonl` file in the working directory. Each line should follow the format:

```json
{"text": "<Ibn Battuta style Q&A in Arabic>"}
```

---

## Training Configuration

| Parameter | Value |
|---|---|
| Epochs | 3 |
| Batch size | 1 (+ grad accumulation × 4) |
| Learning rate | 2e-4 |
| LR scheduler | Cosine |
| Max sequence length | 512 |
| Warmup ratio | 0.05 |
| Quantization | 4-bit NF4 (double quant, bfloat16) |
| LoRA rank (r) | 16 |
| LoRA alpha | 32 |
| LoRA dropout | 0.05 |

---

## Project Structure

```
├── ibn_battuta_arabic_finetuning.ipynb   # Main notebook: all experiments
├── ibn_battuta_arabic_dataset.jsonl      # Training dataset (Arabic Q&A)
├── results/                              # Training outputs per experiment
└── README.md
```

---

## Technologies Used

- **[Jais-family-6.7B-chat](https://huggingface.co/inceptionai/jais-family-6p7b-chat)** — Arabic-first LLM by Inception AI
- **PEFT / LoRA** — Parameter-efficient fine-tuning
- **BitsAndBytes** — 4-bit QLoRA quantization
- **TRL / SFTTrainer** — Supervised fine-tuning
- **Hugging Face Transformers & Datasets**
- **PyTorch**
- **Google Colab** (Tesla T4 GPU)
