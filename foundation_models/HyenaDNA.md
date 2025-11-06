# HyenaDNA Usage Guide

## Overview

**HyenaDNA** is a long-range genomic foundation model pretrained on sequences up to **1 million tokens** at **single nucleotide resolution**. It uses the **Hyena operator**, a subquadratic alternative to attention in Transformers, enabling extremely long context modeling and efficient training.

## GPU Requirements (Suggested)

| Model           | Pretrain  | Fine-tune | Inference |
| --------------- | --------- | --------- | --------- |
| **tiny-1k**     | T4        | T4        | T4        |
| **small-32k**   | A100-40GB | T4        | T4        |
| **medium-160k** | A100-40GB | T4        | T4        |
| **medium-450k** | A100-40GB | A100-40GB | T4        |
| **large-1m**    | A100-80GB | A100-80GB | A100-40GB |

> **Note:** On a Colab T4 (16GB VRAM), HyenaDNA can train up to ~250k nucleotides. Longer sequences require more memory.

---

## Using HyenaDNA for Sequence Classification

Below is an example of fine-tuning **HyenaDNA (medium checkpoint)** on a sequence classification task.

```python
from transformers import AutoModelForSequenceClassification, AutoTokenizer
from transformers import TrainingArguments, Trainer, logging
import torch
from datasets import Dataset

# Load pretrained model
checkpoint = 'LongSafari/hyenadna-medium-160k-seqlen-hf'
max_length = 160_000

tokenizer = AutoTokenizer.from_pretrained(checkpoint, trust_remote_code=True)
model = AutoModelForSequenceClassification.from_pretrained(
    checkpoint, torch_dtype=torch.bfloat16, device_map="auto", trust_remote_code=True
)

# Generate dummy sequences and labels
sequence = 'ACTG' * int(max_length/4)
sequence = [sequence] * 8
labels = [0, 1] * 4
tokenized = tokenizer(sequence)["input_ids"]

# Create a dataset
ds = Dataset.from_dict({"input_ids": tokenized, "labels": labels})
ds.set_format("pt")

# Define training arguments
args = {
    "output_dir": "tmp",
    "num_train_epochs": 1,
    "per_device_train_batch_size": 1,
    "gradient_accumulation_steps": 4,
    "gradient_checkpointing": True,
    "learning_rate": 2e-5,
}
training_args = TrainingArguments(**args)

# Train model
trainer = Trainer(model=model, args=training_args, train_dataset=ds)
result = trainer.train()
print(result)
```

After training, you can save the model:

```python
model.save_pretrained("my_hyenadna_model")
```

The model uses a **single-character tokenizer** with a vocabulary of four nucleotides (A, C, G, T) plus special tokens, achieving true single-nucleotide resolution.