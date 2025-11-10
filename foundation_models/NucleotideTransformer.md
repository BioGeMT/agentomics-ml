# Nucleotide Transformer v2

## Overview

**Nucleotide Transformer v2** is a large-scale foundation language model developed by **InstaDeep**, **NVIDIA**, and **TUM** for genomic sequence understanding. It is part of the *Nucleotide Transformer* collection, pre-trained on over **850 genomes** from diverse species — both model and non-model organisms — encompassing **174 billion nucleotides (~29B tokens)**.

Part of this collection is the nucleotide-transformer-500m-human-ref, a 500M parameters transformer pre-trained on the human reference genome. 

Part of this collection is the nucleotide-transformer-v2-50m-3mer-multi-species, a 50M parameters transformer pre-trained on a collection of 850 genomes from a wide range of species, including model and non-model organisms.

Unlike previous models trained on single reference genomes, this collection integrates information across thousands of genomes, providing highly accurate molecular phenotype predictions.

## Model Variants

| Model                                                            | Type      | Updated      | Parameters | Downloads | Likes |
| ---------------------------------------------------------------- | --------- | ------------ | ---------- | --------- | ----- |
| **InstaDeepAI/nucleotide-transformer-2.5b-multi-species**        | Fill-Mask | Jul 22, 2024 | 2.5B       | 7.13k     | 41    |
| **InstaDeepAI/nucleotide-transformer-2.5b-1000g**                | Fill-Mask | Jul 22, 2024 | 2.5B       | 184       | 8     |
| **InstaDeepAI/nucleotide-transformer-500m-human-ref**            | Fill-Mask | Jul 22, 2024 | 500M       | 869k      | 14    |
| **InstaDeepAI/nucleotide-transformer-500m-1000g**                | Fill-Mask | Jul 22, 2024 | 500M       | 1.63k     | 7     |
| **InstaDeepAI/nucleotide-transformer-v2-50m-multi-species**      | Fill-Mask | Sep 16, 2024 | 55.9M      | 21k       | 5     |
| **InstaDeepAI/nucleotide-transformer-v2-50m-3mer-multi-species** | Fill-Mask | Jun 4, 2024  | —          | 676       | 3     |
| **InstaDeepAI/nucleotide-transformer-v2-100m-multi-species**     | Fill-Mask | Sep 16, 2024 | 97.9M      | 2.31k     | 1     |
| **InstaDeepAI/nucleotide-transformer-v2-250m-multi-species**     | Fill-Mask | Sep 16, 2024 | 250M       | 6.68k     | 3     |
| **InstaDeepAI/nucleotide-transformer-v2-500m-multi-species**     | Fill-Mask | Oct 2024     | 0.5B       | 50.9k     | 27    |

## Example Usage

Retrieve logits and embeddings from a dummy DNA sequence:

```python
from transformers import AutoTokenizer, AutoModelForMaskedLM
import torch

# Load tokenizer and model
tokenizer = AutoTokenizer.from_pretrained(
    "InstaDeepAI/nucleotide-transformer-v2-500m-multi-species", trust_remote_code=True
)
model = AutoModelForMaskedLM.from_pretrained(
    "InstaDeepAI/nucleotide-transformer-v2-500m-multi-species", trust_remote_code=True
)

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
print(f"Using device: {device}")
model = model.to(device)

# Define sequence length and tokenize input
max_length = tokenizer.model_max_length
sequences = [
    "ATTCCGATTCCGATTCCG",
    "ATTTCTCTCTCTCTCTGAGATCGATCGATCGAT"
]

tokens_ids = tokenizer.batch_encode_plus(
    sequences, return_tensors="pt", padding="max_length", max_length=max_length
)["input_ids"].to(device)

# Compute embeddings
attention_mask = (tokens_ids != tokenizer.pad_token_id).to(device)
torch_outs = model(
    tokens_ids,
    attention_mask=attention_mask,
    encoder_attention_mask=attention_mask,
    output_hidden_states=True
)

embeddings = torch_outs['hidden_states'][-1]
print(f"Embeddings shape: {embeddings.shape}")
print(f"Embeddings per token: {embeddings}")

# Compute mean sequence embeddings
attention_mask = torch.unsqueeze(attention_mask, dim=-1)
mean_sequence_embeddings = torch.sum(attention_mask * embeddings, dim=-2) / torch.sum(attention_mask, dim=1)
print(f"Mean sequence embeddings: {mean_sequence_embeddings.cpu().shape}")
```