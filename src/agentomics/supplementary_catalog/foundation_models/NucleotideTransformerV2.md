# Nucleotide Transformer v2 Usage Guide

## Overview

**Nucleotide Transformer v2** is the second generation of InstaDeep's genomic
language-model family. The v2 architecture replaces learned positional
embeddings with rotary embeddings and introduces gated linear units. The models
were pretrained on DNA from 850 species using masked language modeling.

Most v2 checkpoints use 6-mer tokenization when possible. The dedicated
`50m-3mer` checkpoint instead uses finer 3-mer tokenization for experiments on
protein-related downstream tasks.

## Dependencies

Use the dependency versions from the last Agentomics environment that supported
these checkpoints:

```bash
pip install torch==2.9.0 transformers==4.47.1
```

The v2 checkpoints execute their bundled model implementation through
`trust_remote_code=True`. Agentomics previously exercised them with the pinned
Transformers 4 stack above; they were removed after the default environment
upgraded to Transformers 5.5.4.

## Model Variants

| Model | Parameters | Tokenization |
|-------|------------|--------------|
| `InstaDeepAI/nucleotide-transformer-v2-50m-3mer-multi-species` | 50M | 3-mer |
| `InstaDeepAI/nucleotide-transformer-v2-50m-multi-species` | 50M | 6-mer |
| `InstaDeepAI/nucleotide-transformer-v2-100m-multi-species` | 100M | 6-mer |
| `InstaDeepAI/nucleotide-transformer-v2-250m-multi-species` | 250M | 6-mer |
| `InstaDeepAI/nucleotide-transformer-v2-500m-multi-species` | 500M | 6-mer |

## Generating Sequence Embeddings

```python
import torch
from transformers import AutoModelForMaskedLM, AutoTokenizer

checkpoint = "InstaDeepAI/nucleotide-transformer-v2-50m-multi-species"
tokenizer = AutoTokenizer.from_pretrained(
    checkpoint,
    trust_remote_code=True,
)
model = AutoModelForMaskedLM.from_pretrained(
    checkpoint,
    trust_remote_code=True,
)
model.eval()

sequences = [
    "ATTCCGATTCCGATTCCG",
    "ATTTCTCTCTCTCTCTGAGATCGATCGATCGAT",
]
inputs = tokenizer(
    sequences,
    return_tensors="pt",
    padding=True,
)

with torch.no_grad():
    outputs = model(
        **inputs,
        encoder_attention_mask=inputs["attention_mask"],
        output_hidden_states=True,
    )

hidden_states = outputs.hidden_states[-1]
attention_mask = inputs["attention_mask"].unsqueeze(-1)
embeddings = (hidden_states * attention_mask).sum(dim=1) / attention_mask.sum(dim=1)
print(embeddings.shape)
```

The 50M checkpoint is the least resource-intensive option for initial testing.
For a downstream predictor, train and save the task-specific head together with
the pretrained model before inference.
