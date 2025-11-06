# DNABERT-2 Usage Guide

## Overview

**DNABERT-2** is a transformer-based genome foundation model designed for understanding DNA sequences across multiple species. It builds upon the **MosaicBERT** architecture and is available on Hugging Face under `zhihan1996/DNABERT-2-117M`.

## Installation

Before using the model, ensure the following dependencies are installed:

```bash
pip install torch transformers
```

## Loading the Model

```python
import torch
from transformers import AutoTokenizer, AutoModel

tokenizer = AutoTokenizer.from_pretrained(
    "zhihan1996/DNABERT-2-117M", trust_remote_code=True
)
model = AutoModel.from_pretrained(
    "zhihan1996/DNABERT-2-117M", trust_remote_code=True
)
```

This loads both the tokenizer (for encoding DNA sequences) and the model (for generating embeddings).

## Generating DNA Embeddings

Example DNA sequence:

```python
dna = "ACGTAGCATCGGATCTATCTATCGACACTTGGTTATCGATCTACGAGCATCTCGTTAGC"
inputs = tokenizer(dna, return_tensors="pt")["input_ids"]
hidden_states = model(inputs)[0]  # shape: [1, sequence_length, 768]
```

## Pooling Methods

To obtain a single fixed-length vector representation (embedding) from the token-level outputs:

**Mean pooling:**

```python
embedding_mean = torch.mean(hidden_states[0], dim=0)
```

**Max pooling:**

```python
embedding_max = torch.max(hidden_states[0], dim=0)[0]
```

Both methods yield a **768-dimensional embedding** per sequence.

## Usage Tips

* **Mean pooling** captures overall contextual information, while **max pooling** highlights the most activated features.
* Embeddings can be used for downstream tasks like **classification**, **clustering**, or **similarity analysis**.
* For fine-tuning, you can add a task-specific head using the `transformers` library.