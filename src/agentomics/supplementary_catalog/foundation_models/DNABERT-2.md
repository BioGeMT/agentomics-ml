# DNABERT-2 Usage Guide

## Overview

**DNABERT-2** is a transformer model pretrained on genomic sequences from
multiple species. It uses a MosaicBERT-derived architecture and byte-pair
encoding rather than fixed-length nucleotide k-mers. The pretrained checkpoint
is available as `zhihan1996/DNABERT-2-117M` on Hugging Face.

## Dependencies

The upstream implementation targets Python 3.8 and pins these packages:

```bash
pip install \
  torch==1.13.1 \
  transformers==4.29.2 \
  einops==0.6.1 \
  peft==0.3.0 \
  omegaconf==2.3.0 \
  evaluate==0.4.0 \
  accelerate==0.20.3 \
  scikit-learn==1.2.2
```

## Loading the Model

```python
import torch
from transformers import AutoModel, AutoTokenizer
from transformers.models.bert.configuration_bert import BertConfig

checkpoint = "zhihan1996/DNABERT-2-117M"
tokenizer = AutoTokenizer.from_pretrained(
    checkpoint,
    trust_remote_code=True,
)
config = BertConfig.from_pretrained(checkpoint)
model = AutoModel.from_pretrained(
    checkpoint,
    config=config,
    trust_remote_code=True,
)
model.eval()
```

The first call downloads the checkpoint because model weights are not bundled
with Agentomics.

## Generating Sequence Embeddings

```python
dna = "ACGTAGCATCGGATCTATCTATCGACACTTGGTTATCGATCTACGAGCATCTCGTTAGC"
inputs = tokenizer(dna, return_tensors="pt")

with torch.no_grad():
    hidden_states = model(**inputs)[0]

attention_mask = inputs["attention_mask"].unsqueeze(-1)
embedding = (hidden_states * attention_mask).sum(dim=1) / attention_mask.sum(dim=1)
print(embedding.shape)  # [batch_size, 768]
```

Mask-aware mean pooling prevents padding tokens from affecting embeddings when
sequences are processed in batches. These embeddings can be used as inputs to a
downstream classifier or regressor. If a task-specific head is added, train it
and save the complete fitted model before inference; an untrained head produces
meaningless predictions.
