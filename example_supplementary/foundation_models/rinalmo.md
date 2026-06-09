# RiNALMo

## Overview

**RiNALMo** (RiboNucleic Acid Language Model) is a BERT-style transformer trained on large-scale non-coding RNA (ncRNA) sequences using a **masked language modeling (MLM)** objective. It generalizes well to RNA structure prediction and related tasks.

RiNALMo learns from raw RNA sequences without supervision and provides pretrained checkpoints verified to match the official implementation.

## Model Variants

| Variant           | Layers | Hidden Size | Heads | Intermediate | Params (M) | Max Tokens |
| ----------------- | -----: | ----------: | ----: | -----------: | ---------: | ---------: |
| **rinalmo-micro** |     12 |         480 |    20 |         1920 |       33.5 |       1022 |
| **rinalmo-mega**  |     30 |         640 |  2560 |          148 |       1022 |            |
| **rinalmo-giga**  |     33 |        1280 |  5120 |          650 |       1022 |            |

## Quickstart: Masked Language Modeling

```python
import torch
import multimolecule  # Registers RiNALMo in Hugging Face transformers
from transformers import pipeline

device = 0 if torch.cuda.is_available() else -1
print("Using GPU" if device == 0 else "Using CPU")

unmasker = pipeline("fill-mask", model="multimolecule/rinalmo-micro", device=device)

res = unmasker("gguc<mask>cucugguuagaccagaucugagccu")
for r in res:
    print(r)
```

**Example output:**

```python
{'score': 0.34, 'token_str': 'A', 'sequence': 'GGUCACUCUGGUUAGACCAGAUCUGAGCCU'}
```

---

## Get Sequence Embeddings

```python
import torch
from multimolecule import RnaTokenizer, RiNALMoModel

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
print(f"Using device: {device}")

# Load model and tokenizer
tokenizer = RnaTokenizer.from_pretrained("multimolecule/rinalmo-micro")
model = RiNALMoModel.from_pretrained("multimolecule/rinalmo-micro").to(device)

rna = "UAGCUUAUCAGACUGAUGUUG"
inputs = tokenizer(rna, return_tensors="pt").to(device)
outputs = model(**inputs)

# Hidden state embeddings
embeddings = outputs.last_hidden_state  # [batch, seq_len, hidden]

# Mean pooling for sequence-level representation
embedding_mean = embeddings.mean(dim=1).cpu()
print(embedding_mean.shape)
```

---

## Fine-Tuning Examples

### Sequence Classification / Regression

```python
import torch
from multimolecule import RnaTokenizer, RiNALMoForSequencePrediction

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
print(f"Using device: {device}")

tokenizer = RnaTokenizer.from_pretrained("multimolecule/rinalmo-micro")
model = RiNALMoForSequencePrediction.from_pretrained("multimolecule/rinalmo-micro").to(device)

text = "UAGCUUAUCAGACUGAUGUUG"
inputs = tokenizer(text, return_tensors="pt").to(device)
label = torch.tensor([1]).to(device)

outputs = model(**inputs, labels=label)
loss = outputs.loss
print(loss)
```

### Token Classification / Regression

```python
import torch
from multimolecule import RnaTokenizer, RiNALMoForTokenPrediction

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
print(f"Using device: {device}")

tokenizer = RnaTokenizer.from_pretrained("multimolecule/rinalmo-micro")
model = RiNALMoForTokenPrediction.from_pretrained("multimolecule/rinalmo-micro").to(device)

rna = "UAGCUUAUCAGACUGAUGUUG"
inputs = tokenizer(rna, return_tensors="pt").to(device)
labels = torch.randint(2, (len(rna),)).to(device)

outputs = model(**inputs, labels=labels)
print(outputs.loss)
```

### Contact Classification / Regression

```python
import torch
from multimolecule import RnaTokenizer, RiNALMoForContactPrediction

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
print(f"Using device: {device}")

tokenizer = RnaTokenizer.from_pretrained("multimolecule/rinalmo-micro")
model = RiNALMoForContactPrediction.from_pretrained("multimolecule/rinalmo-micro").to(device)

rna = "UAGCUUAUCAGACUGAUGUUG"
inputs = tokenizer(rna, return_tensors="pt").to(device)
labels = torch.randint(2, (len(rna), len(rna))).to(device)

outputs = model(**inputs, labels=labels)
print(outputs.loss)
```
