# ESM-2

## Overview

* **ESM-2**: Transformer protein language models trained with masked language modeling (MLM) on large protein corpora.
* **What they’re good at**: structure/function prediction, mutational effect modeling, per-residue labeling, and fast folding with **ESMFold** (no MSA at inference).
* **ESMFold**: Uses an ESM-2 backbone plus a folding head; much faster than MSA-based pipelines while maintaining strong accuracy.

### Checkpoints

Larger models are generally more accurate but require more memory.

| Checkpoint            | Layers | Parameters |
| --------------------- | -----: | ---------: |
| `esm2_t48_15B_UR50D`  |     48 |        15B |
| `esm2_t36_3B_UR50D`   |     36 |         3B |
| `esm2_t33_650M_UR50D` |     33 |       650M |
| `esm2_t30_150M_UR50D` |     30 |       150M |
| `esm2_t12_35M_UR50D`  |     12 |        35M |
| `esm2_t6_8M_UR50D`    |      6 |         8M |

---

## Installation

```bash
pip install torch transformers
# For ESMFold extras
pip install openfold biopython
```

---

## Get Sequence Embeddings (mean-pooled, special tokens removed)

```python
import torch
from transformers import AutoTokenizer, AutoModel

model_id = "facebook/esm2_t33_650M_UR50D"  # choose a size from the table

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
print(f"Using device: {device}")

# Load tokenizer & base model (no task head)
tokenizer = AutoTokenizer.from_pretrained(model_id)
model = AutoModel.from_pretrained(model_id).to(device)
model.eval()

# Example sequences (one-letter AA codes)
seqs = [
    "MKTFFVAGVILLLATATAMA",
    "GSSGSSGSSGSSGSSGSSGS"
]

# Tokenize (adds <cls>/<eos> as needed)
enc = tokenizer(seqs, return_tensors="pt", padding=True, truncation=True).to(device)

with torch.no_grad():
    out = model(**enc, output_hidden_states=True)

# Last hidden layer: [batch, seq_len, hidden]
hidden = out.last_hidden_state  # or out.hidden_states[-1]

# Build a mask to exclude <cls>, <eos>, and <pad>
ids = enc.input_ids
mask = enc.attention_mask.bool()
for special_id in [tokenizer.cls_token_id, tokenizer.eos_token_id, tokenizer.pad_token_id]:
    if special_id is not None:
        mask = mask & (ids != special_id)

# Mean-pool valid residues per sequence
embeds = []
for i in range(hidden.size(0)):
    h = hidden[i][mask[i]]  # [valid_len, hidden]
    embeds.append(h.mean(dim=0))  # [hidden]
embeds = torch.stack(embeds).cpu()  # [batch, hidden]
print(embeds.shape)  # e.g., torch.Size([2, 1280]) for the 650M model
```

**Notes**

* Input should be uppercase amino acids (A–Z; unknowns often `X`).
* Respect each checkpoint’s `max_position_embeddings` (~1k context). For longer proteins, chunk or crop.
* For supervised tasks: use `EsmForSequenceClassification` or `EsmForTokenClassification` and pass `labels`.
* Mixed precision and gradient checkpointing help with larger checkpoints.

---

## ESMFold Quickstart (no MSA)

```python
from transformers import AutoTokenizer, EsmForProteinFolding
import torch

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
print(f"Using device: {device}")

model = EsmForProteinFolding.from_pretrained("facebook/esmfold_v1").to(device)
tokenizer = AutoTokenizer.from_pretrained("facebook/esmfold_v1")

seq = "MLKNVQVQLV"
inputs = tokenizer([seq], return_tensors="pt", add_special_tokens=False).to(device)
with torch.no_grad():
    outputs = model(**inputs)
positions = outputs.positions   # atom positions
plddt = outputs.plddt           # per-residue confidence
```
