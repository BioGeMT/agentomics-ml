# MoLFormer-XL Usage Guide

## Overview

**MoLFormer-XL** is a chemical language model pretrained on SMILES strings using
masked language modeling. Its architecture combines linear attention with rotary
embeddings. The `ibm/MoLFormer-XL-both-10pct` checkpoint was trained on 10% of
the ZINC and PubChem pretraining datasets.

## Dependencies

The current Hugging Face implementation is tested with these packages:

```bash
pip install torch transformers==5.12.1
```

For an environment that requires Transformers 4, load the model and tokenizer
with `revision="compat-v4"` instead of using the default revision.

## Intended Use and Limitations

MoLFormer-XL can produce fixed molecular representations for downstream
classification, regression, similarity analysis, or visualization. It can also
be fine-tuned for molecular property prediction. It is not intended for molecule
generation and was not tested on molecules larger than approximately 200 atoms.
Invalid or noncanonical SMILES may degrade the resulting representations.

## Generating Molecular Embeddings

```python
import torch
from transformers import AutoModel, AutoTokenizer

checkpoint = "ibm/MoLFormer-XL-both-10pct"
tokenizer = AutoTokenizer.from_pretrained(
    checkpoint,
    trust_remote_code=True,
)
model = AutoModel.from_pretrained(
    checkpoint,
    deterministic_eval=True,
    trust_remote_code=True,
)
model.eval()

smiles = [
    "Cn1c(=O)c2c(ncn2C)n(C)c1=O",
    "CC(=O)Oc1ccccc1C(=O)O",
]
inputs = tokenizer(smiles, padding=True, return_tensors="pt")

with torch.no_grad():
    embeddings = model(**inputs).pooler_output

print(embeddings.shape)  # [batch_size, 768]
```

If a task-specific prediction head is added, train it and save the complete
fitted model before inference; an untrained head produces meaningless
predictions.
