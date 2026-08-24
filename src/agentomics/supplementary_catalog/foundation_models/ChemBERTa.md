# ChemBERTa

## Overview

ChemBERTa is a family of BERT-based models for chemical molecular analysis and property prediction. These models process SMILES representations using transformer architectures and come in two training variants:

- **MLM** (Masked Language Modeling): Pretrained to predict masked tokens in SMILES strings.
- **MTR** (Masked Token Regression): Pretrained using regression on masked tokens, specifically designed for molecular property prediction.

## Model Variants

### MLM Models

| Model | Parameters | Training Data |
|-------|-----------|---------------|
| `DeepChem/ChemBERTa-100M-MLM` | ~100M tokens | 100M molecules from ZINC20 |
| `DeepChem/ChemBERTa-77M-MLM` | 77M | ZINC subset |
| `DeepChem/ChemBERTa-10M-MLM` | 10M | ZINC subset |
| `DeepChem/ChemBERTa-5M-MLM` | 5M | ZINC subset |

### MTR Models

| Model | Parameters |
|-------|-----------|
| `DeepChem/ChemBERTa-77M-MTR` | 77M |
| `DeepChem/ChemBERTa-10M-MTR` | 10M |
| `DeepChem/ChemBERTa-5M-MTR` | 5M |

## Usage

### Masked Language Modeling

```python
from transformers import pipeline

pipe = pipeline("fill-mask", model="DeepChem/ChemBERTa-77M-MLM")
```

### Loading for Feature Extraction (MLM models)

```python
from transformers import AutoTokenizer, AutoModelForMaskedLM

tokenizer = AutoTokenizer.from_pretrained("DeepChem/ChemBERTa-77M-MLM")
model = AutoModelForMaskedLM.from_pretrained("DeepChem/ChemBERTa-77M-MLM")
```

### Loading for Feature Extraction (MTR models)

```python
from transformers import AutoTokenizer, AutoModel

tokenizer = AutoTokenizer.from_pretrained("DeepChem/ChemBERTa-77M-MTR")
model = AutoModel.from_pretrained("DeepChem/ChemBERTa-77M-MTR")
```
