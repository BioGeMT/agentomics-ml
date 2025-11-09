
# ChemBERTa-100M-MLM

ChemBERTa model pretrained on a subset of 100M molecules from ZINC20 dataset using masked language modeling (MLM).

# ChemBERTa-77M-MLM
| Property | Value |
|---|---|
| Model Type | Masked Language Model (MLM) |
| Developer | DeepChem |
| Parameters | 77 Million |
| Model URL | [DeepChem/ChemBERTa-77M-MLM — Hugging Face](https://huggingface.co/DeepChem/ChemBERTa-77M-MLM) |

ChemBERTa-77M-MLM is a BERT-based model for chemical molecular analysis and property prediction that processes SMILES representations using transformer architectures.

## Implementation Details
The model implements a masked language modeling approach with 77 million parameters, specifically adapted for chemical structure understanding. It builds upon the BERT architecture while incorporating chemical-specific modifications to handle SMILES notation effectively.

Pre-trained on extensive chemical compound datasets
Optimized for SMILES representation processing
Implements masked language modeling for chemical structure prediction
Utilizes transformer-based architecture

## Core Capabilities
Molecular property prediction
Chemical structure analysis
SMILES sequence understanding and generation
Chemical similarity assessment
Structure-property relationship modeling

## Frequently Asked Questions
Q: What makes this model unique?

ChemBERTa-77M-MLM stands out for its specialized focus on chemical structures and its ability to process SMILES notation effectively, making it particularly valuable for pharmaceutical research and chemical property prediction tasks.

Q: What are the recommended use cases?

The model is ideal for drug discovery pipelines, chemical property prediction, molecular optimization, and other chemical informatics applications where understanding molecular structure and properties is crucial.

## Usage
```python
# Use a pipeline as a high-level helper
from transformers import pipeline

pipe = pipeline("fill-mask", model="DeepChem/ChemBERTa-77M-MLM")

# Load model directly
from transformers import AutoTokenizer, AutoModelForMaskedLM

tokenizer = AutoTokenizer.from_pretrained("DeepChem/ChemBERTa-77M-MLM")
model = AutoModelForMaskedLM.from_pretrained("DeepChem/ChemBERTa-77M-MLM")
```
# ChemBERTa-10M-MLM
## Usage
```python
# Use a pipeline as a high-level helper
from transformers import pipeline

pipe = pipeline("fill-mask", model="DeepChem/ChemBERTa-10M-MLM")

# Load model directly
from transformers import AutoTokenizer, AutoModelForMaskedLM

tokenizer = AutoTokenizer.from_pretrained("DeepChem/ChemBERTa-10M-MLM")
model = AutoModelForMaskedLM.from_pretrained("DeepChem/ChemBERTa-10M-MLM")
```
# ChemBERTa-5M-MLM
## Usage
```python
from transformers import AutoTokenizer, AutoModelForMaskedLM

tokenizer = AutoTokenizer.from_pretrained("DeepChem/ChemBERTa-100M-MLM")
model = AutoModelForMaskedLM.from_pretrained("DeepChem/ChemBERTa-100M-MLM")

# Use a pipeline as a high-level helper
from transformers import pipeline

pipe = pipeline("fill-mask", model="DeepChem/ChemBERTa-5M-MLM")
# Load model directly
from transformers import AutoTokenizer, AutoModelForMaskedLM

tokenizer = AutoTokenizer.from_pretrained("DeepChem/ChemBERTa-5M-MLM")
model = AutoModelForMaskedLM.from_pretrained("DeepChem/ChemBERTa-5M-MLM")
```

# ChemBERTa-77M-MTR
## Model Details

| Property | Value |
|---|---|
| HF ID | DeepChem/ChemBERTa-77M-MLM |
| Developer | DeepChem |
| Parameter Count | 77 Million |
| Model Type | Chemical Language Model |
| Training Approach | Masked Token Regression |
| Model URL | [DeepChem/ChemBERTa-77M-MLM — Hugging Face](https://huggingface.co/DeepChem/ChemBERTa-77M-MLM) |

## What is ChemBERTa-77M-MTR?
ChemBERTa-77M-MTR is an advanced chemical language model developed by DeepChem that employs masked token regression for molecular property prediction. Built upon the BERT architecture, this model specifically focuses on understanding and processing chemical structures and properties.

## Implementation Details
The model utilizes a 77 million parameter architecture optimized for chemical data processing. It implements masked token regression (MTR) as its primary training objective, differentiating it from traditional masked language modeling approaches.

77M parameter architecture optimized for chemical data
Masked Token Regression training methodology
Built on the ChemBERTa framework
Specialized for molecular property prediction

### Core Capabilities
Chemical structure representation learning
Molecular property prediction
Chemical similarity assessment
Structure-property relationship analysis

## Frequently Asked Questions
Q: What makes this model unique?

ChemBERTa-77M-MTR's uniqueness lies in its masked token regression approach, which is specifically designed for chemical property prediction, unlike traditional masked language modeling used in standard BERT models.

Q: What are the recommended use cases?

The model is best suited for molecular property prediction tasks, drug discovery applications, and chemical structure analysis where understanding structure-property relationships is crucial.

```python
# Load model directly
from transformers import AutoTokenizer, AutoModel

tokenizer = AutoTokenizer.from_pretrained("DeepChem/ChemBERTa-77M-MTR")
model = AutoModel.from_pretrained("DeepChem/ChemBERTa-77M-MTR")
```

# ChemBERTa-10M-MTR
```python
# Load model directly
from transformers import AutoTokenizer, AutoModel

tokenizer = AutoTokenizer.from_pretrained("DeepChem/ChemBERTa-10M-MTR")
model = AutoModel.from_pretrained("DeepChem/ChemBERTa-10M-MTR")
```

# ChemBERTa-5M-MTR
```python
# Load model directly
from transformers import AutoTokenizer, AutoModel

tokenizer = AutoTokenizer.from_pretrained("DeepChem/ChemBERTa-5M-MTR")
model = AutoModel.from_pretrained("DeepChem/ChemBERTa-5M-MTR")
```
