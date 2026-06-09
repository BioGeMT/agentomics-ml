# Foundation Models

This `supplementary/` folder provides documentation for pretrained foundation
models, located in `foundation_models/`. They can be used as feature extractors or
fine-tuned for a downstream task.

| Data Type | Model | Documentation |
|-----------|-------|---------------|
| Protein | ESM-2 | [foundation_models/ESM-2.md](foundation_models/ESM-2.md) |
| DNA | HyenaDNA | [foundation_models/HyenaDNA.md](foundation_models/HyenaDNA.md) |
| DNA | Nucleotide Transformer | [foundation_models/NucleotideTransformer.md](foundation_models/NucleotideTransformer.md) |
| RNA | RiNALMo | [foundation_models/rinalmo.md](foundation_models/rinalmo.md) |
| Molecule (SMILES) | ChemBERTa | [foundation_models/ChemBERTa.md](foundation_models/ChemBERTa.md) |

Each model doc covers what the model is, its available variants with HuggingFace
IDs, loading and fine-tuning examples, and any extra Python packages it needs.

## Random-head gotcha (applies to all of these models)

These checkpoints were pretrained **without** a task head. Loading them with one —
via `AutoModelForSequenceClassification.from_pretrained(<id>)`,
`AutoModelForTokenClassification`, or a custom head you attach on top of the base
model's embeddings — adds a head whose weights are **randomly initialized**.
HuggingFace signals this with a warning such as:

> Some weights of EsmForSequenceClassification were not initialized from the model
> checkpoint at facebook/esm2_t33_650M_UR50D and are newly initialized:
> ['classifier.bias', 'classifier.weight']. You should probably TRAIN this model on
> a down-stream task to be able to use it for predictions and inference.

Consequences for training and inference:

- The random head produces meaningless predictions until it is **trained**.
- After training, **save the full model (base + trained head)**, and have inference
  **load that saved model**.
- Inference must **not** use a random head for predictions.

This is the most common way a foundation-model pipeline silently ends up scoring no
better than chance at inference even though training appeared to work.

## Notes

- Model weights are not bundled. They download on demand when code calls loading
  APIs such as `from_pretrained()`; the first use pays the download time.
- Some models need extra Python packages.
  Each model doc lists them; install them into the run's conda environment when
  needed.
