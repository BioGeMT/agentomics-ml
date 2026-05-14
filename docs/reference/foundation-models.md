# Foundation Models

Pre-trained models for specialized omics domains.

## Overview

Foundation models are large pre-trained models specialized for specific data types. Agentomics-ML can enable selected model families so the agent can use their catalog and local/downloaded weights during a run.

## Available Types

| Type | Domain | Use Case |
|------|--------|----------|
| `dna` | Genomics | DNA sequences, variants |
| `rna` | Transcriptomics | RNA sequences, expression |
| `protein` | Proteomics | Protein sequences, structure |
| `molecule` | Chemistry | Small molecules, drugs |

## Enabling Models

Enable foundation models before running:

```bash
./run.sh --foundation-models-type dna
```

This enables the selected family for the run and makes its catalog available to the agent. In Docker mode, the launcher pulls or builds an image tagged for that foundation-model type. In local mode, models are downloaded into the local workspace cache.

You can also use `--foundation-models-type all` to include every type.

## Multiple Types

Enable a different type per run, or use `all` when the agent should see every configured type:

```bash
./run.sh --foundation-models-type dna
./run.sh --foundation-models-type protein
```

## DNA Models

For genomic sequence data:

- Variant effect prediction
- Regulatory element detection
- Sequence classification

Example datasets:
- Gene expression from DNA features
- SNP effect prediction
- Promoter classification

## RNA Models

For transcriptomic data:

- RNA sequence analysis
- Secondary structure prediction
- Expression-based classification

Example datasets:
- RNA-seq classification
- Splice site prediction
- RNA modification detection

## Protein Models

For protein sequence data:

- Protein function prediction
- Structure-based classification
- Interaction prediction

Example datasets:
- Protein family classification
- Enzyme activity prediction
- Binding site detection

## Molecule Models

For small molecule/chemical data:

- Drug property prediction
- Molecular classification
- Activity prediction

Example datasets:
- Drug-target interaction
- Toxicity prediction
- ADMET properties

## How the Agent Uses Foundation Models

1. **Discovery** - Agent queries available foundation models
2. **Selection** - Agent chooses appropriate model for the data
3. **Embedding** - Features extracted using the model
4. **Training** - Embeddings used as input to ML model

## Configuration

Foundation model configurations are in:

```
foundation_models/models.yaml
```

The YAML file lists each model family with:
- Model names and sources
- Domain type (`dna`, `rna`, `protein`, or `molecule`)
- Brief summaries
- Usage instructions for the agent

## Without Foundation Models

If `--foundation-models-type` is omitted on a fresh run, no foundation models are made available to the agent. Forked runs inherit the source run's foundation-model type when the flag is omitted.

## Storage Requirements

Foundation models can be large:

| Type | Approximate Size |
|------|-----------------|
| DNA | 1-5 GB |
| RNA | 1-5 GB |
| Protein | 2-10 GB |
| Molecule | 0.5-2 GB |

Ensure sufficient disk space in the Docker volume or local environment.

## GPU Acceleration

Foundation models benefit significantly from GPU:

- **With GPU**: Fast embedding generation
- **CPU only**: Much slower, but functional

Use `--cpu-only` if GPU unavailable, but expect longer run times for foundation model-based approaches.

## Custom Foundation Models

To add custom foundation models:

1. Add the family to `foundation_models/models.yaml`
2. Add a companion usage guide under `foundation_models/`
3. Ensure the listed model names can be downloaded by `src/utils/download_foundation_models.py`

## Related

- [Agent Architecture](../how-it-works/architecture.md) - How models are used
- [GPU Settings](../developer/gpu-settings.md) - GPU configuration
