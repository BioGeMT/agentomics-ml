# Foundation Models

Pre-trained models for specialized omics domains.

## Overview

Foundation models are large pre-trained models specialized for specific data types. Agentomics-ML can pre-download these models to speed up agent runs.

## Available Types

| Type | Domain | Use Case |
|------|--------|----------|
| `dna` | Genomics | DNA sequences, variants |
| `rna` | Transcriptomics | RNA sequences, expression |
| `protein` | Proteomics | Protein sequences, structure |
| `molecule` | Chemistry | Small molecules, drugs |

## Pre-downloading Models

Download foundation models before running:

```bash
./run.sh --foundation-model-type dna
```

This downloads relevant models to the Docker image, avoiding download delays during agent execution.

You can also use `--foundation-model-type all` to include every type.

## Multiple Types

Download multiple types by running multiple times:

```bash
./run.sh --foundation-model-type dna
./run.sh --foundation-model-type protein
```

In local mode (`--local`), models are downloaded into the workspace instead of
being baked into a Docker image.

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

Foundation model configurations ship with the Python package under:

```
src/agentomics/resources/foundation_models/
```

The repository also keeps a mirrored `foundation_models/` directory for development and documentation, but packaged installs load the bundled resources directly.

Each type has a configuration specifying:
- Model names and sources
- Download locations
- Usage instructions for the agent

## Without Pre-downloading

If you don't pre-download, the agent can still use foundation models but will download them during execution (slower first run).

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

1. Add configuration to `foundation_models/`
2. Update the download script
3. Add usage instructions for the agent

For packaged support, add the same files under `src/agentomics/resources/foundation_models/` so they are included in installs and Docker images.

## Related

- [Agent Architecture](../how-it-works/architecture.md) - How models are used
- [GPU Settings](../developer/gpu-settings.md) - GPU configuration
