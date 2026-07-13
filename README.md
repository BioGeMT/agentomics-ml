# Agentomics

**Autonomous ML development for supervised learning tasks**

Agentomics creates, tests, and evaluates machine learning models autonomously. Give it a dataset, and it explores various approaches, selects the best model, and produces training code, inference scripts, and evaluation reports.

## How it works

1. **Provide a dataset**: Folder-based splits with input files and labels
2. **Autonomous iterations**: Agentomics experiments with models, features, and strategies (each iteration is one development cycle that proposes, trains, and evaluates a candidate method)
3. **Get results**: Trained model, inference script, training code, and detailed reports

## Who it's for

Agentomics is designed for ML engineers and computational biologists working on supervised learning tasks. It supports:

- **Classification** (binary and multi-class)
- **Regression**

Originally built for biomedical data (protein engineering, drug discovery, regulatory genomics), it works with any data type through a generic folder-based contract.

## Try it

**Prerequisites:**
- [Docker](https://www.docker.com/) installed and running
- [Conda](https://docs.conda.io/en/latest/miniconda.html) for dataset preparation
- An API key from [OpenRouter](https://openrouter.ai/), [OpenAI](https://platform.openai.com/), or [Anthropic](https://www.anthropic.com/)

**Setup note:** First use downloads a Docker container image (several GB) and an example dataset. The example-data helper creates and reuses a dedicated Conda environment named `agentomics-datasets`.

**Cost note:** Starting a run invokes the configured LLM. External providers may charge for usage, and each additional iteration generally uses more tokens and compute.

```bash
git clone https://github.com/BioGeMT/agentomics-ml.git
cd agentomics-ml

# Configure provider credentials
cp .env.example .env
# Edit .env and add your API key (OPENROUTER_API_KEY, OPENAI_API_KEY, or ANTHROPIC_API_KEY)

# Download example dataset
./scripts/download_example_dataset.sh --dataset breast_cancer

# Optional: Validate setup and dataset before running
./run.sh --doctor
./run.sh --validate-dataset breast_cancer

# Quick demo run (recommended for first time - 2 iterations)
./run.sh --preset demo --dataset breast_cancer

# Or: Start interactive run with custom configuration
./run.sh
```

**Pre-flight validation:**
- `./run.sh --doctor` - Check environment, provider credentials, disk space
- `./run.sh --validate-dataset <name>` - Validate dataset structure, labels format, train/validation ID overlap

Both commands exit quickly without creating environments, modifying data, or contacting providers.

**Demo preset:** For first-time users, `--preset demo` runs 2 iterations with summary verbosity to verify setup and produce sample output. This is not sufficient for model comparison or scientific conclusions.

**Interactive wizard:** Without the demo preset, the wizard will prompt you to select:
- **Model**: Choose from your configured provider's available models
- **Dataset**: Select from downloaded datasets
- **Iterations**: Number of development cycles to run

## What you get

After a run completes, outputs appear in `outputs/<agent_id>/`:

```text
outputs/<agent_id>/
├── best_iteration_snapshot/
│   ├── model_training/
│   │   ├── train.py              # Reproduce training
│   │   └── training_artifacts/   # Trained model and related artifacts
│   ├── model_inference/
│   │   └── inference.py          # Inference implementation
│   └── environment.yml            # Conda environment
└── reports/
    ├── markdown/                  # Iteration reports
    └── pdf/                       # PDF iteration reports and plots
```

Use the inference wrapper for predictions on new data:

```bash
./scripts/inference.sh \
  --agent-dir outputs/<agent_id> \
  --input <input-folder> \
  --output predictions.csv
```

See [Inference Guide](docs/user-guide/inference.md) for details.

## Use your own data

Place your dataset in `datasets/<name>/` following the [dataset contract](docs/user-guide/datasets.md):

**Folder-based layout** (for any data type):
```text
datasets/my_dataset/
├── train/
│   ├── input/              # Your data files (any format)
│   └── labels.csv          # id,label
├── validation/             # Optional
│   ├── input/
│   └── labels.csv
└── metadata.json           # {"task_type": "classification"}
```

**Flat CSV layout** (for tabular data):
```text
datasets/my_dataset/
├── train.csv               # Features + labels in one file
├── validation.csv          # Optional
└── metadata.json           # {"task_type": "classification", "label_column": "target"}
```

See [Preparing Datasets](docs/user-guide/datasets.md) for the complete technical specification and [Dataset Best Practices](docs/user-guide/dataset-best-practices.md) for guidance on preventing data leakage and ensuring scientific validity.

## Privacy and execution

**Data handling:**
- Agentomics executes model training in a Docker container with read-only access to your datasets
- Training code, features, dataset structure, and summary statistics may be sent to your configured LLM provider
- Hidden test data remains isolated from the agent and LLM

**Provider options:**
- **External providers** (OpenAI, Anthropic, OpenRouter): Data-derived context leaves your machine. Review your organization's data governance policies before using with sensitive data.
- **Local providers** (Ollama): LLM runs on your machine. Data-derived context stays local.

**Important:** Docker isolation protects your filesystem but does not prevent data from reaching the configured LLM provider. Choose your provider based on your data sensitivity.

## Installation options

### Docker (recommended)

Default mode using pre-built images:

```bash
./run.sh
```

Build images locally instead:

```bash
./run.sh --build-images
```

### Local mode (no Docker)

For development or environments where Docker is unavailable:

```bash
./run.sh --local
```

**Security notice:** Local mode executes agent-generated code without containerization. Use only in secure environments.

### Ollama (local LLMs)

For privacy-focused or offline use with local models, see the [complete Ollama setup guide](docs/getting-started/installation.md#ollama-local-llms).

See [Installation Guide](docs/getting-started/installation.md) for complete setup instructions and [Provider Configuration](docs/configuration/providers.md) for all supported LLM providers.

## Key features

- **Generic**: Folder-based contract supports any data type (images, sequences, tabular, audio, etc.)
- **Secure**: Docker containers with read-only dataset mounts and an isolated writable workspace
- **Reproducible**: Outputs include training scripts, inference code, and Conda environments
- **Trustworthy**: Hidden test sets remain agent-inaccessible; evaluation is programmatic
- **Foundation models**: Leverage Hugging Face models for embeddings and fine-tuning
- **Multiple providers**: OpenAI, Anthropic, OpenRouter, local Ollama, or custom OpenAI-compatible endpoints

## Documentation

- **Getting Started**
  - [Quick Start](docs/getting-started/quick-start.md)
  - [Installation](docs/getting-started/installation.md)
- **User Guides**
  - [Preparing Datasets](docs/user-guide/datasets.md) - Technical specification
  - [Dataset Best Practices](docs/user-guide/dataset-best-practices.md) - Scientific guidance
  - [Running the Agent](docs/user-guide/running-agent.md)
  - [Understanding Outputs](docs/user-guide/outputs.md)
- **Configuration**
  - [CLI Options](docs/configuration/cli-options.md)
  - [Provider Setup](docs/configuration/providers.md)
- **Full documentation**: [https://biogemt.github.io/agentomics-ml/](https://biogemt.github.io/agentomics-ml/)

## News

- **Agentomics published in Bioinformatics** - Martinek *et al.* (2026)
- Now supports any data type through supplementary materials and foundation models

## Citation

If you use Agentomics in your work, please cite:

Martinek *et al.* (2026).
*Agentomics: An Agentic System that Autonomously Develops Novel State-of-the-Art Solutions for Biomedical Machine Learning Tasks*.
Bioinformatics ([https://doi.org/10.1093/bioinformatics/btag250](https://doi.org/10.1093/bioinformatics/btag250))

Preprint: [https://www.biorxiv.org/content/10.64898/2026.01.27.702049v1](https://www.biorxiv.org/content/10.64898/2026.01.27.702049v1)

## Contributing

We welcome issues, suggestions, and contributions. Contact: [martinekvlastimil95@gmail.com](mailto:martinekvlastimil95@gmail.com)

## Roadmap

Features in development:
- Improved local model support and configuration
- Remote GPU support for GCP

## License

MIT. See [LICENSE](LICENSE).
