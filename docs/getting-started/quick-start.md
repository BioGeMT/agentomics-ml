# Quick Start

Get your first Agentomics run started using Docker and an example dataset.

## Prerequisites

- [Docker](https://docs.docker.com/get-docker/) installed and running
- [Conda](https://docs.conda.io/en/latest/miniconda.html) for dataset preparation
- An API key from a supported provider:
  - [OpenRouter](https://openrouter.ai/) (recommended for variety)
  - [OpenAI](https://platform.openai.com/)
  - [Anthropic](https://www.anthropic.com/)

For local models with Ollama, see the [Ollama installation section](installation.md#ollama-local-llms).

**Note:** First use will download a Docker container image (several GB) and an example dataset. The example dataset download creates a temporary Conda environment.

## Steps

### 1. Clone the repository

```bash
git clone https://github.com/BioGeMT/agentomics-ml.git
cd agentomics-ml
```

### 2. Configure provider credentials

Create a `.env` file with your API key:

```bash
cp .env.example .env
# Edit .env and add your provider key:
# OPENROUTER_API_KEY=your-key-here
# OPENAI_API_KEY=your-key-here
# or ANTHROPIC_API_KEY=your-key-here
```

**Privacy notice:** External providers (OpenRouter, OpenAI, Anthropic) will receive data-derived context including dataset structure, features, and code. For sensitive data, use local Ollama models instead. See [Privacy and execution](../../README.md#privacy-and-execution).

### 3. Download an example dataset

```bash
./scripts/download_example_dataset.sh --dataset breast_cancer
```

List all available examples:

```bash
./scripts/download_example_dataset.sh --list
```

### 4. Run the agent

```bash
./run.sh
```

**Before running:** Starting a run invokes the configured LLM. External providers may charge for usage, and each additional iteration generally uses more tokens and compute.

### 5. Follow the interactive prompts

The wizard will ask you to:

1. **Select a model** - Agentomics uses the configured provider automatically (or asks you to choose when multiple providers are configured), then you select a model
2. **Select a dataset** - Pick from downloaded datasets (e.g., `breast_cancer`)
3. **Configure iterations** - How many development cycles to run

**What is an iteration?** An iteration is one autonomous development cycle in which Agentomics proposes, trains, and evaluates a candidate method. More iterations give Agentomics more opportunities to find better solutions, but each iteration uses LLM tokens and compute time.

**Validation metric:** For classification, the default is AUROC. For regression, the default is MAE (mean absolute error). To choose a different metric, use `--val-metric <metric>` or see `./run.sh --list-metrics`.

## What happens next

Agentomics will:

1. **Prepare your dataset**: Convert to internal format and validate structure
2. **Run iterations**: Each iteration explores a different ML approach
3. **Save the best model**: Selected by validation metric performance
4. **Generate reports**: Markdown and PDF summaries

Outputs appear in `outputs/<agent_id>/`:

```text
outputs/<agent_id>/
├── best_iteration_snapshot/
│   ├── model_training/
│   │   ├── train.py              # Recreate training
│   │   └── training_artifacts/   # Trained model and related artifacts
│   ├── model_inference/
│   │   └── inference.py          # Inference implementation
│   └── environment.yml            # Dependencies
└── reports/
    ├── markdown/                  # Iteration reports
    └── pdf/                       # PDF iteration reports and plots
```

## Use the model for inference

After a run completes, use the inference wrapper:

```bash
./scripts/inference.sh \
  --agent-dir outputs/<agent_id> \
  --input <input-folder> \
  --output predictions.csv
```

The `inference.py` file is the underlying implementation called by this wrapper. See the [Inference Guide](../user-guide/inference.md) for details.

## Using your own dataset

Place your dataset in `datasets/<your_dataset_name>/`:

**Folder-based layout** (for any data type):

```text
datasets/my_dataset/
├── train/
│   ├── input/          # Your data files (images, CSVs, sequences, etc.)
│   └── labels.csv      # Required: id,label
├── validation/         # Optional
│   ├── input/
│   └── labels.csv
└── metadata.json       # Required: {"task_type": "classification" or "regression"}
```

**Flat CSV layout** (for tabular data):

```text
datasets/my_dataset/
├── train.csv           # Features + labels in one file
├── validation.csv      # Optional
└── metadata.json       # Required: {"task_type": "...", "label_column": "target"}
```

**Hidden test data** (optional, for final unbiased evaluation):

Keep test data separate so the agent never sees it:

```text
test_datasets/my_dataset/
└── test/
    ├── input/
    └── labels.csv
```

See [Preparing Datasets](../user-guide/datasets.md) for the complete technical contract and [Dataset Best Practices](../user-guide/dataset-best-practices.md) for scientific guidance on preventing data leakage.

## Troubleshooting

**Docker not found:**
```bash
# Install Docker from https://docs.docker.com/get-docker/
# Ensure the daemon is running
docker ps  # Should show running containers or empty list
```

**Permission denied (Docker):**
```bash
# Add your user to the docker group (Linux)
sudo usermod -aG docker $USER
# Log out and log back in
```

**Missing dataset:**
```bash
# Download specific dataset
./scripts/download_example_dataset.sh --dataset breast_cancer

# Or list all available datasets
./scripts/download_example_dataset.sh --list
```

**Provider authentication failed:**
- Verify your API key is correct in `.env`
- Check that the provider variable name matches (e.g., `OPENROUTER_API_KEY`)
- Ensure the `.env` file is in the repository root

## Next steps

- [Installation Options](installation.md) - Docker build, local mode, Ollama setup
- [Running the Agent](../user-guide/running-agent.md) - Advanced configuration
- [CLI Options](../configuration/cli-options.md) - All available flags
- [Provider Configuration](../configuration/providers.md) - Configure additional LLM providers
