# Quick Start

Get Agentomics-ML running in under 5 minutes using pre-built Docker images.

## Prerequisites

- [Docker](https://docs.docker.com/get-docker/) installed and running
- An API key from a configured provider, such as [OpenRouter](https://openrouter.ai/) or [OpenAI](https://platform.openai.com/)

## Steps

### 1. Clone the Repository

```bash
git clone https://github.com/BioGeMT/Agentomics-ML.git
cd Agentomics-ML
```

### 2. Create a .env File and Set a Key

Docker mode requires a `.env` file in the repo root.

```bash
cp .env.example .env
# Edit .env and set at least one provider key:
# OPENROUTER_API_KEY, OPENAI_API_KEY, or ANTHROPIC_API_KEY
```

### 3. Run the Agent

```bash
./run.sh
```

### 4. Follow the Interactive Prompts

The agent will prompt you to:

1. **Select a model** - Choose from available LLMs
2. **Select a dataset** - Use your own or download examples
3. **Configure iterations** - How many optimization cycles to run

The validation metric defaults to `AUROC` for classification and `MAE` for regression. To choose one explicitly, pass `--val-metric`; see `./run.sh --list-metrics`.

## Using Your Own Dataset

Place your data in `datasets/<your_dataset_name>/`:

```text
datasets/my_dataset/
├── train/
│   ├── input/          # Required: model input files
│   └── labels.csv      # Required: id,label
├── validation/         # Optional
│   ├── input/
│   └── labels.csv
├── test/               # Optional hidden test set
│   ├── input/
│   └── labels.csv
└── dataset_description.md
```

See [Preparing Datasets](../user-guide/datasets.md) for details.

## Example Datasets

Download example dataset to try:

```bash
./scripts/download_example_dataset.sh
```

List other available examples with:

```bash
./scripts/download_example_dataset.sh --list
```

## What Happens Next

The agent will:

1. Prepare your dataset
2. Run iterative ML development cycles
3. Save the best model to `outputs/<agent_id>/`

Results include trained models, inference scripts, markdown reports in `outputs/<agent_id>/reports/markdown/`, and PDF reports in `outputs/<agent_id>/reports/pdf/`.

## Next Steps

- [Installation Options](installation.md) - Docker build, local mode, Ollama
- [Running the Agent](../user-guide/running-agent.md) - Advanced usage
- [CLI Options](../configuration/cli-options.md) - All available flags
