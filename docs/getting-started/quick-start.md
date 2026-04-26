# Quick Start

Get Agentomics-ML running in under 5 minutes using pre-built Docker images.

## Prerequisites

- [Docker](https://docs.docker.com/get-docker/) installed and running
- An API key from [OpenRouter](https://openrouter.ai/) or [OpenAI](https://platform.openai.com/)

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
# Edit .env and set at least one API key:
# OPENROUTER_API_KEY or OPENAI_API_KEY
```

### 3. Run the Agent

```bash
./run.sh
```

The default Docker mode downloads pre-built Docker images from Docker Hub automatically, which is the fastest way to get started.

### 4. Follow the Interactive Prompts

The agent will prompt you to:

1. **Select a model** - Choose from available LLMs
2. **Select a dataset** - Use your own or download examples
3. **Configure iterations** - How many optimization cycles to run
4. **Choose validation metric** - see `./run.sh --list-metrics`

## Using Your Own Dataset

Place your data in `datasets/<your_dataset_name>/`:

```
datasets/my_dataset/
├── train.csv           # Required: training data
├── validation.csv      # Optional: validation data
├── test.csv            # Optional: hidden test set
└── dataset_description.md  # Optional: domain context
```

See [Preparing Datasets](../user-guide/datasets.md) for details.

## Example Datasets

Download example datasets to try:

```bash
./download_example_datasets.sh
```

## What Happens Next

The agent will:

1. Prepare your dataset
2. Run iterative ML development cycles
3. Save the best model to `outputs/<agent_id>/`

Results include trained models, inference scripts, and detailed reports in `outputs/<agent_id>/reports/`, plus PDF reports in `outputs/<agent_id>/pdf_reports/`.

## Next Steps

- [Installation Options](installation.md) - Docker build, local mode, Ollama
- [Running the Agent](../user-guide/running-agent.md) - Advanced usage
- [CLI Options](../configuration/cli-options.md) - All available flags
