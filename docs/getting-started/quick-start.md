# Quick Start

Get Agentomics-ML running in under 5 minutes using pre-built Docker images.

## Prerequisites

- [Docker](https://docs.docker.com/get-docker/) installed and running
- An API key from [OpenRouter](https://openrouter.ai/), [OpenAI](https://platform.openai.com/), or [Anthropic](https://console.anthropic.com/)

## Steps

### 1. Clone the Repository

```bash
git clone https://github.com/BioGeMT/Agentomics-ML.git
cd Agentomics-ML
```

### 2. Set Your API Key

```bash
export OPENROUTER_API_KEY="your-key-here"
```

Or create a `.env` file (see `.env.example` for template).

### 3. Run the Agent

```bash
./run.sh --pull-images
```

The `--pull-images` flag automatically downloads pre-built Docker images from Docker Hub, which is the fastest way to get started.

### 4. Follow the Interactive Prompts

The agent will prompt you to:

1. **Select a model** - Choose from available LLMs
2. **Select a dataset** - Use your own or download examples
3. **Configure iterations** - How many optimization cycles to run
4. **Choose validation metric** - ACC, AUROC, F1, etc.

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
conda env create -f competitors/environment_datasets.yaml
conda activate agentomics-datasets
python src/utils/create_datasets.py
conda deactivate
```

## What Happens Next

The agent will:

1. Prepare your dataset
2. Run iterative ML development cycles
3. Save the best model to `outputs/<agent_id>/`

Results include trained models, inference scripts, and detailed reports.

## Next Steps

- [Installation Options](installation.md) - Docker build, local mode, Ollama
- [Running the Agent](../user-guide/running-agent.md) - Advanced usage
- [CLI Options](../configuration/cli-options.md) - All available flags
