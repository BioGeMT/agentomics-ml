# Quick Start

Get Agentomics-ML running in a few minutes with the pre-built Docker image.

## Prerequisites

- [Docker](https://docs.docker.com/get-docker/) installed and running
- An API key from a configured provider, such as [OpenRouter](https://openrouter.ai/) or [OpenAI](https://platform.openai.com/)

## Steps

### 1. Install the CLI and an Example Dataset

Install the package from PyPI, then download a single example dataset to try
(`AGO2_CLASH_Hejret2023`). No repository clone is required:

```bash
python3 -m pip install agentomics
agentomics-download-dataset --dataset AGO2_CLASH_Hejret2023
```

### 2. Set a Provider Key

Export at least one provider key (or put it in a `.env` file in the current
directory):

```bash
export OPENROUTER_API_KEY=...   # or OPENAI_API_KEY / ANTHROPIC_API_KEY
```

### 3. Run the Agent

```bash
agentomics-run --dataset AGO2_CLASH_Hejret2023
```

`agentomics-run` launches the Agentomics Docker image for you, reads datasets
from `./datasets`, and writes this run's results to `./outputs/<agent_id>/`.

Drop `--dataset AGO2_CLASH_Hejret2023` to pick a model,
dataset, and iteration count interactively instead.

The validation metric defaults to `AUROC` for classification and `MAE` for regression. To choose one explicitly, pass `--val-metric`; list options with `--list-metrics` in place of the run arguments.

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
├── test/               # Optional; hidden from the agent and evaluated afterward
│   ├── input/
│   └── labels.csv
├── test_leftout/       # Optional additional hidden evaluation set
│   ├── input/
│   └── labels.csv
└── dataset_description.md
```

Held-out test splits are optional. Every top-level directory beginning with
`test` is withheld from the agent worker. After a successful run, Agentomics
evaluates the best iteration on each one and records separate output artifacts,
report sections, and W&B metric namespaces. Run training with
`--dataset my_dataset`. See
[Preparing Datasets](../user-guide/datasets.md) for details.

## Example Datasets

Download example dataset to try:

```bash
agentomics-download-dataset
```

List other available examples with:

```bash
agentomics-download-dataset --list
```

## What Happens Next

The agent will:

1. Prepare your dataset
2. Run iterative ML development cycles
3. Save the best model to the run's output directory (`outputs/<agent_id>/`)

Results include trained models, inference scripts, markdown reports in `reports/markdown/`, and PDF reports in `reports/pdf/`.

## Next Steps

- [Installation](installation.md) - Docker and Ollama setup
- [Running the Agent](../user-guide/running-agent.md) - Advanced usage
- [CLI Options](../configuration/cli-options.md) - All available flags
