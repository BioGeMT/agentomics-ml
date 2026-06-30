# Quick Start

Get Agentomics-ML running in a few minutes with the pre-built Docker image. For
a Conda-based setup, see [Local mode](installation.md#local-mode-no-docker).

## Prerequisites

- [Docker](https://docs.docker.com/get-docker/) installed and running
- An API key from a configured provider, such as [OpenRouter](https://openrouter.ai/) or [OpenAI](https://platform.openai.com/)

## Steps

### 1. Get the Repository and an Example Dataset

Clone the repo and download a single example dataset to try
(`AGO2_CLASH_Hejret2023`). The download script creates a small conda
environment and writes the data to `datasets/`:

```bash
git clone https://github.com/BioGeMT/Agentomics-ML.git
cd Agentomics-ML

./scripts/download_example_dataset.sh --dataset AGO2_CLASH_Hejret2023
```

### 2. Set a Provider Key

```bash
cp .env.example .env
# Edit .env and set at least one provider key:
# OPENROUTER_API_KEY, OPENAI_API_KEY, or ANTHROPIC_API_KEY
```

### 3. Run the Agent

Mount your `datasets/` folder and a fresh directory for this run's results:

```bash
mkdir -p outputs/my_run_1

docker run --rm -it \
  --env-file .env \
  -v "$(pwd)/datasets:/repository/datasets" \
  -v "$(pwd)/outputs/my_run_1:/workspace" \
  -e HOST_UID=$(id -u) -e HOST_GID=$(id -g) \
  biogemt/agentomics:latest \
  --dataset AGO2_CLASH_Hejret2023
```

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
└── dataset_description.md
```

(Optional) Put hidden test data under the matching `test_datasets/` folder:

```text
test_datasets/my_dataset/
└── test/
    ├── input/
    └── labels.csv
```

This held-out data is optional and is **not** used automatically during a run —
score the finished model on it afterward with `scripts/inference.sh` (see
[Running Inference](../user-guide/inference.md)). Run training with
`--dataset my_dataset`. See [Preparing Datasets](../user-guide/datasets.md) for details.

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
3. Save the best model to the mounted output directory (`outputs/my_run_1/`)

Results include trained models, inference scripts, markdown reports in `reports/markdown/`, and PDF reports in `reports/pdf/`.

## Next Steps

- [Installation Options](installation.md) - Docker mode, local mode, Ollama
- [Running the Agent](../user-guide/running-agent.md) - Advanced usage
- [CLI Options](../configuration/cli-options.md) - All available flags
