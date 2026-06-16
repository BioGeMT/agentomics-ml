# Running Inference

Use a trained model to predict on new data with `scripts/inference.sh`. It can
also compute metrics against true labels, and evaluate every archived iteration.

## Requirements

- A completed agent output directory (a finished `outputs/<agent_id>` run)
- [Docker](https://docs.docker.com/get-docker/) **or** [Conda](https://docs.conda.io/en/latest/miniconda.html)

## Basic Usage

```bash
./scripts/inference.sh \
  --agent-dir outputs/<agent_id> \
  --input /path/to/input_folder \
  --output /path/to/predictions.csv
```

## Arguments

### Required

| Argument | Description |
|----------|-------------|
| `--agent-dir` | Path to completed agent output folder |
| `--input` | Path to an input folder without labels |
| `--output` | Path where predictions will be saved |

### Optional

| Argument | Description |
|----------|-------------|
| `--label-col` | Name of the true label column in `--input`. When set, metrics are computed against it and written to `<output>.metrics.json` next to `--output` |
| `--code-path` | Code directory to use, relative to `--agent-dir` (default: `best_iteration_snapshot`) |
| `--all-iterations` | Run inference for every `run/iteration_N` against `--input` (see below) |
| `--remove-conda-env` | Remove the model conda environment after inference |
| `--cpu-only` | Run without GPU |
| `--help` | Show help message |

## Docker Mode

Override the image entrypoint to run `inference.sh`, mounting the run and a
directory holding your input (and receiving the output):

```bash
docker run --rm --gpus all \
  -v "$(pwd)/outputs/my_run_1:/agent" \
  -v "$(pwd)/data:/data" \
  -e HOST_UID=$(id -u) -e HOST_GID=$(id -g) \
  --entrypoint /repository/scripts/inference.sh \
  biogemt/agentomics:latest \
  --agent-dir /agent \
  --input /data/new.csv \
  --output /data/predictions.csv \
  --label-col label
```

Predictions (and metrics, when `--label-col` is set) appear in `./data`. Drop
`--gpus all` (or add `--cpu-only` after the image) to run on CPU. The model
environment is reused from the run, or rebuilt from `environment.yml` under
`/agent` if absent — pass `--remove-conda-env` to discard it afterwards.

## Input Data Format

Your input folder should:

- Match the structure of the training split's `input/` folder
- Contain the sample IDs needed by the generated `inference.py`
- Not include `labels.csv` or target labels

For a tabular dataset, the folder can contain a CSV file:

```text
new_samples/input/
└── data.csv
```

```csv
id,feature1,feature2,feature3
sample-1,1.2,3.4,5.6
sample-2,7.8,9.0,1.2
```

## Output Format

The generated `inference.py` must preserve input sample IDs and write a prediction for every sample. Classification runs produce `id`, `prediction`, and probability columns when probabilities are available. Regression runs produce `id` and `prediction`. Additional columns are run-specific.

## Computing Metrics

Pass `--label-col` to score predictions against the true labels in `--input`.
Metrics are computed by the bundled evaluator using the run's task type and
label mapping, aligned to predictions by `id`, and written to
`<output>.metrics.json` next to `--output`.

```bash
./scripts/inference.sh \
  --agent-dir outputs/<agent_id> \
  --input labeled_data.csv \
  --output predictions.csv \
  --label-col label
# -> predictions.csv and predictions.metrics.json
```

## Evaluating Every Iteration

`--all-iterations` runs inference on each `run/iteration_N` snapshot in turn:

```bash
./scripts/inference.sh \
  --agent-dir outputs/<agent_id> \
  --input labeled_data.csv \
  --output preds/out.csv \
  --label-col label \
  --all-iterations
```

Per-iteration predictions are written to `preds/<iteration>_predictions.csv`
(and `<iteration>_predictions.metrics.json` when `--label-col` is set). Each
iteration's environment is rebuilt and removed in turn, and a failing iteration
warns and continues.

## What's in best_iteration_snapshot

```
outputs/<agent_id>/best_iteration_snapshot/
├── model_inference/
│   └── inference.py        # Inference script
├── model_training/
│   ├── train.py            # Training script
│   └── training_artifacts/ # Model files (format varies)
├── runtime_info/
│   └── iteration_metadata.json
├── environment.yml
├── .conda/                 # Model conda environment (if present)
└── ...                     # Other artifacts (tokenizers, etc.)
```

## GPU Support

GPU is used automatically when the container can see it. Pass `--gpus all` to the
`docker run` command (Docker), and it works out of the box in local mode. To
force CPU, add `--cpu-only` or omit `--gpus`.

## Troubleshooting

### "environment.yml not found"

The code path must contain `environment.yml` (or `runtime_info/environment.yml`).
Check that the run completed and produced a model.

### "Column mismatch"

Ensure your input folder has the same top-level files and folders as the
training split's `input/` folder. Files inside matching top-level folders may
differ.

### "Model file not found"

Check that `best_iteration_snapshot/` contains the model artifacts. If the agent run failed, there may be no trained model.

### GPU out of memory

Use `--cpu-only`, or reduce the batch size in
`best_iteration_snapshot/model_inference/inference.py`.

## Next Steps

- [Understanding Outputs](outputs.md) - Full output structure
- [Re-training Models](training.md) - Train with new data
