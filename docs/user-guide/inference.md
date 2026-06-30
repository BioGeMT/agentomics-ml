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
  --input /path/to/data.csv \
  --output /path/to/predictions.csv
```

`--input` accepts either a **CSV file** or a **contract split folder** (`input/`
+ optional `labels.csv`). See [Input Data Format](#input-data-format).

## Arguments

### Required

| Argument | Description |
|----------|-------------|
| `--agent-dir` | Path to completed agent output folder |
| `--input` | A CSV file, or a contract split folder (`input/` + optional `labels.csv`) |
| `--output` | Path where predictions will be saved |

### Optional

| Argument | Description |
|----------|-------------|
| `--label-col` | Label column in the input **CSV** — when set, metrics are computed against it and written to `<output>.metrics.json`. Ignored for a split folder; there, metrics run only if the folder has a `labels.csv` |
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

`--input` can be either of two shapes:

**A CSV file** — the common case. It is converted into a contract split before
inference. Provide the same feature columns the model was trained on. If the CSV
also has a label column, pass `--label-col <name>` to compute metrics (the label
is split out, not fed to the model); omit it for prediction-only.

```csv
feature1,feature2,feature3,label
1.2,3.4,5.6,positive
7.8,9.0,1.2,negative
```

**A contract split folder** — `input/` (+ optional `labels.csv`), matching the
structure of the training split's `input/`:

```text
new_samples/
├── input/
│   └── data.csv      # feature files (same top-level structure as training)
└── labels.csv        # optional (id,label or id,numeric_label) -> enables metrics
```

For both shapes, `input/` must match the training split's top-level structure and
carry the sample IDs the generated `inference.py` expects (a CSV without an `id`
column gets sequential IDs assigned during conversion). Metrics are produced when
labels are available — via `--label-col` (CSV) or a present `labels.csv` (folder).

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
