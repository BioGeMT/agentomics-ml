# Re-training Models

After the agent completes a run, you can re-train its model on new data using
`agentomics-retrain`. The command reuses the run's training script, conda
environment, and trained label mapping, so the model and preprocessing stay
identical — only the data changes.

## When to Use

- Train on updated or expanded datasets
- Refit the same model on a new split

## Requirements

- A completed agent output directory (a finished `outputs/<agent_id>` run)
- The Agentomics package (`python3 -m pip install agentomics`)
- [Docker](https://docs.docker.com/get-docker/)

## Basic Usage

```bash
agentomics-retrain \
  --agent-dir outputs/<agent_id> \
  --dataset-dir /path/to/dataset \
  --artifacts-dir /path/to/output_artifacts
```

`--dataset-dir` is a dataset folder containing both a **`train`** and a
**`validation`** split (both required). See [Data Format](#data-format).

## Arguments

### Required

| Argument | Description |
|----------|-------------|
| `--agent-dir` | Completed agent output folder |
| `--dataset-dir` | Dataset folder with `train` and `validation` splits (see [Data Format](#data-format)) |
| `--artifacts-dir` | Where to write the new training artifacts |

### Optional

| Argument | Description |
|----------|-------------|
| `--label-col` | Label column name for **CSV-form** splits (overrides `metadata.json`). Not needed for folder splits, or when `metadata.json` declares `label_column` |
| `--iteration-dir` | Code directory to use, relative to `--agent-dir` (default: `best_iteration_snapshot`) |
| `--cpu-only` | Run without GPU |
| `--image` | Docker image to use (default: `biogemt/agentomics:<installed-package-version>`; use this option for an explicit override) |
| `--help` | Show help message |

## Data Format

`--dataset-dir` must contain a **`train`** and a **`validation`** split — both are
required, since `train.py` always receives validation data (and may use it for
early stopping). Each split can take either form (the same two forms the main run
accepts):

**Folder splits** — labels already separated into the contract shape:
```text
dataset/
├── train/
│   ├── input/          # feature files (same structure as the original run)
│   └── labels.csv      # columns: id,label
└── validation/
    ├── input/
    └── labels.csv
```

**Raw CSV splits** — features and label together in one file:
```text
dataset/
├── train.csv           # feature columns + a label column
├── validation.csv
└── metadata.json       # optional: {"label_column": "<name>"}
```
For CSV splits, name the label column with `--label-col` or a `metadata.json`
entry `label_column`.

The folder must contain **only** the splits (plus optional `metadata.json` /
`dataset_description.md`); extra files trigger a validation error.

Label *values* are mapped to numbers using the **run's own mapping**, so your
classes must match those the run was trained on — you don't encode them yourself.
A class the model never saw is rejected with an error.

## Docker Execution

The command mounts the agent directory,
dataset, and artifact output directory automatically:

```bash
agentomics-retrain \
  --agent-dir outputs/my_run_1 \
  --dataset-dir new_dataset \
  --artifacts-dir outputs/my_run_1/retrained_artifacts
```

The retrained artifacts appear under `outputs/my_run_1/retrained_artifacts/`.
Add `--cpu-only` to train without GPU access. No API key is needed — re-training
does not call an LLM.

## How It Works

1. Prepares `--dataset-dir` into the run's contract format — each split as
   `input/` + a numeric `labels.csv` — using the run's task type and trained
   label mapping. CSV-form splits are converted to this shape automatically;
   folder-form splits are used as-is (only their labels are numericized).
2. Restores a temporary container-local environment from
   `runtime_info/environment.tar.gz` when re-training the best snapshot in
   `full` mode; otherwise it rebuilds the environment from `environment.yml`.
3. Runs the run's `model_training/train.py` on the prepared `train`/`validation`.
4. Writes artifacts to `--artifacts-dir` and prints a summary.

## GPU Support

GPU access is enabled by default. Disable it with `--cpu-only`.

## Output

```
artifacts_dir/
├── ...                 # Artifacts produced by train.py
```

## Troubleshooting

### "Prepared validation split not found"

`--dataset-dir` must contain a `validation` split (a `validation/` folder or a
`validation.csv`), not just `train`. Re-training always needs validation data.

### "CSV dataset requires 'label_column' ..."

For CSV-form splits, name the label column with `--label-col`, or add
`label_column` to a `metadata.json` in the dataset folder.

### "... has unsupported top-level entries ..."

The dataset folder contains files other than the splits. Keep only `train`/
`validation` (or `train.csv`/`validation.csv`) plus optional `metadata.json` and
`dataset_description.md`.

### "environment.yml not found"

The iteration directory must contain `runtime_info/environment.yml`.
Check that the run completed and produced a model.

### "labels.csv contains labels absent from label_to_scalar"

Your data contains a class the model wasn't trained on. The labels must match the
classes the run was trained on.

## Next Steps

- [Running Inference](inference.md) - Make predictions with trained models
- [Understanding Outputs](outputs.md) - Explore what the agent produces
