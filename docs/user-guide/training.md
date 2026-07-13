# Re-training Models

After the agent completes a run, you can re-train its model on new data using
`scripts/train.sh`. The script reuses the run's training script, conda
environment, and trained label mapping, so the model and preprocessing stay
identical — only the data changes.

## When to Use

- Train on updated or expanded datasets
- Refit the same model on a new split

## Requirements

- A completed agent output directory (a finished `outputs/<agent_id>` run)
- [Docker](https://docs.docker.com/get-docker/) **or** [Conda](https://docs.conda.io/en/latest/miniconda.html)

## Basic Usage

```bash
./scripts/train.sh \
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

## Docker Mode

To re-train inside the container, override the image entrypoint to run
`train.sh`, and mount the run and your dataset. Writing artifacts into the
already-mounted run directory avoids a separate output mount:

```bash
docker run --rm --gpus all \
  -v "$(pwd)/outputs/my_run_1:/agent" \
  -v "$(pwd)/new_dataset:/data:ro" \
  -e HOST_UID=$(id -u) -e HOST_GID=$(id -g) \
  --entrypoint /repository/scripts/train.sh \
  biogemt/agentomics:latest \
  --agent-dir /agent \
  --dataset-dir /data \
  --artifacts-dir /agent/retrained_artifacts
```

The retrained artifacts appear under `outputs/my_run_1/retrained_artifacts/`.
Drop `--gpus all` (or add `--cpu-only` after the image) to train on CPU. No API
key is needed — re-training does not call an LLM.

## How It Works

1. Prepares `--dataset-dir` into the run's contract format — each split as
   `input/` + a numeric `labels.csv` — using the run's task type and trained
   label mapping. CSV-form splits are converted to this shape automatically;
   folder-form splits are used as-is (only their labels are numericized).
2. Reuses the model's conda environment under the code path's `.conda/envs/`,
   recreating it from `environment.yml` if it's missing.
3. Runs the run's `model_training/train.py` on the prepared `train`/`validation`.
4. Writes artifacts to `--artifacts-dir` and prints a summary.

## GPU Support

GPU is used automatically if available. Disable it with `--cpu-only` (local), or
by omitting `--gpus` (Docker).

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

The code path must contain `environment.yml` (or `runtime_info/environment.yml`).
Check that the run completed and produced a model.

### "labels.csv contains labels absent from label_to_scalar"

Your data contains a class the model wasn't trained on. The labels must match the
classes the run was trained on.

## Next Steps

- [Running Inference](inference.md) - Make predictions with trained models
- [Understanding Outputs](outputs.md) - Explore what the agent produces
