# Re-training Models

After the agent completes a run, you can re-train its model on new data using
`scripts/train.sh`. The script reuses the run's training script and conda
environment, so the model and preprocessing stay identical — only the data
changes.

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
  --train-data /path/to/new_train.csv \
  --validation-data /path/to/new_validation.csv \
  --artifacts-dir /path/to/output_artifacts \
  --label-col label
```

## Arguments

### Required

| Argument | Description |
|----------|-------------|
| `--agent-dir` | Completed agent output folder |
| `--train-data` | New training CSV — raw, including the label column |
| `--validation-data` | New validation CSV — raw, including the label column |
| `--artifacts-dir` | Where to write the new training artifacts |
| `--label-col` | Name of the label column in `--train-data`/`--validation-data` |

### Optional

| Argument | Description |
|----------|-------------|
| `--code-path` | Code directory to use, relative to `--agent-dir` (default: `best_iteration_snapshot`) |
| `--cpu-only` | Run without GPU |
| `--help` | Show help message |

## Docker Mode

To re-train inside the container, override the image entrypoint to run
`train.sh`, and mount the run and your new data. Writing artifacts into the
already-mounted run directory avoids a separate output mount:

```bash
docker run --rm --gpus all \
  -v "$(pwd)/outputs/my_run_1:/agent" \
  -v "$(pwd)/new_data:/data:ro" \
  -e HOST_UID=$(id -u) -e HOST_GID=$(id -g) \
  --entrypoint /repository/scripts/train.sh \
  biogemt/agentomics:latest \
  --agent-dir /agent \
  --train-data /data/train.csv \
  --validation-data /data/validation.csv \
  --artifacts-dir /agent/retrained_artifacts \
  --label-col label
```

The retrained artifacts appear under `outputs/my_run_1/retrained_artifacts/`.
Drop `--gpus all` (or add `--cpu-only` after the image) to train on CPU. No API
key is needed — re-training does not call an LLM.

## How It Works

1. Converts your raw `--train-data`/`--validation-data` into the prepared format
   (`id` + `numeric_label`) the training script expects, using the run's own
   label mapping.
2. Reuses the model's conda environment under the code path's `.conda/envs/`,
   recreating it from `environment.yml` if it's missing.
3. Runs the run's `model_training/train.py` on the converted data.
4. Writes artifacts to `--artifacts-dir` and prints a summary.

## Data Format

Provide raw CSVs that contain:

- The same feature columns as the original training data
- A label column, whose name you pass via `--label-col`

Label values are mapped to the run's numeric labels automatically, so your
labels must match the classes the run was trained on — you don't need to encode
them yourself.

## GPU Support

GPU is used automatically if available. Disable it with `--cpu-only` (local), or
by omitting `--gpus` (Docker).

## Output

```
artifacts_dir/
├── ...                 # Artifacts produced by train.py
```

## Troubleshooting

### "environment.yml not found"

The code path must contain `environment.yml` (or `runtime_info/environment.yml`).
Check that the run completed and produced a model.

### "Column mismatch"

New data must have the same feature columns and label classes as the original
training data.

## Next Steps

- [Running Inference](inference.md) - Make predictions with trained models
- [Understanding Outputs](outputs.md) - Explore what the agent produces
