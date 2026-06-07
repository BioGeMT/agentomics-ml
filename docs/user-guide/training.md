# Re-training Models

After the agent completes a run, you can re-train the model with new data using the `scripts/train.sh` script.

## When to Use

- Train on updated or expanded datasets
- Fine-tune with additional samples
- Reproduce training with different data splits

## Basic Usage

```bash
./scripts/train.sh \
  --agent-dir outputs/<agent_id> \
  --train-data /path/to/train \
  --validation-data /path/to/validation \
  --artifacts-dir /path/to/output_artifacts
```

## Required Arguments

| Argument | Description |
|----------|-------------|
| `--agent-dir` | Path to completed agent output folder |
| `--train-data` | Path to a training split folder with `input/` and `labels.csv` |
| `--validation-data` | Path to a validation split folder with `input/` and `labels.csv` |
| `--artifacts-dir` | Where to save new training artifacts |

## Optional Arguments

| Argument | Description |
|----------|-------------|
| `--cpu-only` | Run without GPU |
| `--local` | Run locally without Docker |
| `--help` | Show help message |

## Example

```bash
# Re-train using new data
./scripts/train.sh \
  --agent-dir outputs/enchanted_fixing_reigned \
  --train-data datasets/updated_data/train \
  --validation-data datasets/updated_data/validation \
  --artifacts-dir outputs/retrained_model
```

## How It Works

The script:

1. Loads the agent's `model_training/train.py` script from `best_iteration_snapshot/`
2. Uses the agent's conda environment
3. Runs training with the new data
4. Saves artifacts to the specified directory

## Data Format

Your new split folders must match the format expected by the agent's training script:

- Same `input/` structure as the original training data
- `labels.csv` with `id` and `numeric_label`
- Matching IDs between input files and labels

## Output

After training completes, you'll find:

```
artifacts_dir/
├── ...                 # Artifacts produced by train.py
```

## Docker vs Local Mode

By default, training runs in Docker for isolation. Use `--local` for direct execution:

```bash
# Docker mode (default)
./scripts/train.sh --agent-dir outputs/my_agent ...

# Local mode
./scripts/train.sh --local --agent-dir outputs/my_agent ...
```

## GPU Support

GPU is used automatically if available. To disable:

```bash
./scripts/train.sh --cpu-only --agent-dir outputs/my_agent ...
```

## Troubleshooting

### "Docker image not found"

The script first looks for a local `agentomics_img`, then for the matching pre-built Docker Hub image, and builds locally if needed. Use `--local` to avoid Docker.

### "Agent directory not found"

Ensure the path points to a completed agent output in `outputs/`.

### "Column mismatch"

Your new data must have the same column structure as the original training data.

## Next Steps

- [Running Inference](inference.md) - Make predictions with trained models
- [Understanding Outputs](outputs.md) - Explore what the agent produces
