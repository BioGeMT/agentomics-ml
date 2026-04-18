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
  --train-data /path/to/new_train.csv \
  --validation-data /path/to/new_validation.csv \
  --artifacts-dir /path/to/output_artifacts
```

## Required Arguments

| Argument | Description |
|----------|-------------|
| `--agent-dir` | Path to completed agent output folder |
| `--train-data` | Path to new training CSV file |
| `--validation-data` | Path to new validation CSV file |
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
  --train-data datasets/updated_data/train.csv \
  --validation-data datasets/updated_data/validation.csv \
  --artifacts-dir outputs/retrained_model
```

## How It Works

The script:

1. Loads the agent's `model_training/train.py` script from `best_iteration_snapshot/`
2. Uses the agent's conda environment
3. Runs training with the new data
4. Saves artifacts to the specified directory

## Data Format

Your new data files must match the format expected by the agent's training script:

- Same column names as original training data
- Same feature encoding/preprocessing expectations
- Target column with same name and format

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

Run `./run.sh` once to build the Docker image, or use `--local` mode.

### "Agent directory not found"

Ensure the path points to a completed agent output in `outputs/`.

### "Column mismatch"

Your new data must have the same column structure as the original training data.

## Next Steps

- [Running Inference](inference.md) - Make predictions with trained models
- [Understanding Outputs](outputs.md) - Explore what the agent produces
