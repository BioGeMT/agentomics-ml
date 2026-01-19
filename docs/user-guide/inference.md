# Running Inference

Use trained models to make predictions on new data with `inference.sh`.

## Basic Usage

```bash
./inference.sh \
  --agent-dir outputs/<agent_id> \
  --input /path/to/new_data.csv \
  --output /path/to/predictions.csv
```

## Required Arguments

| Argument | Description |
|----------|-------------|
| `--agent-dir` | Path to completed agent output folder |
| `--input` | Path to input CSV file (without labels) |
| `--output` | Path where predictions will be saved |

## Optional Arguments

| Argument | Description |
|----------|-------------|
| `--cpu-only` | Run without GPU |
| `--local` | Run locally without Docker |
| `--help` | Show help message |

## Example

```bash
./inference.sh \
  --agent-dir outputs/enchanted_fixing_reigned \
  --input new_samples.csv \
  --output predictions.csv
```

## Input Data Format

Your input file should:

- Be a CSV file
- Have the same feature columns as training data
- **Not** include the target/label column

Example:
```csv
feature1,feature2,feature3
1.2,3.4,5.6
7.8,9.0,1.2
```

## Output Format

Predictions are saved as a CSV with a single column:

```csv
predictions
positive
negative
positive
```

The order matches your input data rows.

## Docker vs Local Mode

```bash
# Docker mode (default, recommended)
./inference.sh --agent-dir outputs/my_agent ...

# Local mode
./inference.sh --local --agent-dir outputs/my_agent ...
```

## GPU Support

GPU is used automatically if available:

```bash
# Use GPU (default)
./inference.sh --agent-dir outputs/my_agent ...

# CPU only
./inference.sh --cpu-only --agent-dir outputs/my_agent ...
```

## Using the Inference Script Directly

For more control, use the agent's inference script directly:

```bash
# Navigate to agent's best run
cd outputs/<agent_id>/best_run_files

# Activate the environment
conda activate .conda/envs/<agent_id>_env

# Run inference
python inference.py --input /path/to/data.csv --output /path/to/predictions.csv
```

## What's in best_run_files

```
outputs/<agent_id>/best_run_files/
├── inference.py           # Inference script
├── train.py               # Training script
├── model.joblib           # Trained model (or other format)
├── .conda/                # Conda environment
└── ...                    # Other artifacts (tokenizers, etc.)
```

## Troubleshooting

### "Docker image not found"

Run `./run.sh` once to build the Docker image, or use `--local` mode.

### "Column mismatch"

Ensure your input CSV has the same feature columns as the training data (minus the target column).

### "Model file not found"

Check that `best_run_files/` contains the model artifacts. If the agent run failed, there may be no trained model.

### GPU out of memory

Use `--cpu-only` flag or reduce batch size in the inference script.

## Batch Inference

For large datasets, the inference script handles batching automatically. If you need custom batch sizes, modify the `inference.py` script in `best_run_files/`.

## Next Steps

- [Understanding Outputs](outputs.md) - Full output structure
- [Re-training Models](training.md) - Train with new data
