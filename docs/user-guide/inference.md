# Running Inference

Use trained models to make predictions on new data with `scripts/inference.sh`.

## Basic Usage

```bash
./scripts/inference.sh \
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
| `--code-path` | Relative path inside `--agent-dir` containing generated code; defaults to `best_iteration_snapshot` |
| `--remove-conda-env` | Remove the generated inference Conda environment after the run |
| `--help` | Show help message |

## Example

```bash
./scripts/inference.sh \
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

The generated `inference.py` must preserve the input `id` column and write a prediction for every row. Classification runs produce `prediction` plus `probability_<class_id>` columns. Regression runs produce `prediction`. Additional columns are run-specific.

## Docker vs Local Mode

```bash
# Docker mode (default, recommended)
./scripts/inference.sh --agent-dir outputs/my_agent ...

# Local mode
./scripts/inference.sh --local --agent-dir outputs/my_agent ...
```

## GPU Support

GPU is used automatically if available:

```bash
# Use GPU (default)
./scripts/inference.sh --agent-dir outputs/my_agent ...

# CPU only
./scripts/inference.sh --cpu-only --agent-dir outputs/my_agent ...
```

## What's in best_iteration_snapshot

```
outputs/<agent_id>/best_iteration_snapshot/
├── model_inference/
│   └── inference.py       # Inference script
├── model_training/
│   ├── train.py           # Training script
│   └── training_artifacts/ # Model files (format varies)
├── runtime_info/
│   └── iteration_metadata.json
├── environment.yml
├── .conda/                # Conda environment
└── ...                    # Other artifacts (tokenizers, etc.)
```

## Troubleshooting

### "Docker image not found"

The script first looks for a local `agentomics_img`, then for the matching pre-built Docker Hub image, and builds locally if needed. Use `--local` to avoid Docker.

### "Column mismatch"

Ensure your input CSV has the same feature columns as the training data (minus the target column).

### "Model file not found"

Check that `best_iteration_snapshot/` contains the model artifacts. If the agent run failed, there may be no trained model.

### GPU out of memory

Use `--cpu-only` flag or reduce batch size in the inference script.

## Batch Inference

For large datasets, the inference script handles batching automatically. If you need custom batch sizes, modify `best_iteration_snapshot/model_inference/inference.py`.

## Next Steps

- [Understanding Outputs](outputs.md) - Full output structure
- [Re-training Models](training.md) - Train with new data
