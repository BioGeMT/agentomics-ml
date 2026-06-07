# Running Inference

Use trained models to make predictions on new data with `scripts/inference.sh`.

## Basic Usage

```bash
./scripts/inference.sh \
  --agent-dir outputs/<agent_id> \
  --input /path/to/input_folder \
  --output /path/to/predictions.csv
```

## Required Arguments

| Argument | Description |
|----------|-------------|
| `--agent-dir` | Path to completed agent output folder |
| `--input` | Path to an input folder without labels |
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
  --input new_samples/input \
  --output predictions.csv
```

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

Ensure your input folder has the same top-level files and folders as the
training split's `input/` folder. Files inside matching top-level folders may
differ.

### "Model file not found"

Check that `best_iteration_snapshot/` contains the model artifacts. If the agent run failed, there may be no trained model.

### GPU out of memory

Use `--cpu-only` flag or reduce batch size in the inference script.

## Batch Inference

For large datasets, the inference script handles batching automatically. If you need custom batch sizes, modify `best_iteration_snapshot/model_inference/inference.py`.

## Next Steps

- [Understanding Outputs](outputs.md) - Full output structure
- [Re-training Models](training.md) - Train with new data
