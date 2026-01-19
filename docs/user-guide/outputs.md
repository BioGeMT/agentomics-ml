# Understanding Outputs

After a run completes, results are saved to `outputs/<agent_id>/`.

## Output Structure

```
outputs/<agent_id>/
├── best_run_files/           # Best iteration artifacts
│   ├── inference.py          # Inference script
│   ├── train.py              # Training script
│   ├── model.joblib          # Trained model
│   ├── .conda/               # Conda environment
│   ├── iteration_number.txt  # Which iteration was best
│   └── metadata.json         # Model metadata
├── iteration_0/              # First iteration files
├── iteration_1/              # Second iteration files
├── ...
├── reports/                  # Run reports
│   ├── run_report_iter_0.txt
│   ├── run_report_iter_1.txt
│   └── final_report.txt
├── extras/                   # Additional files
├── logs/                     # Execution logs
└── README.md                 # Run summary
```

## best_run_files

The most important directory - contains the best-performing iteration's artifacts.

| File | Description |
|------|-------------|
| `inference.py` | Script to run predictions |
| `train.py` | Script that trained the model |
| `model.joblib` | Trained model (format varies) |
| `.conda/` | Complete conda environment |
| `metadata.json` | Training configuration |
| `iteration_number.txt` | Which iteration this came from |

### Using the Best Model

```bash
# Run inference
./inference.sh --agent-dir outputs/<agent_id> --input data.csv --output predictions.csv

# Or directly
cd outputs/<agent_id>/best_run_files
conda activate .conda/envs/<agent_id>_env
python inference.py --input data.csv --output predictions.csv
```

## Iteration Directories

Each iteration's files are preserved:

```
iteration_N/
├── data_exploration.json     # Data analysis results
├── data_split.json           # Train/val split info
├── data_representation.json  # Feature encoding
├── model_architecture.json   # Model configuration
├── train.py                  # Training script
├── inference.py              # Inference script
├── training_artifacts/       # Model and artifacts
└── metrics.json              # Validation metrics
```

## Reports

### Iteration Reports

`reports/run_report_iter_N.txt` - Summary of each iteration:

- Data exploration findings
- Model architecture chosen
- Training details
- Validation metrics

### Final Report

`reports/final_report.txt` - Complete run summary:

- Best iteration and why
- All iteration metrics comparison
- Test set results (if test data provided)
- Recommendations

## Metrics

Metrics are tracked for each iteration:

**Classification:**

- ACC (Accuracy)
- AUROC (Area Under ROC Curve)
- AUPRC (Area Under Precision-Recall Curve)
- F1, Precision, Recall

**Regression:**

- MSE, RMSE (Mean Squared Error)
- MAE (Mean Absolute Error)
- R2 (R-squared)
- Pearson Correlation

## Workspace Structure

During execution, the agent uses a workspace:

```
workspace/
├── runs/<agent_id>/         # Active run directory
├── snapshots/<agent_id>/    # Best iteration snapshot
└── fallbacks/<agent_id>/    # Backup for recovery
```

After completion, everything is copied to `outputs/`.

## W&B Logging

If W&B is configured, you'll also find:

- Experiment tracking at wandb.ai
- Agent traces with Weave
- Metric plots and comparisons
- Artifact versioning

See [Environment Variables](../configuration/environment.md) for W&B setup.

## Reproducing Results

To reproduce a run:

1. Use the same dataset
2. Use the same model and parameters
3. Set the same random seed (if applicable)

The `train.py` and `inference.py` scripts contain all logic needed to reproduce the model.

## Cleaning Up

Remove old runs:

```bash
# Remove specific run
rm -rf outputs/<agent_id>

# Remove all runs (careful!)
rm -rf outputs/*
```

The workspace is cleaned automatically, but you can manually clean:

```bash
rm -rf workspace/runs/*
rm -rf workspace/snapshots/*
```

## Next Steps

- [Running Inference](inference.md) - Use your trained model
- [Workspace Structure](../reference/workspace-structure.md) - Detailed workspace layout
- [Metrics](../reference/metrics.md) - All available metrics
