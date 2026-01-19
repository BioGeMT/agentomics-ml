# Understanding Outputs

After a run completes, results are saved to `outputs/<agent_id>/`.

## Output Structure

```
outputs/<agent_id>/
├── best_run_files/           # Best iteration artifacts
│   ├── train.py              # Training script
│   ├── inference.py          # Inference script
│   ├── training_artifacts/   # Model and artifacts
│   ├── validation_metrics.txt
│   ├── train_metrics.txt
│   ├── eval_predictions_train.csv
│   ├── eval_predictions_validation.csv
│   ├── structured_outputs.txt
│   ├── config.json
│   ├── environment.yml
│   └── iteration_number.txt  # Which iteration was best
├── run_files/                # All iterations + data splits
│   ├── train.csv
│   ├── validation.csv
│   ├── iteration_0/
│   ├── iteration_1/
│   └── ...
├── reports/                  # Run reports
│   ├── run_report_iter_0.md
│   ├── run_report_iter_1.md
│   └── ...
├── pdf_reports/              # PDF versions + plots
│   ├── iteration_0.pdf
│   ├── iteration_1.pdf
│   └── plots/
├── extras/                   # Additional files
└── README.md                 # Run summary
```

## best_run_files

The most important directory - contains the best-performing iteration's artifacts.

| File | Description |
|------|-------------|
| `inference.py` | Script to run predictions |
| `train.py` | Script that trained the model |
| `training_artifacts/` | Trained model files (format varies) |
| `config.json` | Run configuration snapshot |
| `environment.yml` | Export of the conda env used |
| `iteration_number.txt` | Which iteration this came from |

### Using the Best Model

```bash
# Run inference
./inference.sh --agent-dir outputs/<agent_id> --input data.csv --output predictions.csv

# Or directly (local mode)
conda run -p outputs/<agent_id>/best_run_files/.conda/envs/<agent_id>_env \
  python outputs/<agent_id>/best_run_files/inference.py \
  --input data.csv --output predictions.csv
```

## Iteration Directories

Each iteration's files are preserved under `run_files/iteration_N/`:

```
run_files/iteration_N/
├── train.py                  # Training script
├── inference.py              # Inference script
├── training_artifacts/       # Model and artifacts
├── validation_metrics.txt
└── ...                       # Other iteration artifacts
```

## Reports

### Iteration Reports

`reports/run_report_iter_N.md` - Summary of each iteration:

- Data exploration findings
- Model architecture chosen
- Training details
- Validation metrics

### PDF Reports

`pdf_reports/iteration_N.pdf` - PDF report per iteration, plus plots in `pdf_reports/plots/`.

## Metrics

Metrics are tracked for each iteration:

Metrics depend on the selected validation metric and task type. See
`./run.sh --list-metrics` for the current list.

## Workspace Structure

During execution, the agent uses a workspace:

```
workspace/
├── runs/<agent_id>/         # Active run directory
├── snapshots/<agent_id>/    # Best iteration snapshot
├── reports/                 # Iteration reports
├── extras/                  # Logs and metrics
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

In Docker mode, the temporary workspace volume is removed after a run. In local
mode, you can manually clean:

```bash
rm -rf workspace/runs/*
rm -rf workspace/snapshots/*
```

## Next Steps

- [Running Inference](inference.md) - Use your trained model
- [Workspace Structure](../reference/workspace-structure.md) - Detailed workspace layout
- [Metrics](../reference/metrics.md) - All available metrics
