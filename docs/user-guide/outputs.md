# Understanding Outputs

After a run completes, results are saved to `outputs/<agent_id>/`.

## Output Structure

```
outputs/<agent_id>/
├── best_iteration_snapshot/           # Best iteration snapshot
│   ├── model_training/
│   │   ├── train.py          # Training script
│   │   └── training_artifacts/
│   ├── model_inference/
│   │   └── inference.py      # Inference script
│   ├── validation_evaluation/
│   │   ├── eval_predictions_train.csv
│   │   └── eval_predictions_validation.csv
│   ├── runtime_info/
│   │   └── iteration_metadata.json
│   ├── environment.yml
│   └── .conda/
├── run/                      # All iterations + shared run state
│   ├── shared/
│   │   ├── config.json
│   │   └── splits/
│   ├── iteration_0/
│   ├── iteration_1/
│   └── ...
├── reports/
│   ├── markdown/
│   │   ├── run_report_iter_0.md
│   │   ├── run_report_iter_1.md
│   │   └── ...
│   └── pdf/
│       ├── iteration_0.pdf
│       ├── iteration_1.pdf
│       └── plots/
├── extras/                   # Additional files
└── README.md                 # Run summary
```

## best_iteration_snapshot

The most important directory - contains the best-performing iteration's artifacts.

| File | Description |
|------|-------------|
| `model_inference/inference.py` | Script to run predictions |
| `model_training/train.py` | Script that trained the model |
| `model_training/training_artifacts/` | Trained model files (format varies) |
| `runtime_info/iteration_metadata.json` | Which iteration produced the snapshot |
| `environment.yml` | Export of the conda env used |
| `.conda/` | Bundled Conda environment for execution |

### Using the Best Model

```bash
./scripts/inference.sh --agent-dir outputs/<agent_id> --input data.csv --output predictions.csv
```

## Iteration Directories

Each iteration's files are preserved under `run/iteration_N/`:

```
run/iteration_N/
├── model_training/
│   ├── train.py
│   └── training_artifacts/
├── model_inference/
│   └── inference.py
├── runtime_info/
│   ├── environment.yml
│   ├── iteration_metadata.json
│   └── iteration_state.json
└── ...                       # Other iteration artifacts
```

## Reports

### Iteration Reports

`reports/markdown/run_report_iter_N.md` - Summary of each iteration:

- Data exploration findings
- Model architecture chosen
- Training details
- Validation metrics

### PDF Reports

`reports/pdf/iteration_N.pdf` - PDF report per iteration, plus plots in `reports/pdf/plots/`.

## Metrics

Metrics are tracked for each iteration:

Metrics depend on the selected validation metric and task type. See
`./run.sh --list-metrics` for the current list.

## Workspace Structure

During execution, the agent uses a workspace:

```
workspace/
├── run/                     # Active run directory
├── best_iteration_snapshot/    # Best iteration snapshot
├── reports/                 # Iteration reports
├── extras/                  # Logs and metrics
└── fallbacks/               # Backup for recovery
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

The `model_training/train.py` and `model_inference/inference.py` scripts contain all logic needed to reproduce the model.

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
rm -rf workspace/run/*
rm -rf workspace/best_iteration_snapshot/*
```

## Next Steps

- [Running Inference](inference.md) - Use your trained model
- [Workspace Structure](../reference/workspace-structure.md) - Detailed workspace layout
- [Metrics](../reference/metrics.md) - All available metrics
