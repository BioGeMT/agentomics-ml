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
│   │   ├── eval_predictions_validation.csv
│   │   └── output.json
│   └── runtime_info/
│       ├── iteration_metadata.json
│       ├── environment.yml
│       └── environment.tar.gz         # Present in full export mode
├── run/                      # All iterations + shared run state
│   ├── shared/
│   │   ├── config.json
│   │   └── splits/
│   ├── iteration_0/
│   ├── iteration_1/
│   └── ...
├── reports/
│   ├── best_iteration.md
│   ├── best_iteration.pdf
│   ├── markdown/
│   │   ├── run_report_iter_0.md
│   │   ├── run_report_iter_1.md
│   │   └── ...
│   └── pdf/
│       ├── iteration_0.pdf
│       ├── iteration_1.pdf
│       └── plots/
├── logs/                     # Logs and metrics
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
| `runtime_info/environment.yml` | Portable definition of the Conda environment used |
| `runtime_info/environment.tar.gz` | Packed environment for fast container-local restoration in `full` mode |
| `eval_predictions_<split>.csv` | Best-model predictions for a test-prefixed split |
| `eval_predictions_<split>.numeric_labels.csv` | Numeric labels used for that split's metrics |
| `eval_predictions_<split>.metrics.json` | Metrics for that test-prefixed split |

For example, `test/` produces `eval_predictions_test.*`, while
`test_leftout/` produces `eval_predictions_test_leftout.*`.

### Using the Best Model

```bash
agentomics-inference --agent-dir outputs/<agent_id> --input data/input --output predictions.csv
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

`reports/best_iteration.md` and `reports/best_iteration.pdf` are copies of the
selected iteration's reports and provide stable paths to the final result.

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
`agentomics-run --list-metrics` for the current list.

## Where Outputs Are Stored

The agent writes directly to the run workspace as it works — there is no
separate staging area or temporary volume:

- The host workspace defaults to `outputs/<agent_id>/` and is mounted at
  `/workspace` in the container.

## W&B Logging

If W&B is configured, you'll also find:

- Experiment tracking at wandb.ai
- Final held-out metrics under `<split>/<metric>` for every test-prefixed split
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

## Next Steps

- [Running Inference](inference.md) - Use your trained model
- [Workspace Structure](../reference/workspace-structure.md) - Detailed workspace layout
- [Metrics](../reference/metrics.md) - All available metrics
