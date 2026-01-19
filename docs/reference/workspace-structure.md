# Workspace Structure

How Agentomics-ML organizes files during and after execution.

## Directory Overview

```
agentomics-ml/
├── datasets/                 # Raw input datasets
├── prepared_datasets/        # Prepared training data
├── prepared_test_sets/       # Prepared test data (hidden)
├── workspace/                # Active execution workspace
│   ├── runs/                 # Current run files
│   ├── snapshots/            # Best iteration snapshots
│   ├── reports/              # Iteration reports
│   ├── extras/               # Logs and extra artifacts
│   └── fallbacks/            # Backup for recovery
└── outputs/                  # Final results
```

## datasets/

Your raw input datasets:

```
datasets/my_dataset/
├── train.csv              # Training data (required)
├── validation.csv         # Validation data (optional)
├── test.csv               # Test data (optional)
└── dataset_description.md # Domain info (optional)
```

## prepared_datasets/

After preparation, datasets are formatted for the agent:

```
prepared_datasets/my_dataset/
├── train.csv              # Processed training data
├── validation.csv         # Processed validation data
├── train.no_label.csv     # Training data without labels
├── validation.no_label.csv
├── dataset_description.md # Copied/created description
└── metadata.json          # Task info (type, classes, etc.)
```

## prepared_test_sets/

Test data is separated to ensure it stays hidden:

```
prepared_test_sets/my_dataset/
├── test.csv               # Test data with labels
└── test.no_label.csv      # Test data without labels
```

The agent never sees files in this directory during training.

## workspace/

Active execution area:

### workspace/runs/

Current run working directory:

```
workspace/runs/<agent_id>/
├── train.csv                    # Copy of prepared data
├── validation.csv
├── dataset_description.md
├── train.py                     # Generated training script
├── inference.py                 # Generated inference script
├── training_artifacts/          # Model and artifacts
├── .conda/                      # Conda environment
└── iteration_0/                 # Iteration-specific snapshot
```

### workspace/snapshots/

Best iteration backup:

```
workspace/snapshots/<agent_id>/
├── train.py
├── inference.py
├── training_artifacts/
└── .conda/
```

Updated whenever a new best iteration is achieved.

### workspace/fallbacks/

Recovery backup for split changes:

```
workspace/fallbacks/<agent_id>/
├── train.csv
├── validation.csv
└── split_fingerprint.json
```

Used to restore data if a split change causes issues.

### workspace/reports/

Iteration reports are written here during runs. These are copied to
`outputs/<agent_id>/reports/` after completion.

### workspace/extras/

Logs and auxiliary artifacts (metrics, run logs) are stored here and copied to
`outputs/<agent_id>/extras/`.

## outputs/

Final results after run completion:

```
outputs/<agent_id>/
├── best_run_files/           # Best iteration artifacts
│   ├── inference.py          # Inference script
│   ├── train.py              # Training script
│   ├── training_artifacts/   # Model and artifacts
│   ├── validation_metrics.txt
│   ├── train_metrics.txt
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
├── extras/                   # Additional files and logs
└── README.md                 # Run summary
```

## File Notes

Iteration contents and artifact names can vary by run. Use
`outputs/<agent_id>/README.md` for the most accurate per-run details.

## Cleanup

### Remove Specific Run

```bash
rm -rf outputs/<agent_id>
```

### Clean Workspace

```bash
rm -rf workspace/runs/*
rm -rf workspace/snapshots/*
rm -rf workspace/fallbacks/*
```

### Clean Everything

```bash
rm -rf outputs/*
rm -rf workspace/*
rm -rf prepared_datasets/*
rm -rf prepared_test_sets/*
```

## Docker Volumes

In Docker mode, workspace is mounted as a volume:

- Code repository: Read-only
- Workspace: Read-write
- Outputs: Read-write

This isolates agent execution from the host system.

## Related

- [Understanding Outputs](../user-guide/outputs.md) - Using output files
- [Running Inference](../user-guide/inference.md) - Using trained models
