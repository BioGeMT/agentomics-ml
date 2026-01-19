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
│   ├── model.joblib
│   └── ...
├── .conda/                      # Conda environment
│   └── envs/<agent_id>_env/
├── iteration_0/                 # Iteration-specific files
│   ├── data_exploration.json
│   ├── data_split.json
│   ├── data_representation.json
│   ├── model_architecture.json
│   └── ...
├── iteration_1/
└── ...
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

## outputs/

Final results after run completion:

```
outputs/<agent_id>/
├── best_run_files/           # Best iteration artifacts
│   ├── inference.py          # Inference script
│   ├── train.py              # Training script
│   ├── model.joblib          # Trained model
│   ├── .conda/               # Complete conda environment
│   ├── iteration_number.txt  # Which iteration was best
│   └── metadata.json         # Model metadata
├── iteration_0/              # All iteration files
├── iteration_1/
├── ...
├── reports/                  # Run reports
│   ├── run_report_iter_0.txt
│   ├── run_report_iter_1.txt
│   └── final_report.txt
├── extras/                   # Additional files
├── logs/                     # Execution logs
└── README.md                 # Run summary
```

## File Descriptions

### Core Files

| File | Description |
|------|-------------|
| `train.py` | Script that trains the model |
| `inference.py` | Script that makes predictions |
| `model.joblib` | Trained model (format varies) |
| `metadata.json` | Task type, classes, configuration |

### Iteration Files

| File | Description |
|------|-------------|
| `data_exploration.json` | Data analysis results |
| `data_split.json` | Train/validation split info |
| `data_representation.json` | Feature encoding scheme |
| `model_architecture.json` | Model configuration |
| `metrics.json` | Evaluation metrics |

### Reports

| File | Description |
|------|-------------|
| `run_report_iter_N.txt` | Summary of iteration N |
| `final_report.txt` | Complete run summary |

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
