# Workspace Structure

How Agentomics-ML organizes files during and after execution.

A run's files live in a single workspace directory:

- **Local mode:** `outputs/<agent_id>/`
- **Docker mode:** the host directory you mount at `/workspace`

## Repository Layout

```text
agentomics-ml/
├── datasets/                 # Public train/validation datasets
├── test_datasets/            # Hidden test datasets
└── outputs/                  # Final results

../workspace/runs/<agent_id>/ # Local-mode active workspace
├── run/                      # Current run files
├── best_iteration_snapshot/  # Best iteration snapshot
├── reports/                  # Iteration reports
├── extras/                   # Logs and extra artifacts
└── fallbacks/                # Reserved recovery area
```

## datasets/

Public datasets use split folders:

```text
datasets/my_dataset/
├── train/
│   ├── input/
│   └── labels.csv
├── validation/             # Optional
│   ├── input/
│   └── labels.csv
├── supplementary/          # Optional: dataset-level source materials
│   └── README.md           # Optional: describes the supplementary materials
├── metadata.json           # Optional if task type is supplied at preparation
└── dataset_description.md  # Optional domain information
```

Hidden test data uses a matching separate root:

```text
test_datasets/my_dataset/
└── test/
    ├── input/
    └── labels.csv
```

Each unprepared `labels.csv` must include `id` and `label` columns. Preparing a
dataset under `datasets/` rewrites its split labels in place with `id` and
`numeric_label`, then writes `metadata.json` with `"prepared": true`. Only
`train` and `validation` are supported under `datasets/`; only `test` is
supported under `test_datasets/`. The `input/` interface is recorded at
preparation time, must match across all splits, and must not be modified during
a run. `test_datasets/` is optional and is **not** prepared or evaluated by the
run (see below).

After preparation, the public dataset directory is the agent-facing dataset:

```text
datasets/my_dataset/
├── train/
│   ├── input/
│   └── labels.csv          # id,numeric_label
├── validation/
│   ├── input/
│   └── labels.csv          # id,numeric_label
├── supplementary/          # Dataset-level source materials, if provided
├── dataset_description.md
└── metadata.json           # includes "prepared": true
```

`test_datasets/` holds optional held-out data the agent never sees during a run.
The run no longer prepares or evaluates it automatically; to score the finished
model on it, run `scripts/inference.sh` (or `scripts/train.sh`) against the split
afterward — those prepare it on the fly. Its `labels.csv` stays in raw `id,label`
form:

```text
test_datasets/my_dataset/
└── test/
    ├── input/
    └── labels.csv          # id,label
```

## Active Workspace

Active execution area. In local mode this is `../workspace/runs/<agent_id>/`; in Docker mode it is the temporary `/workspace` volume.

### run/

Current run working directory:

```text
<workspace_root>/run/
├── shared/
│   ├── .conda/                  # Shared Conda environment
│   ├── config.json
│   ├── environment.yml
│   └── splits/
├── current_iteration/
│   ├── current_step/            # Active step workspace
│   └── runtime_info/
├── iteration_0/                 # Archived iteration
├── iteration_1/
└── ...
```

### best_iteration_snapshot/

Best iteration snapshot:

```text
<workspace_root>/best_iteration_snapshot/
├── model_training/
│   ├── train.py
│   └── training_artifacts/
├── model_inference/
│   └── inference.py
├── runtime_info/
├── environment.yml
└── .conda/
```

Updated whenever a new best iteration is achieved.

### fallbacks/

Reserved recovery area:

```text
<workspace_root>/fallbacks/
```

This directory may be empty for normal runs.

### run/shared/splits/

Versioned train/validation split folders:

```text
<workspace_root>/run/shared/splits/
└── split_0/
    ├── train/
    │   ├── input/
    │   └── labels.csv
    ├── validation/
    │   ├── input/
    │   └── labels.csv
    └── mini_train/
        ├── input/
        └── labels.csv
```

Each time the agent changes the train/validation split, a new `split_<n>/`
folder is created. Iteration outputs record which split version they used.
The `input/` structure must match the original recorded structure across all
splits and must not be modified. The `mini_train/` folder is a small subset
of training data (at most 100 samples) used for quick script validation.

### reports/

Iteration reports are written here during runs. These are copied to
`outputs/<agent_id>/reports/` after completion.

### extras/

Logs and auxiliary artifacts (metrics, run logs) are stored here and copied to
`outputs/<agent_id>/extras/`.

## outputs/

Final results after run completion:

```text
outputs/<agent_id>/
├── best_iteration_snapshot/           # Best iteration artifacts
│   ├── model_training/
│   │   ├── train.py
│   │   └── training_artifacts/
│   ├── model_inference/
│   │   └── inference.py
│   ├── runtime_info/
│   ├── environment.yml
│   └── .conda/
├── run/                      # All iterations + data splits
│   ├── shared/
│   │   ├── config.json
│   │   └── splits/
│   │       └── split_0/
│   │           ├── train/
│   │           │   ├── input/
│   │           │   └── labels.csv
│   │           ├── validation/
│   │           │   ├── input/
│   │           │   └── labels.csv
│   │           └── mini_train/
│   │               ├── input/
│   │               └── labels.csv
│   ├── iteration_0/
│   ├── iteration_1/
│   └── ...
├── reports/
│   ├── markdown/
│   │   ├── run_report_iter_0.md
│   │   └── ...
│   └── pdf/
│       ├── iteration_0.pdf
│       └── plots/
├── extras/                     # Logs and auxiliary artifacts
├── fallbacks/                  # Reserved recovery area (often empty)
└── README.md                   # Run summary
```

### run/

The working directory. `shared/` holds the run config, the shared conda
environment, the prepared dataset copy, and the train/validation splits.
`current_iteration/` is the active iteration while the run is in progress;
completed iterations are archived as `iteration_N/`.

### best_iteration_snapshot/

The best iteration's exported model, scripts, and environment — updated whenever
a new best iteration is achieved. Use it for inference and re-training.

### reports/ and extras/

Per-iteration markdown and PDF reports, plus logs and auxiliary artifacts,
written directly to the workspace as the run progresses.

## File Notes

Iteration contents and artifact names can vary by run. Use `<step_id>/output.json`
inside each archived iteration or best iteration snapshot as the structured source of
truth for step outputs. Use `outputs/<agent_id>/README.md` for the most accurate
per-run details.

## Cleanup

```bash
# Remove a specific run
rm -rf outputs/<agent_id>

# Clean everything
rm -rf outputs/* prepared_datasets/*
```

## Docker Mode

The repository is baked into the image; you mount your datasets at
`/repository/datasets` and a host directory at `/workspace` to receive the run's
output. The agent runs entirely inside the container, isolating execution from
the host. See [Installation](../getting-started/installation.md).

## Related

- [Understanding Outputs](../user-guide/outputs.md) - Using output files
- [Running Inference](../user-guide/inference.md) - Using trained models
