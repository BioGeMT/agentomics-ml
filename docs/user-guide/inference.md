# Running Inference

Use a trained model to predict on new data with `agentomics-inference`. It can
also compute metrics against true labels, and evaluate every archived iteration.

## Requirements

- A completed agent output directory (a finished `outputs/<agent_id>` run)
- The Agentomics package (`python3 -m pip install agentomics`)
- [Docker](https://docs.docker.com/get-docker/)

## Basic Usage

```bash
agentomics-inference \
  --agent-dir outputs/<agent_id> \
  --input /path/to/data.csv \
  --output /path/to/predictions.csv
```

`--input` accepts either a **CSV file** or a **contract split folder** (`input/`
+ optional `labels.csv`). See [Input Data Format](#input-data-format).

## Arguments

### Required

| Argument | Description |
|----------|-------------|
| `--agent-dir` | Path to completed agent output folder |
| `--input` | A CSV file, or a contract split folder (`input/` + optional `labels.csv`) |
| `--output` | Path where predictions will be saved |

### Optional

| Argument | Description |
|----------|-------------|
| `--label-col` | Label column in the input **CSV** — when set, metrics are computed against it and written to `<output>.metrics.json`. Ignored for a split folder; there, metrics run only if the folder has a `labels.csv` |
| `--iteration-dir` | Iteration directory to use, relative to `--agent-dir` (default: `best_iteration_snapshot`) |
| `--all-iterations` | Run inference for every `run/iteration_N` against `--input` (see below) |
| `--wandb-prefix` | W&B metric namespace when labels are available (default: input file or directory name) |
| `--cpu-only` | Run without GPU |
| `--image` | Docker image to use (default: `biogemt/agentomics:<installed-package-version>`; use this option for an explicit override) |
| `--help` | Show help message |

## Docker Execution

Docker is the default execution mode. The command mounts the agent directory,
input, and output directory into the configured image automatically:

```bash
agentomics-inference \
  --agent-dir outputs/my_run_1 \
  --input data/new.csv \
  --output data/predictions.csv \
  --label-col label
```

Predictions (and metrics, when `--label-col` is set) appear in `./data`. Add
`--cpu-only` to run without GPU access. The model environment is restored inside
the container from `runtime_info/environment.tar.gz`, or rebuilt from
`runtime_info/environment.yml` when no usable archive is available.

Use `--image <name>` to select another image explicitly, such as a local
development build.

## Input Data Format

`--input` can be either of two shapes:

**A CSV file** — the common case. It is converted into a contract split before
inference. Provide the same feature columns the model was trained on. If the CSV
also has a label column, pass `--label-col <name>` to compute metrics (the label
is split out, not fed to the model); omit it for prediction-only.

```csv
feature1,feature2,feature3,label
1.2,3.4,5.6,positive
7.8,9.0,1.2,negative
```

**A contract split folder** — `input/` (+ optional `labels.csv`), matching the
structure of the training split's `input/`:

```text
new_samples/
├── input/
│   └── data.csv      # feature files (same top-level structure as training)
└── labels.csv        # optional (id,label or id,numeric_label) -> enables metrics
```

For both shapes, `input/` must match the training split's top-level structure and
carry the sample IDs the generated `inference.py` expects (a CSV without an `id`
column gets sequential IDs assigned during conversion). Metrics are produced when
labels are available — via `--label-col` (CSV) or a present `labels.csv` (folder).

## Output Format

The generated `inference.py` must preserve input sample IDs and write a prediction for every sample. Classification runs produce `id`, `prediction`, and probability columns when probabilities are available. Regression runs produce `id` and `prediction`. Additional columns are run-specific.

## Computing Metrics

Pass `--label-col` to score predictions against the true labels in `--input`.
Metrics are computed by the bundled evaluator using the run's task type and
label mapping, aligned to predictions by `id`, and written to
`<output>.metrics.json` next to `--output`.

```bash
agentomics-inference \
  --agent-dir outputs/<agent_id> \
  --input labeled_data.csv \
  --output predictions.csv \
  --label-col label
# -> predictions.csv and predictions.metrics.json
```

If `WANDB_API_KEY`, `WANDB_PROJECT_NAME`, and `WANDB_ENTITY` are available and
the original run has a W&B run ID, those metrics are also appended to that run.
Use `--wandb-prefix test_external` to log keys such as
`test_external/AUROC`. Without an explicit prefix, the input file or directory
name is used. W&B failures warn and do not fail inference.

## Evaluating Every Iteration

`--all-iterations` runs inference on each `run/iteration_N` snapshot in turn:

```bash
agentomics-inference \
  --agent-dir outputs/<agent_id> \
  --input labeled_data.csv \
  --output preds/out.csv \
  --label-col label \
  --all-iterations
```

The command reports the number of successful and failed iterations. Individual
failures do not stop the remaining iterations, but the command exits with an
error if every iteration fails.

Per-iteration predictions are written to `preds/<iteration>_predictions.csv`
(and `<iteration>_predictions.metrics.json` when `--label-col` is set). Each
iteration's environment is rebuilt and removed in turn, and a failing iteration
warns and continues. W&B keys include an additional `/iteration_N` namespace
when `--all-iterations` is used.

## What's in best_iteration_snapshot

```
outputs/<agent_id>/best_iteration_snapshot/
├── model_inference/
│   └── inference.py        # Inference script
├── model_training/
│   ├── train.py            # Training script
│   └── training_artifacts/ # Model files (format varies)
├── runtime_info/
│   ├── iteration_metadata.json
│   ├── environment.yml
│   └── environment.tar.gz  # Present with --conda-export-mode full
└── ...                     # Other artifacts (tokenizers, etc.)
```

## GPU Support

GPU is used automatically when the container can see it. To force CPU, add
`--cpu-only`.

## Troubleshooting

### "environment.yml not found"

The iteration directory must contain `runtime_info/environment.yml`.
Check that the run completed and produced a model.

### "Column mismatch"

Ensure your input folder has the same top-level files and folders as the
training split's `input/` folder. Files inside matching top-level folders may
differ.

### "Model file not found"

Check that `best_iteration_snapshot/` contains the model artifacts. If the agent run failed, there may be no trained model.

### GPU out of memory

Use `--cpu-only`, or reduce the batch size in
`best_iteration_snapshot/model_inference/inference.py`.

## Next Steps

- [Understanding Outputs](outputs.md) - Full output structure
- [Re-training Models](training.md) - Train with new data
