# Preparing Datasets

Agentomics-ML works with CSV datasets for classification or regression tasks.

## Quick Setup

Create a folder in `datasets/` with your data:

```
datasets/my_dataset/
├── train.csv              # Required
├── validation.csv         # Optional
├── test.csv               # Optional
├── dataset_description.md # Optional
└── dataset_config.json    # Optional — avoids interactive prompts during dataset preparation
```

## File Requirements

### train.csv (Required)

Your training data with features and a target column.

```csv
feature1,feature2,feature3,target
1.2,3.4,5.6,positive
7.8,9.0,1.2,negative
```

### validation.csv (Optional)

Separate validation data. If not provided, the agent creates a train/validation split from `train.csv`.

### test.csv (Optional)

Hidden test set for final evaluation. The agent never sees this data during training - it's only used to report final metrics.

### dataset_description.md (Optional)

Domain information to help the agent understand your data:

```markdown
# Gene Expression Dataset

This dataset contains RNA-seq expression levels from tumor samples.

## Features
- Columns 1-100: Gene expression values (log2 TPM)
- Samples are from breast cancer patients

## Target
- `class`: tumor subtype (Basal, Her2, LumA, LumB, Normal)

## Notes
- Data is already normalized
- Consider using models that handle high-dimensional data
```

## Dataset Config File (Optional)

Add an optional `dataset_config.json` to your dataset folder to avoid interactive prompts during dataset preparation.

**`task_type` is the most important field** — without it you'll be prompted every time you prepare the dataset.

```json
{
    "task_type": "classification",
    "target_col": "label",
    "positive_class": 1,
    "negative_class": 0
}
```

Fields:

- `task_type` (optional): `"classification"` or `"regression"`; if omitted, you will be prompted during dataset preparation.
- `target_col` (optional): column name to predict; auto-detected if omitted.
- `positive_class` (optional): value that counts as "positive"; only applicable for some binary classification metrics, auto-detected if omitted.
- `negative_class` (optional): value that counts as "negative"; only applicable for some binary classification metrics, auto-detected if omitted.

Include only the fields you need — at minimum just `task_type`. Values from this file take precedence over auto-detection, but CLI flags (`--task-type`, `--target-col`, etc.) override the config file.

## Target Column Detection

The target column is resolved in this order:
1. CLI flag (`--target-col`)
2. `dataset_config.json` (`target_col` field)
3. Auto-detection from common names: `class`, `target`, `label`, `y`
4. Interactive prompt (if running interactively)

If all of the above fail, preparation will raise an error.

## Manual Dataset Preparation

For more control, run preparation separately:

```bash
# Create preparation environment
conda env create -f envs/environment_prepare.yaml
conda activate agentomics-prepare-env

# Prepare datasets
python src/prepare_datasets.py --prepare-all
```

### Preparation Options

```bash
python src/prepare_datasets.py --help
```

Key options:

| Option | Description |
|--------|-------------|
| `--dataset-dir` | Specific dataset to prepare |
| `--task-type` | Specify `classification` or `regression` |
| `--target-col` | Specify target column name |
| `--positive-class` | Define positive class for binary classification |
| `--negative-class` | Define negative class for binary classification |

*Note: already-prepared datasets are skipped on re-runs (preserves `--positive-class`/`--negative-class`). To re-prepare, delete the folder under `prepared_datasets/` (and under `prepared_test_sets/` if a test set was provided) and rerun the preparation script.*

## Prepared Dataset Structure

After preparation, datasets are stored in:

```
prepared_datasets/my_dataset/
├── train.csv              # Training data
├── validation.csv         # Validation data (created if not provided)
├── dataset_description.md # Copied/created description
└── metadata.json          # Task type, classes, etc.

prepared_test_sets/my_dataset/
├── test.csv               # Test data (if provided)
└── test.no_label.csv      # Test data without labels
```

## Example Datasets

Download example datasets:

```bash
./scripts/download_example_datasets.sh
```

## Data Format Tips

### Classification

- Target column should contain class labels (strings or integers)
- Binary: `positive`/`negative`, `1`/`0`, `yes`/`no`
- Multi-class: `class_a`, `class_b`, `class_c`
- Multi-label classification is not supported (use a single label per row)

### Regression

- Target column should contain numeric values
- Select `regression` during preparation or pass `--task-type regression`

### Feature Columns

- Numeric features work best
- Categorical features are supported (encoded automatically)
- Missing values are handled, but clean data performs better

## Common Issues

### "Could not detect target column"

Solution: Add `--target-col your_column_name` to preparation command, or rename your target column to `class`, `target`, `label`, or `y`.

### "Task type required"

Solution (preferred): Add a `dataset_config.json` to your dataset folder with `{"task_type": "classification"}` or `{"task_type": "regression"}`.

Alternative: Pass `--task-type classification` or `--task-type regression` to the preparation command, or run preparation interactively and select when prompted.

## Next Steps

- [Running the Agent](running-agent.md) - Use your prepared dataset
- [Understanding Outputs](outputs.md) - See what the agent produces
