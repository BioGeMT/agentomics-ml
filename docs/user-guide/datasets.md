# Preparing Datasets

Agentomics-ML works with CSV datasets for classification or regression tasks.

## Quick Setup

Create a folder in `datasets/` with your data:

```
datasets/my_dataset/
├── train.csv              # Required
├── validation.csv         # Optional
├── test.csv               # Optional
└── dataset_description.md # Optional
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

## Target Column Detection

The agent auto-detects the target column from common names:

- `class`
- `target`
- `label`
- `y`

If auto-detection fails, you'll be prompted to select the target column during preparation.

For non-interactive preparation, pass `--target-col` to avoid prompts.

## Manual Dataset Preparation

For more control, run preparation separately:

```bash
# Create preparation environment
conda env create -f environment_prepare.yaml
conda activate agentomics-prepare-env

# Prepare datasets
agentomics-prepare-datasets --prepare-all
```

### Preparation Options

```bash
agentomics-prepare-datasets --help
```

Key options:

| Option | Description |
|--------|-------------|
| `--dataset-dir` | Specific dataset to prepare |
| `--task-type` | Force `classification` or `regression` |
| `--target-col` | Specify target column name |
| `--positive-class` | Define positive class for binary classification |
| `--negative-class` | Define negative class for binary classification |

## Prepared Dataset Structure

After preparation, datasets are stored in:

```
prepared_datasets/my_dataset/
├── train.csv              # Training data
├── validation.csv         # Validation data (created if not provided)
├── train.no_label.csv     # Training data without labels (for inference)
├── validation.no_label.csv
├── dataset_description.md # Copied/created description
└── metadata.json          # Task type, classes, etc.

prepared_test_sets/my_dataset/
├── test.csv               # Test data (if provided)
└── test.no_label.csv
```

## Example Datasets

Download example datasets:

```bash
./download_example_datasets.sh
```

Or through the installed package:

```bash
python -m pip install ".[datasets]"
agentomics-download-datasets
```

## Data Format Tips

### Classification

- Target column should contain class labels (strings or integers)
- Binary: `positive`/`negative`, `1`/`0`, `yes`/`no`
- Multi-class: `class_a`, `class_b`, `class_c`
- Multi-label classification is not supported (use a single label per row)

### Regression

- Target column should contain numeric values
- The agent auto-detects regression when target is continuous

### Feature Columns

- Numeric features work best
- Categorical features are supported (encoded automatically)
- Missing values are handled, but clean data performs better

## Common Issues

### "Could not detect target column"

Solution: Add `--target-col your_column_name` to preparation command, or rename your target column to `class`, `target`, `label`, or `y`.

### "Task type unclear"

Solution: Add `--task-type classification` or `--task-type regression` to force the task type.

## Next Steps

- [Running the Agent](running-agent.md) - Use your prepared dataset
- [Understanding Outputs](outputs.md) - See what the agent produces
