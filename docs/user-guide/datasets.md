# Preparing Datasets

Agentomics-ML uses folder-based dataset splits. Each split has an `input/`
folder with the data files and a `labels.csv` file with labels.

## Quick Setup

Create a folder in `datasets/` with your data:

```text
datasets/my_dataset/
├── train/
│   ├── input/              # Required: data files
│   └── labels.csv          # Required: id,label
├── validation/             # Optional
│   ├── input/
│   └── labels.csv
├── test/                   # Optional hidden test set
│   ├── input/
│   └── labels.csv
├── supplementary/          # Optional: dataset-level source materials
├── metadata.json           # Optional if --task-type is provided
└── dataset_description.md  # Optional domain context
```

Only `train`, `validation`, and `test` are supported split names.

## Split Requirements

### input/

The `input/` folder can contain any files in any format — the agent's generated
training and inference scripts interpret the contents. The system does not
enforce a specific mapping between IDs and input data; the agent figures out
how to load data for each ID based on the format it finds.

Common patterns:

- **Tabular data**: `input/` contains a CSV or parquet file. IDs are row
  identifiers within that file (e.g. a patient ID column).
  ```text
  train/input/data.csv       # all samples in one file
  ```
- **Per-sample files** (images, audio, etc.): `input/` contains a subdirectory
  with one file per sample. IDs are typically filename stems.
  ```text
  train/input/images/img_001.png, img_002.png, ...
  ```

Make the relationship between IDs and data obvious so the agent can infer it.
For example, use filename stems as IDs for file datasets, or include an ID
column in tabular files that matches `labels.csv`.

### labels.csv

Every raw labeled split must include `labels.csv` with exactly these columns:

```csv
id,label
sample-1,cancer
sample-2,no_cancer
```

Requirements:

- `id` is required, non-empty, and unique within the split
- `label` is required and non-empty
- Extra columns are not supported
- Train and validation IDs must not overlap
- Classification labels may be friendly strings; preparation maps them to integer class IDs
- Regression labels must be numeric values

### supplementary/ Optional

Supporting/supplementary materials (PDFs, papers, helper scripts) can be placed
in a `supplementary/` folder inside the dataset directory (as a sibling to the train folder). These are copied
during preparation and made available to the agent. The agent can read
these materials for context but must copy needed files into its working directory
before using them. Generated training and inference scripts must not reference
`supplementary/` directly. If you update supplementary materials, re-prepare the
dataset to ensure the agent has the latest version.

You can describe what the supplementary folder contains and how it relates to
the task in `dataset_description.md` to help the agent use it effectively.

When forking a run, the supplementary folder is inherited from the source run's
workspace — not re-copied from the raw or prepared dataset. If you update
supplementary materials after a run has started, start a new run (re-prepare the
dataset) rather than forking to pick up the changes.

### input/ structure

The top-level entries in `train/input/` are recorded at dataset preparation
time and define the split interface. All splits must have matching top-level
`input/` files and folders: `validation/input/` and `test/input/` are validated
against `train/input/` during preparation, and the agent cannot add, remove, or
rename top-level `input/` entries during the split step. Files inside matching
top-level folders may differ between splits, which supports datasets where each
split contains different sample files.

For per-sample file datasets (images, audio, etc.), place files inside a
subdirectory rather than directly under `input/`:

```text
# Correct — subdirectory contents may differ between splits
train/input/images/cat_01.png, cat_02.png
validation/input/images/cat_03.png

# Wrong — top-level files must match exactly across splits
train/input/cat_01.png, cat_02.png
validation/input/cat_03.png          # fails: different top-level files
```

### validation/ Optional

If `validation/` is not provided, the agent creates train and validation split
folders from `train/` during the run.

### test/ Optional

The hidden test split is used only for final evaluation. The agent does not get
access to `prepared_test_sets/` during training.

### metadata.json Optional

Preparation does not infer task type from generic input files. If you do not
provide `metadata.json`, pass `--task-type classification` or
`--task-type regression` during single-dataset preparation, or run single-dataset
preparation interactively. For `--prepare-all`, every dataset must provide
`metadata.json`. For classification datasets, Agentomics derives class IDs from
`labels.csv` if `label_to_scalar` is absent.

Example:

```json
{
  "task_type": "classification",
  "positive_class": "cancer",
  "negative_class": "no_cancer"
}
```

### dataset_description.md Optional

Any domain information can help the agent understand your data. You may include how IDs in
`labels.csv` relate to data in `input/` (e.g., row identifiers in a CSV, or
filename stems). The agent discovers this during exploration, but providing it
can be helpful.

```markdown
# Gene Expression Dataset

This dataset contains RNA-seq expression levels from tumor samples.

## Features
- Input files contain log2 TPM expression values
- Samples are from breast cancer patients

## Target
- `label`: tumor subtype (Basal, Her2, LumA, LumB, Normal)

## Data format
`input/` contains a single CSV with a `patient_id` column and expression value columns.
Each ID in `labels.csv` matches the `patient_id` column in the input CSV.

## Notes
- Data is already normalized
- Consider models that handle high-dimensional data
```

## Converting CSV Files

If your data is in flat CSV files (features and labels in one table), use the
CSV converter to create the folder-based layout:

```bash
PYTHONPATH=src python src/datasets/csv_converter.py \
    --train-csv data/train.csv \
    --test-csv data/test.csv \
    --label-column target \
    --task-type classification \
    --output-dir datasets/my_dataset
```

| Option | Description |
|--------|-------------|
| `--train-csv` | Path to training CSV (required) |
| `--validation-csv` | Path to validation CSV (optional) |
| `--test-csv` | Path to test CSV (optional) |
| `--label-column` | Name of the column containing labels (required) |
| `--id-column` | Name of the ID column (auto-generated if not provided) |
| `--task-type` | `classification` or `regression` (optional, writes metadata.json) |
| `--output-dir` | Output dataset directory (required) |

The converter can also be called from Python:

```python
from datasets.csv_converter import convert_csv_dataset

convert_csv_dataset(
    output_dir=Path("datasets/my_dataset"),
    label_column="target",
    splits={"train": train_df, "test": test_df},
    task_type="classification",
)
```

The label column in the source CSV can have any name — specify it with
`--label-column`. The converter writes it as `label` in `labels.csv`, which is
the required column name for the folder-based format.

After conversion, run dataset preparation as usual.

## Manual Dataset Preparation

For more control, run preparation separately:

```bash
conda env create -f envs/environment_prepare.yaml
conda activate agentomics-prepare-env

python src/prepare_datasets.py --dataset-dir datasets/my_dataset --task-type classification
```

To prepare all datasets, include `metadata.json` in each dataset folder.
`--task-type` is intentionally limited to single-dataset preparation because
different datasets may have different task types.

```bash
python src/prepare_datasets.py --prepare-all
```

Key options:

| Option | Description |
|--------|-------------|
| `--dataset-dir` | Specific dataset to prepare |
| `--task-type` | Single-dataset `classification` or `regression` value when `metadata.json` is absent |
| `--positive-class` | Raw label value to encode as numeric class `1` for binary classification |
| `--negative-class` | Raw label value to encode as numeric class `0` for binary classification |

Running preparation for a single dataset skips the dataset if it is already
prepared under `prepared_datasets/`.


## Prepared Dataset Structure

After preparation, datasets are stored in:

```text
prepared_datasets/my_dataset/
├── train/
│   ├── input/
│   └── labels.csv
├── validation/
│   ├── input/
│   └── labels.csv
├── supplementary/          # Dataset-level source materials, if provided
├── dataset_description.md
└── metadata.json

prepared_test_sets/my_dataset/
└── test/
    ├── input/
    └── labels.csv
```

Prepared `labels.csv` files contain `id,numeric_label`; raw label values are
kept in `metadata.json` through `label_to_scalar`.

## Example Datasets

Download example datasets:

```bash
./scripts/download_example_dataset.sh --all
```

## Common Issues

### "Required split folder is missing or incomplete"

Check that `train/input/` exists and that `train/labels.csv` is present.

### "labels.csv is invalid"

Check that raw `labels.csv` has `id` and `label` columns, no duplicate or empty
IDs, and non-empty labels.

### "metadata.json is required"

Pass `--task-type classification` or `--task-type regression`, or add a
`metadata.json` file with `task_type`.

## Next Steps

- [Running the Agent](running-agent.md) - Use your prepared dataset
- [Understanding Outputs](outputs.md) - See what the agent produces
