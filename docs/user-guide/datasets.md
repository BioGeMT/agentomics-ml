# Preparing Datasets

Agentomics uses folder-based dataset splits. Each split has an `input/`
folder with the data files and a `labels.csv` file with labels.

For scientific best practices on preventing data leakage, privacy-safe identifiers, and split strategy, see [Dataset Best Practices](dataset-best-practices.md).

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
├── supplementary/          # Optional: dataset-level source materials
├── metadata.json           # Optional if --task-type is provided
└── dataset_description.md  # Optional domain context
```

Hidden test data belongs in a separate root so the agent never sees it during
training:

```text
test_datasets/my_dataset/
└── test/
    ├── input/
    └── labels.csv
```

Only `train` and `validation` are supported in `datasets/`. Only `test` is
supported in `test_datasets/`.

**Important:** Source datasets must contain real files and directories. Symbolic links are rejected for security reasons.

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

Every labeled split must include `labels.csv` with exactly these columns:

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

Supporting/supplementary materials (PDFs, papers, helper scripts, foundation model docs or weights, ...) can be placed in a `supplementary/` folder inside the dataset directory (as a sibling to the train folder). The supplementary folder is made available to the agent as a read-only folder and the agent can copy any files into its working directory.

You can describe what the supplementary folder contains and how it relates to
the task in `supplementary/README.md` to help the agent use it effectively.

#### Using existing / foundation models

To make models available to the agent, copy the models or their docs
into the dataset's `supplementary/` folder before starting the run.
If you put model weights or other large files in the supplementary/ folder, the agent might copy them multiple times during the run and their full copies might be present multiple times in the exported runs. To reduce disk use, it is recommended to provide code snippets that will download the models on-demand into `~/.cache` (e.g. using huggingface).
Ready-to-use examples live in `example_supplementary/`, covering protein (ESM-2),
DNA (HyenaDNA, Nucleotide Transformer), RNA (RiNALMo), and molecule (ChemBERTa)
models. To use them in your runs, copy them into your dataset's supplementary folder:

```bash
mkdir -p datasets/<your_dataset>/supplementary
cp -r example_supplementary/. datasets/<your_dataset>/supplementary/
```

#### Note on using supplementary + forking

Forked runs use the dataset referenced by the source run config. If you change
dataset files after a run, start a fresh run when you need that changed data to
be part of the experiment.

### input/ structure

The top-level entries in `train/input/` are recorded at dataset preparation
time and define the split interface. All splits must have matching top-level
`input/` files and folders: `validation/input/` and `test_datasets/<name>/test/input/` are validated
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

The hidden test split is used only for final evaluation. Keep it under
`test_datasets/<dataset>/test/`, not under `datasets/<dataset>/test/`.
The agent does not get access to `test_datasets/` during training.

### metadata.json Optional

Preparation does not infer task type from generic input files. Task type can be specified through `metadata.json`, the `--task-type` flag, or an interactive prompt.

| Scenario | Interactive mode | Non-interactive mode |
|----------|------------------|----------------------|
| Folder dataset, no metadata.json | Prompted for task type | Must use `--task-type` flag |
| Folder dataset with metadata.json | No prompt needed | No flag needed |
| Flat CSV, public data | Prompted for label_column | metadata.json must specify `label_column` |
| Flat CSV, hidden test data | N/A | metadata.json must specify `label_column` |

For `--prepare-all`, every dataset must provide `metadata.json`.

**Required field:**

```json
{
  "task_type": "classification"
}
```

Valid values: `"classification"` or `"regression"`.

**Optional binary classification fields** (both required if used):

```json
{
  "task_type": "classification",
  "positive_class": "cancer",
  "negative_class": "no_cancer"
}
```

For classification datasets, Agentomics derives class IDs from `labels.csv` if `label_to_scalar` is absent.

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

## Flat CSV Files

If your data is in flat CSV files with features and labels in one table, you can
place them directly in the dataset directories. Agentomics converts them inside
the run workspace when the run starts; your source CSV files are not modified.

```text
datasets/my_dataset/
├── train.csv
├── validation.csv          # optional
├── metadata.json
└── dataset_description.md  # optional

test_datasets/my_dataset/
├── test.csv
└── metadata.json           # label_column, optional id_column
```

Only these CSV names are auto-detected: `train.csv`, optional `validation.csv`,
and `test.csv` for hidden test data.

For CSV datasets, `metadata.json` should identify the label column and task type:

```json
{
  "task_type": "classification",
  "label_column": "target",
  "id_column": "sample_id"
}
```

`id_column` is optional. If it is absent, Agentomics generates sample IDs. In
interactive runs, Agentomics can ask for the public dataset `label_column`; in
non-interactive runs, add it to `metadata.json`. Hidden test CSV datasets must
provide `label_column` in `test_datasets/<name>/metadata.json`.

The label column in the source CSV can have any name. During conversion it is
written as `label` in `labels.csv`, which is the required column name for the
folder-based format.

## Prepared Dataset Structure

After preparation, the same dataset directory contains numeric labels:

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

test_datasets/my_dataset/
└── test/
    ├── input/
    └── labels.csv          # id,numeric_label
```

Prepared `labels.csv` files contain `id,numeric_label`; original label values are
kept in `metadata.json` through `label_to_scalar`.

## Example Datasets

List available example datasets:

```bash
./scripts/download_example_dataset.sh --list
```

Download one example dataset:

```bash
./scripts/download_example_dataset.sh --dataset digits_images
```

Download all registered examples:

```bash
./scripts/download_example_dataset.sh --all
```

Useful small examples are available for different input types:

- `breast_cancer` - tabular CSV input generated from scikit-learn
- `digits_images` - PNG image files generated from scikit-learn digits
- `spoken_digits` - WAV audio files from the Free Spoken Digit Dataset

The files inside `input/` can use different formats. Agentomics asks the agent
to inspect `input/` and generate loading code that maps each `labels.csv` `id`
to its sample data.

### Tabular Example

Use one table with an ID column:

```text
train/
├── input/
│   └── data.csv
└── labels.csv
```

`labels.csv.id` should match the `id` column in `input/data.csv`.

### Image Example

Use one file per sample inside an image subdirectory:

```text
train/
├── input/
│   └── images/
│       ├── img_001.png
│       └── img_002.png
└── labels.csv
```

`labels.csv.id` should match the image filename stem, for example `img_001`
maps to `input/images/img_001.png`.

### Audio Example

Use one file per sample inside an audio subdirectory:

```text
train/
├── input/
│   └── audio/
│       ├── sample_001.wav
│       └── sample_002.wav
└── labels.csv
```

`labels.csv.id` should match the audio filename stem, for example `sample_001`
maps to `input/audio/sample_001.wav`.

## Common Issues

### "Required split folder is missing or incomplete"

**Cause:** `train/input/` or `train/labels.csv` not found.

**Fix:** Ensure `datasets/<name>/train/` contains both `input/` folder and `labels.csv`.

### "labels.csv is invalid"

**Cause:** Wrong column names, duplicate IDs, or empty values.

**Fix:**
- Verify columns are exactly `id` and `label` (not `ID`, `sample_id`, `target`, etc.)
- Check for duplicate or empty IDs
- Ensure all labels are non-empty

### "metadata.json is required"

**Cause:** Task type cannot be determined.

**Fix:** Add `metadata.json` with `task_type` field, or use `--task-type` flag, or run interactively.

### "Top-level input structure mismatch"

**Cause:** Different top-level files or folders in `train/input/` vs `validation/input/`.

**Fix:** Ensure all splits have identical top-level `input/` structure. For per-sample files, use a subdirectory:

```text
# Correct
train/input/images/sample_001.png
validation/input/images/sample_099.png

# Wrong
train/input/sample_001.png
validation/input/sample_099.png
```

### Symlink rejected

**Cause:** Source dataset contains symbolic links.

**Fix:** Copy real files instead of using symlinks. Symlink support is not available for security reasons.

## Next Steps

- [Dataset Best Practices](dataset-best-practices.md) - Scientific best practices for preventing data leakage and ensuring valid evaluation
- [Running the Agent](running-agent.md) - Use your dataset
- [Understanding Outputs](outputs.md) - See what the agent produces
