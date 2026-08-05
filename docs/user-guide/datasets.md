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
├── test/                   # Optional; held out from the agent
│   ├── input/
│   └── labels.csv
├── supplementary/          # Optional: dataset-level source materials
├── metadata.json           # Optional if --task-type is provided
└── dataset_description.md  # Optional domain context
```

The optional `test/` split is stored with the dataset, but is excluded from the
agent worker's mounts and agent-facing prepared data. After model development,
Agentomics mounts it read-only in a separate evaluation container and evaluates
only the best iteration against it.
Additional top-level directories whose names start with `test` are handled the
same way and evaluated separately.

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
Agentomics ships documentation for a set of foundation models, covering protein
(ESM-2), DNA (HyenaDNA, Nucleotide Transformer), RNA (RiNALMo), and molecule
(ChemBERTa). Use `agentomics-add-supplementary` to attach one to a dataset:

```bash
agentomics-add-supplementary --model ESM-2 --dataset-dir datasets/<your_dataset>
```

This writes the model's documentation and the foundation model README into
`datasets/<your_dataset>/supplementary/foundation_models/`, leaving the rest of
the dataset's `supplementary/` folder untouched. Run it once per model you want
to make available. Model names are case-insensitive; `agentomics-add-supplementary --help`
lists the available ones.

#### Note on using supplementary + forking

Forked runs use the dataset referenced by the source run config. If you change
dataset files after a run, start a fresh run when you need that changed data to
be part of the experiment.

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

The hidden test split is used only for final evaluation. Keep it under
`datasets/<dataset>/test/`. The run excludes it from the agent-facing prepared
dataset, then automatically evaluates the best iteration on it after training.

### metadata.json Optional

Agentomics does not infer task type from generic input files. If you do not
provide `metadata.json`, pass `--task-type classification` or
`--task-type regression` to `agentomics-run`, or select the task type when
prompted in an interactive run. For classification datasets, Agentomics derives
class IDs from `labels.csv` if `label_to_scalar` is absent.

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

## Flat CSV Files

If your data is in flat CSV files with features and labels in one table, you can
place them directly in the dataset directories. Agentomics converts them inside
the run workspace when the run starts; your source CSV files are not modified.

```text
datasets/my_dataset/
├── train.csv
├── validation.csv          # optional
├── test.csv                # optional, held out from the agent
├── metadata.json
└── dataset_description.md  # optional
```

Only these CSV names are auto-detected: `train.csv`, optional `validation.csv`,
and `test.csv` for hidden test data. Multiple hidden test sets use the
folder-based format described above.

For CSV datasets, `metadata.json` should identify the label column and task type:

```json
{
  "task_type": "classification",
  "label_column": "target",
  "id_column": "sample_id"
}
```

`id_column` is optional. If it is absent, Agentomics generates sample IDs. In
interactive runs, Agentomics can ask for the dataset `label_column`; in
non-interactive runs, add it to `metadata.json`.

The label column in the source CSV can have any name. During conversion it is
written as `label` in `labels.csv`, which is the required column name for the
folder-based format.

## Prepared Dataset Structure

Your source `datasets/<name>/` folder is mounted read-only and is never
modified. When a run starts, Agentomics prepares the public `train` and
`validation` splits into the run workspace, converting labels to
`id,numeric_label`:

```text
outputs/<agent_id>/run/shared/splits/split_0/
├── train/
│   ├── input/
│   └── labels.csv          # id,numeric_label
└── validation/
    ├── input/
    └── labels.csv          # id,numeric_label
```

The original label values are preserved through `label_to_scalar` in the run
config. The held-out `test/` split is not part of this prepared data — it is
excluded from the agent's mounts and prepared separately in a temporary
directory only for the post-run evaluation.

## Example Datasets

List available example datasets:

```bash
agentomics-download-dataset --list
```

Download one example dataset:

```bash
agentomics-download-dataset --dataset digits_images
```

Download all registered examples:

```bash
agentomics-download-dataset --all
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

## Validate Before a Run (Optional)

Check that a dataset satisfies the format contract without launching a run.
`agentomics-check-dataset` runs the same preparation and validation the agent
would, on the host and without Docker, then reports the result. Your dataset is
never modified.

```bash
agentomics-check-dataset --dataset-dir path/to/my_dataset
```

## Common Issues

### "Required split folder is missing or incomplete"

Check that `train/input/` exists and that `train/labels.csv` is present.

### "labels.csv is invalid"

Check that `labels.csv` has `id` and `label` columns, no duplicate or empty
IDs, and non-empty labels.

### "metadata.json is required"

Pass `--task-type classification` or `--task-type regression`, or add a
`metadata.json` file with `task_type`.

## Next Steps

- [Running the Agent](running-agent.md) - Use your dataset
- [Understanding Outputs](outputs.md) - See what the agent produces
