# Dataset Best Practices for Agentomics

This guide explains scientific and organizational practices for preparing datasets that work reliably with Agentomics. For the complete technical contract and all supported input formats, see [Preparing Datasets](datasets.md).

## Purpose and scope

This document is for ML engineers and computational biologists preparing datasets for Agentomics. It covers:

- Scientific cautions to prevent data leakage and invalid evaluation
- Privacy-safe identifier design
- When to choose flat CSV versus folder-based layouts
- Validation strategy and common errors

This guide distinguishes between:
- **Required:** enforced by Agentomics and blocks a run
- **Recommended:** improves reliability or reproducibility but does not block a run
- **Scientific caution:** a decision that Agentomics cannot validate universally

## Choose your input layout

Agentomics supports two input layouts. Choose based on your data structure:

### Flat CSV layout

**Use when:** Your data is one-row-per-sample tabular data with features and labels in a single table.

**Minimal example:**
```text
datasets/my_dataset/
├── train.csv
├── validation.csv      # optional
└── metadata.json
```

Each CSV contains all features plus a label column. See [Flat CSV Files](datasets.md#flat-csv-files) for the full contract.

**This is the shortest path for beginners with tabular data.**

### Folder-based layout

**Use when:** You have per-sample files (images, audio, sequences), complex multi-file inputs, or data that cannot fit in a flat table.

**Minimal example:**
```text
datasets/my_dataset/
├── train/
│   ├── input/
│   └── labels.csv
└── metadata.json
```

See [Split Requirements](datasets.md#split-requirements) for the full contract.

## Minimal working examples

### Example 1: Flat CSV (quickest for tabular data)

**train.csv:**
```csv
sample_id,feature_1,feature_2,feature_3,disease_status
s001,2.3,1.8,0.5,healthy
s002,3.1,2.2,0.7,disease
s003,1.9,1.5,0.4,healthy
```

**metadata.json:**
```json
{
  "task_type": "classification",
  "label_column": "disease_status",
  "id_column": "sample_id"
}
```

Agentomics converts this internally to the folder contract during preparation.

### Example 2: Folder-based with per-sample image files

**Directory structure:**
```text
datasets/cell_images/
├── train/
│   ├── input/
│   │   └── images/
│   │       ├── cell_001.png
│   │       ├── cell_002.png
│   │       └── cell_003.png
│   └── labels.csv
└── metadata.json
```

**train/labels.csv:**
```csv
id,label
cell_001,normal
cell_002,abnormal
cell_003,normal
```

**metadata.json:**
```json
{
  "task_type": "classification"
}
```

The ID `cell_001` maps to the file `train/input/images/cell_001.png` by filename stem convention.

## Required contract

See [Preparing Datasets](datasets.md) for the authoritative technical contract. Key enforced requirements:

- **Required:** `train/` split with `input/` folder and `labels.csv`
- **Required:** `labels.csv` has exactly two columns named `id` and `label`
- **Required:** All IDs are unique within each split
- **Required:** All IDs and labels are non-empty
- **Required:** Task type is specified via `metadata.json`, `--task-type` flag, or interactive prompt
- **Required:** Top-level `input/` structure matches across all splits
- **Required:** Hidden test data lives in `test_datasets/<name>/test/`, not `datasets/`
- **Required:** Source datasets contain real files and directories; symlinks are rejected

## Task type and metadata decision table

| Scenario | Interactive mode | Non-interactive mode |
|----------|------------------|----------------------|
| Folder dataset, no metadata.json | Prompted for task type | Must use `--task-type` flag |
| Folder dataset with metadata.json | No prompt needed | No flag needed |
| Flat CSV, public data | Prompted for label_column | metadata.json must specify `label_column` |
| Flat CSV, hidden test data | N/A | metadata.json must specify `label_column` |

### Metadata fields

**task_type** (required in one form):
```json
{
  "task_type": "classification"
}
```
Valid values: `"classification"` or `"regression"`

**Flat CSV fields** (when using train.csv layout):
```json
{
  "task_type": "classification",
  "label_column": "target",
  "id_column": "sample_id"
}
```
`id_column` is optional. If absent, Agentomics generates IDs.

**Binary classification classes** (optional, both required if used):
```json
{
  "task_type": "classification",
  "positive_class": "disease",
  "negative_class": "healthy"
}
```
Only meaningful for binary classification. Must be supplied together.

**Recommended: Include dataset version for reproducibility** (no special runtime semantics):
```json
{
  "task_type": "classification",
  "dataset_version": "2026-07-13"
}
```
Unknown metadata keys are preserved by dataset preparation.

## ID mapping and privacy-safe identifiers

### ID requirements

**Required:**
- IDs must be stable and unique within each split
- IDs must map consistently to input data across all splits

**Recommended:**
- IDs should be opaque and must not contain names, medical-record numbers, email addresses, or other direct identifiers
- Use generated UUIDs or hashed identifiers for sensitive data

**Scientific caution:**
- If multiple samples belong to the same patient, specimen, family, batch, or study site, that grouping must be documented and respected when creating splits to prevent group leakage

### ID-to-input mapping conventions

The system does not enforce a universal ID-to-data mapping for all modalities. Common conventions that Agentomics can validate or that make the mapping obvious:

**Tabular data:**
- Flat CSV: `id_column` in metadata.json specifies which column contains IDs
- Folder CSV: An ID column in `input/data.csv` matches IDs in `labels.csv`

**Per-sample files (images, audio):**
- Filename stems map to IDs: `cell_001` matches `cell_001.png`
- Files must be in a subdirectory within `input/`, e.g., `input/images/`

**FASTA sequences:**
- Sequence headers (without `>`) map to IDs

**Custom formats:**
- Explicit documentation in `dataset_description.md` is required
- Automatic validation is best-effort for non-standard formats

## Split strategy and leakage cautions

### Preventing data leakage

**Exact duplicate IDs** (required, partially enforced):
- Train and validation IDs must not overlap
- Current validator enforces uniqueness within each `labels.csv`
- Cross-split intersection check is documented but not fully enforced across all preparation paths

**Group leakage** (scientific caution, not enforceable):

Many biomedical datasets contain multiple related samples:
- Multiple rows from the same patient
- Homologous sequences from the same protein family
- Repeated compounds with similar structure
- Technical replicates from the same specimen
- Images from the same batch or scanner

**Recommended split strategy:**
1. Identify the independent unit in your data (patient, family, batch, etc.)
2. Split on that unit, not on individual samples
3. Document the unit and your split strategy in `dataset_description.md`
4. Consider stratification for balanced class distribution

### Learned preprocessing and transformation order

**Scientific caution:**

Preserve raw or minimally processed inputs when practical. Any learned transformation must be fitted using training data only and then applied unchanged to validation and test data.

**Transformations that learn from data and risk leakage:**
- Imputation of missing values using mean/median/mode
- Scaling and normalization using dataset statistics
- Vocabulary construction from text
- Feature selection or dimensionality reduction
- Target-informed filtering or reweighting

**Safe preprocessing (may be done before preparation if documented):**
- Lossless format conversion (e.g., WAV to FLAC)
- Per-sample transformations that do not learn from the dataset (e.g., resizing all images to a fixed dimension)
- Domain-specific encoding that does not depend on dataset statistics

**Recommended approach:**
- Provide raw or minimally processed data to Agentomics
- Document any preprocessing in `dataset_description.md`
- Let the agent or your supplementary code fit transformations on training data only

### Hidden test set guidelines

**Scientific caution:**

The hidden test set in `test_datasets/` is protected from the agent during training but is used for final evaluation. Do not use test set results to:
- Choose between models or configurations
- Tune hyperparameters
- Select preprocessing strategies
- Decide when to stop the agent

Multiple peeks at test metrics invalidate the evaluation. Use validation metrics for model selection.

## Dataset description template

**Recommended:** Create `dataset_description.md` in your dataset directory. This helps the agent understand your data and is especially important for complex or domain-specific formats.

```markdown
# [Dataset Name]

Brief overview of the prediction task and data source.

## Task
- Task type: classification or regression
- Target: what the label represents
- Application context

## Data format
- How IDs in labels.csv map to input data
- File formats and organization
- Any special encoding or conventions

## Samples and splits
- Total sample count
- Class distribution (for classification) or value range (for regression)
- Independent unit for splitting (sample, patient, family, batch)
- Any group structure or dependencies

## Preprocessing applied
- Transformations already applied to the data
- Whether transformations were fit on training data only
- Any normalization, scaling, or filtering

## Domain-specific notes
- Biological context or technical details
- Known limitations or biases
- Suggested modeling approaches or relevant literature
```

## Supplementary materials

**Optional:** Use the `supplementary/` folder for domain resources that may help the agent:
- Research papers or documentation
- Foundation model usage examples
- Helper scripts or code snippets

**Recommended practices:**
- Create `supplementary/README.md` explaining what each file contains and how it should be used
- For foundation models, provide download scripts or code that loads models into `~/.cache` rather than including large weights directly
- Use ready-made examples from `example_supplementary/` for protein (ESM-2), DNA (HyenaDNA, Nucleotide Transformer), RNA (RiNALMo), and molecule (ChemBERTa) models

See [Using existing / foundation models](datasets.md#using-existing--foundation-models) for details.

## Common errors and fixes

### Error: "Required split folder is missing or incomplete"
**Cause:** `train/input/` or `train/labels.csv` not found
**Fix:** Ensure `datasets/<name>/train/` contains both `input/` folder and `labels.csv`

### Error: "labels.csv is invalid"
**Cause:** Wrong column names, duplicate IDs, or empty values
**Fix:**
- Verify columns are exactly `id` and `label` (not `ID`, `sample_id`, `target`, etc.)
- Check for duplicate or empty IDs
- Ensure all labels are non-empty

### Error: "metadata.json is required"
**Cause:** Task type cannot be determined
**Fix:** Add `metadata.json` with `task_type` field, or use `--task-type` flag, or run interactively

### Error: "Top-level input structure mismatch"
**Cause:** Different top-level files or folders in `train/input/` vs `validation/input/`
**Fix:** Ensure all splits have identical top-level `input/` structure. For per-sample files, use a subdirectory:
```text
# Correct
train/input/images/sample_001.png
validation/input/images/sample_099.png

# Wrong
train/input/sample_001.png
validation/input/sample_099.png
```

### Error: Symlink rejected
**Cause:** Source dataset contains symbolic links
**Fix:** Copy real files instead of using symlinks. Symlink support requires a separate security design and is currently not supported.

### Unexpected: Agent cannot load data
**Cause:** ID-to-input mapping is unclear
**Fix:**
- For file-based data, use filename stems as IDs
- For tabular data, include an ID column that matches `labels.csv`
- Document the mapping explicitly in `dataset_description.md`

### Scientific error: Validation accuracy too high
**Cause:** Possible data leakage from group overlap or learned preprocessing
**Fix:**
- Verify train and validation IDs are completely disjoint
- Check for related samples (same patient, family, batch) across splits
- Review preprocessing: ensure transformations were fit on training data only

## Privacy and data handling

**Important:** External LLM providers may receive prompts or agent-generated context derived from data exploration. This may include:
- Column names, file names, or IDs from your dataset
- Summary statistics or example values
- Error messages containing file paths

**For sensitive data:**
- Use a local model provider (e.g., Ollama) configured in `.env`
- Choose an appropriate execution environment (on-premises compute, encrypted storage)
- Use opaque IDs that do not contain personally identifiable information
- Review your organization's data governance policies

**Note:** Docker isolation alone does not prevent data from reaching the configured model provider. Choose your provider and execution environment accordingly.

## Validation checklist

Before running Agentomics:

- [ ] I have chosen flat CSV or folder-based layout appropriate for my data
- [ ] All required folders and files exist (`train/input/`, `labels.csv`)
- [ ] `labels.csv` has exactly two columns: `id,label`
- [ ] All IDs are unique within each split
- [ ] Train and validation splits have no overlapping IDs
- [ ] Top-level `input/` structure is identical across all splits
- [ ] `metadata.json` exists and specifies `task_type` (or I will use `--task-type` flag)
- [ ] Hidden test data is in `test_datasets/`, not `datasets/`
- [ ] IDs are privacy-safe (no names, medical records, or direct identifiers)
- [ ] ID-to-input mapping is obvious or documented in `dataset_description.md`
- [ ] I have identified the independent unit and split on that unit to prevent group leakage
- [ ] Any learned preprocessing was fit on training data only
- [ ] `dataset_description.md` provides context for the agent
- [ ] No symlinks in source dataset
- [ ] I have reviewed data handling and privacy requirements for my use case

## Prepared dataset behavior

After running dataset preparation, `labels.csv` files contain `id,numeric_label` instead of `id,label`. Original label values are preserved in `metadata.json` through the `label_to_scalar` mapping. Do not copy this internal format into source datasets.

## Quick reference

| Aspect | Requirement | Recommendation |
|--------|-------------|----------------|
| Input layout | Flat CSV or folder-based | Flat CSV for simple tabular data |
| Dataset location | `datasets/<name>/` | Use descriptive, underscore-separated names |
| Test location | `test_datasets/<name>/test/` | Required; keeps test data isolated |
| labels.csv columns | Exactly `id,label` | Use privacy-safe IDs |
| metadata.json | `task_type` required in some form | Always include for reproducibility |
| dataset_description.md | Optional | Highly recommended |
| validation split | Optional | Recommended for split control |
| supplementary/ | Optional | Use for foundation models and domain resources |
| ID overlap | Train and validation must be disjoint | Document and respect group structure |
| Preprocessing | Must not leak validation/test information | Fit on training data only |
| Symlinks | Not supported | Use real files |

## Additional resources

- [Preparing Datasets](datasets.md) - Complete technical contract and all supported formats
- [Running the Agent](running-agent.md) - How to start a run with your dataset
- [Understanding Outputs](outputs.md) - What Agentomics produces
- Example datasets: Run `./scripts/download_example_dataset.sh --list`
- [Agentomics documentation](https://biogemt.github.io/agentomics-ml/)

## Getting help

If you encounter issues:
1. Check this best practices guide and [Preparing Datasets](datasets.md)
2. Download and examine example datasets
3. [Open an issue](https://github.com/BioGeMT/agentomics-ml/issues) with dataset structure and error output
4. Email: martinekvlastimil95@gmail.com
