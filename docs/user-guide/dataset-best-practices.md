# Input Data Best Practices for Agentomics

This guide provides best practices for preparing input data for Agentomics to ensure optimal performance and avoid common pitfalls.

## Table of Contents
- [Dataset Structure](#dataset-structure)
- [Labels File Best Practices](#labels-file-best-practices)
- [Input Data Organization](#input-data-organization)
- [Metadata Configuration](#metadata-configuration)
- [Dataset Description](#dataset-description)
- [Supplementary Materials](#supplementary-materials)
- [Common Pitfalls and Solutions](#common-pitfalls-and-solutions)
- [Data Type Specific Guidelines](#data-type-specific-guidelines)

## Dataset Structure

### Directory Layout
Follow this structure for training data:

```text
datasets/my_dataset/
├── train/
│   ├── input/              # Required: data files
│   └── labels.csv          # Required: id,label
├── validation/             # Optional but recommended
│   ├── input/
│   └── labels.csv
├── supplementary/          # Optional: supporting materials
├── metadata.json           # Required (or use --task-type flag)
└── dataset_description.md  # Highly recommended
```

For test data (kept separate to prevent data leakage):

```text
test_datasets/my_dataset/
└── test/
    ├── input/
    └── labels.csv
```

### Best Practices

1. **Keep test data isolated**: Always place test data in `test_datasets/` to ensure the agent never accesses it during training
2. **Provide validation split**: While optional, providing a validation split gives you more control over data splits
3. **Use clear naming**: Dataset names should be descriptive and use underscores instead of spaces (e.g., `protein_stability_prediction`)
4. **Maintain consistent structure**: All splits (train, validation, test) must have matching top-level `input/` structure

## Labels File Best Practices

### Requirements

Your `labels.csv` must:
- Contain exactly two columns: `id` and `label`
- Have unique IDs within each split
- Have non-empty IDs and labels
- Ensure no ID overlap between train and validation splits

### Format Guidelines

**Classification tasks:**
```csv
id,label
sample_001,positive
sample_002,negative
sample_003,positive
```

**Regression tasks:**
```csv
id,label
patient_A,23.5
patient_B,45.2
patient_C,12.8
```

### Best Practices

1. **Use descriptive IDs**: Make IDs meaningful and traceable (e.g., `patient_123` instead of `1`)
2. **Consistent ID format**: Use the same ID format across all splits
3. **Clear label names**: For classification, use human-readable labels (e.g., `cancer`, `healthy` instead of `0`, `1`)
4. **No extra columns**: Remove any extra columns before creating the dataset
5. **Check for duplicates**: Verify no duplicate IDs exist within each split
6. **Match input data**: Ensure every ID in labels.csv corresponds to data in the input/ folder

## Input Data Organization

### Top-Level Structure Rules

The top-level entries in `train/input/` define the interface that all splits must match.

**For tabular data** (single file):
```text
train/input/data.csv
validation/input/data.csv
test/input/data.csv
```

**For per-sample files** (images, audio, sequences):
```text
# Correct - use a subdirectory
train/input/images/sample_001.png
train/input/images/sample_002.png
validation/input/images/sample_003.png

# Wrong - top-level files must match exactly
train/input/sample_001.png
validation/input/sample_003.png  # Error: different top-level files
```

### Best Practices

1. **Use subdirectories for file collections**: Always place per-sample files (images, audio, etc.) in a subdirectory within `input/`
2. **Make ID mapping obvious**: Use filename stems as IDs for file-based datasets
3. **Consistent file formats**: Use the same file format across all splits
4. **Organized structure**: For complex datasets, use clear subdirectory names (e.g., `input/protein_sequences/`, `input/features/`)
5. **Avoid special characters**: Use alphanumeric characters and underscores in filenames

## Metadata Configuration

### Required Fields

```json
{
  "task_type": "classification",  // or "regression"
  "positive_class": "cancer",     // for binary classification
  "negative_class": "healthy"
}
```

For CSV datasets, add:

```json
{
  "task_type": "classification",
  "label_column": "target",
  "id_column": "sample_id"  // optional
}
```

### Best Practices

1. **Always include metadata.json**: Even if you can use `--task-type` flag, metadata.json ensures reproducibility
2. **Document class meanings**: For classification, clearly specify what each class represents
3. **Version your metadata**: Include a comment with dataset version or date
4. **Include relevant parameters**: Add any dataset-specific information that might be useful

## Dataset Description

Create a `dataset_description.md` file that helps the agent understand your data:

```markdown
# [Dataset Name]

Brief overview of what this dataset contains and the prediction task.

## Features
- Description of input features
- Data source and collection method
- Any preprocessing applied

## Target
- What the label represents
- Class distribution (for classification)
- Value range (for regression)

## Data Format
- How IDs in labels.csv map to input data
- File formats used
- Any special encoding or format details

## Important Notes
- Domain-specific considerations
- Known limitations or biases
- Suggested modeling approaches
```

### Best Practices

1. **Be specific about ID mapping**: Clearly explain how IDs relate to input files
2. **Include domain context**: Provide relevant background information for the task
3. **Note preprocessing**: Document any normalization, scaling, or transformations applied
4. **Mention data quirks**: Highlight any unusual aspects of the data
5. **Keep it concise**: Aim for clarity over comprehensiveness

## Supplementary Materials

Use the `supplementary/` folder for:
- Research papers related to the dataset
- Foundation model weights or documentation
- Helper scripts or code snippets
- Domain-specific resources

### Best Practices

1. **Create supplementary/README.md**: Explain what each file is and how it should be used
2. **Prefer download scripts over large files**: For foundation models, provide code to download them to `~/.cache` rather than including weights directly
3. **Use example_supplementary/**: Copy relevant examples from `example_supplementary/` folder for protein, DNA, RNA, or molecule tasks
4. **Organize by purpose**: Create subdirectories like `supplementary/papers/`, `supplementary/models/`, etc.

Example supplementary/README.md:

```markdown
# Supplementary Materials

## Foundation Models
- `esm2_usage.py`: Example code for loading ESM-2 protein embeddings
- Download models on-demand using huggingface transformers

## Literature
- `reference_paper.pdf`: Original dataset publication

## Helper Scripts
- `preprocessing_utils.py`: Utility functions for data processing
```

## Common Pitfalls and Solutions

### Pitfall 1: Inconsistent Split Structures
**Problem**: Top-level `input/` files differ between train and validation
**Solution**: Ensure all splits have identical top-level `input/` structure; use subdirectories for files that differ

### Pitfall 2: ID Mismatches
**Problem**: IDs in labels.csv don't match input data identifiers
**Solution**: For file-based data, use filename stems as IDs; for tabular data, ensure ID column matches labels.csv

### Pitfall 3: Missing Metadata
**Problem**: Preparation fails because task type is unknown
**Solution**: Always include metadata.json or use `--task-type` flag

### Pitfall 4: Invalid Labels.csv
**Problem**: Extra columns, duplicate IDs, or empty values
**Solution**: Validate labels.csv has exactly `id,label` columns, all IDs are unique and non-empty

### Pitfall 5: Train/Validation ID Overlap
**Problem**: Same IDs appear in both train and validation splits
**Solution**: Ensure completely disjoint ID sets across splits

### Pitfall 6: Test Data Leakage
**Problem**: Test data placed in `datasets/` instead of `test_datasets/`
**Solution**: Always put test split in `test_datasets/<name>/test/`

### Pitfall 7: Large Files in Supplementary
**Problem**: Foundation model weights copied multiple times, consuming disk space
**Solution**: Provide download scripts instead of including large weights directly

## Data Type Specific Guidelines

### Tabular Data

**Structure:**
```text
train/
├── input/
│   └── data.csv  # Contains all features + ID column
└── labels.csv    # id,label
```

**Best Practices:**
- Include an ID column in your data.csv that matches labels.csv IDs
- Use clear column names for features
- Handle missing values before preparation
- Document feature meanings in dataset_description.md

**Example data.csv:**
```csv
patient_id,age,gene_expr_1,gene_expr_2
patient_001,45,2.3,1.8
patient_002,52,3.1,2.2
```

### Image Data

**Structure:**
```text
train/
├── input/
│   └── images/
│       ├── img_001.png
│       ├── img_002.png
│       └── ...
└── labels.csv
```

**Best Practices:**
- Use consistent image format (PNG, JPG, etc.)
- Use filename stems as IDs (e.g., `img_001` for `img_001.png`)
- Ensure consistent image dimensions within dataset or document if variable
- Consider preprocessing (resizing, normalization) before preparation if needed
- Include image dimension information in dataset_description.md

### Audio Data

**Structure:**
```text
train/
├── input/
│   └── audio/
│       ├── sample_001.wav
│       ├── sample_002.wav
│       └── ...
└── labels.csv
```

**Best Practices:**
- Use standard audio formats (WAV, MP3, FLAC)
- Use filename stems as IDs
- Document audio properties (sample rate, channels, duration)
- Consider providing preprocessing recommendations in dataset_description.md

### Sequence Data (Protein, DNA, RNA)

**Structure:**
```text
train/
├── input/
│   └── sequences.fasta  # or sequences.csv
└── labels.csv
```

**Best Practices:**
- For FASTA format, use sequence headers as IDs (without '>')
- For CSV format, include sequence_id and sequence columns
- Document sequence type and any special alphabet in dataset_description.md
- Consider including relevant foundation models in supplementary/
- Use example_supplementary/ resources for ESM-2, HyenaDNA, etc.

### Text Data

**Structure:**
```text
train/
├── input/
│   └── texts.csv  # or texts/ directory with .txt files
└── labels.csv
```

**Best Practices:**
- For single CSV: include id and text columns
- For file-based: use one .txt file per sample with filename stem as ID
- Document text preprocessing (tokenization, cleaning) in dataset_description.md
- Specify text length distribution and any truncation decisions

## Validation Checklist

Before running Agentomics, verify:

- [ ] All required folders exist (train/input/, train/labels.csv)
- [ ] labels.csv has exactly two columns: id,label
- [ ] All IDs are unique within each split
- [ ] No ID overlap between train and validation
- [ ] Top-level input/ structure matches across all splits
- [ ] metadata.json exists and specifies task_type
- [ ] Test data is in test_datasets/, not datasets/
- [ ] dataset_description.md provides clear context
- [ ] ID mapping between labels.csv and input data is obvious
- [ ] File formats are consistent across splits

## Quick Reference

| Aspect | Requirement | Recommendation |
|--------|-------------|----------------|
| Dataset location | `datasets/<name>/` | Use descriptive names |
| Test location | `test_datasets/<name>/test/` | Keep separate from training data |
| labels.csv columns | Exactly `id,label` | Use meaningful IDs |
| metadata.json | `task_type` required | Include all relevant fields |
| dataset_description.md | Optional | Highly recommended |
| validation split | Optional | Recommended for control |
| supplementary/ | Optional | Use for domain resources |
| input/ structure | Consistent across splits | Use subdirectories for files |

## Additional Resources

- [Datasets Documentation](docs/user-guide/datasets.md)
- [Running the Agent](docs/user-guide/running-agent.md)
- [Understanding Outputs](docs/user-guide/outputs.md)
- Example datasets: Run `./scripts/download_example_dataset.sh --list`

## Getting Help

If you encounter issues:
1. Check this best practices guide
2. Review example datasets
3. Consult the [documentation](https://biogemt.github.io/agentomics-ml/)
4. [Open an issue](https://github.com/BioGeMT/agentomics-ml/issues)
5. Email: martinekvlastimil95@gmail.com
