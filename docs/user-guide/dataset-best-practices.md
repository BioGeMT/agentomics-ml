# Dataset Best Practices for Agentomics

This guide explains scientific and organizational practices for preparing datasets that ensure valid experimental results with Agentomics. For the complete technical specification of supported dataset formats and requirements, see [Preparing Datasets](datasets.md).

## Purpose and scope

This document is for ML engineers and computational biologists who need to:

- Prevent data leakage that invalidates experimental results
- Design privacy-safe identifiers for sensitive data
- Create valid train/validation/test splits
- Understand preprocessing constraints

**This guide covers decisions that Agentomics cannot validate automatically.** For technical format requirements, error messages, and supported layouts, see [Preparing Datasets](datasets.md).

## Preventing data leakage

Data leakage occurs when information from validation or test sets influences training, invalidating your evaluation metrics. Agentomics provides technical safeguards, but scientific validity depends on your dataset design.

### Exact duplicate IDs

**Required (partially enforced):**
- Train and validation IDs must not overlap
- Current validator enforces uniqueness within each `labels.csv`
- Cross-split intersection check is documented but not fully enforced across all preparation paths

Agentomics will reject duplicates within a split but may not catch all cross-split overlaps. Verify this manually for critical evaluations.

### Group leakage

**Scientific caution (not enforceable):**

Biomedical datasets often contain multiple related samples that must be kept together:

- Multiple samples from the same patient, family member, or donor
- Homologous sequences from the same protein family
- Structurally similar compounds or chemical analogs
- Technical replicates from the same specimen
- Images from the same tissue slide, scanner, or batch
- Time-series measurements from the same subject

**If related samples appear in both training and validation, your validation metrics will be optimistically biased.**

**Recommended split strategy:**

1. **Identify the independent unit:** Determine what constitutes a truly independent observation (patient, family, batch, study site, etc.)
2. **Split on that unit:** Ensure all samples from the same unit stay in the same split
3. **Document the unit:** Explain your grouping logic in `dataset_description.md`
4. **Consider stratification:** Balance class distribution across splits while respecting group boundaries

**Example:** For a patient dataset with multiple tissue samples per patient, split by patient ID, not by sample ID. All samples from patient_A must be in training or validation, never split between them.

### Learned preprocessing and transformation order

**Scientific caution:**

Any transformation that learns parameters from data creates leakage risk. Fit transformations on training data only, then apply the same transformation to validation and test data.

**Transformations that learn from data and risk leakage:**
- Imputation of missing values using mean, median, mode, or model-based methods
- Scaling and normalization using dataset mean/std, min/max, quantiles
- Vocabulary construction, tokenization dictionaries, or word embeddings from text
- Feature selection based on correlation, variance, or statistical tests
- Dimensionality reduction (PCA, UMAP, t-SNE)
- Target-informed filtering, weighting, or augmentation

**Safe preprocessing (may be done before preparation if documented):**
- Lossless format conversion (e.g., WAV to FLAC, TIFF to PNG)
- Fixed per-sample transformations (e.g., resize all images to 224×224)
- Domain-specific encoding with no learned parameters (e.g., one-hot encoding a fixed alphabet)

**Recommended approach:**
1. Provide raw or minimally processed data to Agentomics
2. Document any preprocessing applied in `dataset_description.md`, noting whether it was fit on training data only
3. Let the agent or your supplementary code fit transformations on training data during the run

**Counterexample (leakage):**
```python
# WRONG: Fits scaler on all data before splitting
from sklearn.preprocessing import StandardScaler
scaler = StandardScaler().fit(all_data)  # Sees validation/test statistics
train_scaled = scaler.transform(train_data)
val_scaled = scaler.transform(val_data)
```

**Correct approach:**
```python
# CORRECT: Fits scaler on training data only
from sklearn.preprocessing import StandardScaler
scaler = StandardScaler().fit(train_data)  # Only sees training statistics
train_scaled = scaler.transform(train_data)
val_scaled = scaler.transform(val_data)  # Applies training parameters
```

### Hidden test set guidelines

**Scientific caution:**

The hidden test set in `test_datasets/` is protected from the agent during training but is used for final evaluation. Multiple evaluations on the test set invalidate its purpose as a held-out assessment.

**Do not use test set results to:**
- Choose between different model architectures or configurations
- Tune hyperparameters or regularization strength
- Select preprocessing strategies or feature engineering approaches
- Decide when to stop iterating or which iteration to deploy
- Compare different datasets or problem formulations

**Use validation metrics for all model selection decisions.** The test set provides a single, final assessment of your chosen approach.

**Multiple peeks at test metrics constitute a form of indirect training on the test set** and will produce optimistically biased estimates of real-world performance.

## Privacy-safe identifiers

**Recommended:**

IDs in `labels.csv` and file names become part of the dataset contract and may appear in:
- Agent-generated code and error messages
- Prompts sent to LLM providers (unless using local-only models)
- Logs and reports

**For sensitive data:**
- IDs must be opaque and must not contain names, medical-record numbers, email addresses, dates of birth, geographic identifiers, or other personally identifiable information
- Use generated UUIDs, hashed identifiers, or sequential pseudonyms (e.g., `sample_001`, `specimen_a3f2`)
- Maintain a secure mapping file outside the dataset directory if you need to trace pseudonyms back to real identifiers
- Document any ID generation method in `dataset_description.md`

**Required:**
- IDs must be stable (do not change between runs)
- IDs must be unique within each split
- IDs must map consistently to input data across all splits

**Traceability vs. privacy:** Use opaque IDs in portable dataset files. Maintain traceability through secure, access-controlled mapping files in your data management system, not in the dataset itself.

## Privacy and data handling

**Important:** LLM providers may receive prompts or agent-generated context derived from data exploration, including:
- Column names, file names, or sample IDs
- Summary statistics (mean, min, max, distribution)
- Example values from the first few rows
- Error messages containing file paths or data snippets

**For sensitive, regulated, or proprietary data:**
- Use a local model provider (e.g., Ollama, local Codex) configured in `.env`
- Deploy on-premises or in a secure cloud environment that meets your organization's requirements
- Use privacy-safe identifiers and column names (e.g., `feature_1` instead of `patient_SSN`)
- Review your organization's data governance policies for AI/ML tool usage
- Consider whether data use agreements permit sending metadata to external providers

**Docker isolation alone does not prevent data from reaching the configured model provider.** The provider receives whatever the LLM client sends. Choose your provider and execution environment based on your data sensitivity and regulatory requirements.

## Dataset description best practices

**Highly recommended:** Create `dataset_description.md` in your dataset directory. For complex or domain-specific data, this may be the difference between a successful run and an agent that cannot load your data.

Include:

1. **Independent unit for splitting:** Explicitly state what constitutes an independent observation and whether samples are grouped (by patient, family, batch, etc.)

2. **Preprocessing applied:** Document what transformations were applied and whether they were fit on training data only. Example:
   ```markdown
   ## Preprocessing
   - Images resized to 224×224 (fixed transformation, no leakage risk)
   - Gene expression log2-transformed (per-sample, no leakage risk)
   - Z-score normalization: mean and std computed on training set only, applied to all splits
   ```

3. **Known limitations or biases:** Mention class imbalance, missing data patterns, batch effects, or domain shift between splits

4. **ID mapping for custom formats:** If using a non-standard format, explain explicitly how IDs map to input data

See [datasets.md](datasets.md#datasetdescriptionmd-optional) for a complete template.

## Validation checklist (scientific concerns)

Before running Agentomics, verify:

- [ ] Train and validation splits have no overlapping IDs
- [ ] I have identified the independent unit in my data (patient, family, batch, etc.)
- [ ] I have split on the independent unit to prevent group leakage
- [ ] All related samples (from the same patient/family/batch) are in the same split
- [ ] Any learned preprocessing (scaling, imputation, vocabulary) was fit on training data only
- [ ] IDs are privacy-safe and do not contain personally identifiable information
- [ ] `dataset_description.md` documents my split strategy and preprocessing
- [ ] I will not use test set results to select models or tune hyperparameters
- [ ] I have reviewed data handling and privacy requirements for my use case
- [ ] My execution environment and model provider are appropriate for my data sensitivity

For technical validation (file formats, folder structure, required fields), see [Preparing Datasets](datasets.md#common-issues).

## When to use this guide vs. the technical docs

| Question | Refer to |
|----------|----------|
| What file formats are supported? | [Preparing Datasets](datasets.md) |
| How do I structure my dataset directory? | [Preparing Datasets](datasets.md#quick-setup) |
| What columns are required in labels.csv? | [Preparing Datasets](datasets.md#labelscsv) |
| How do I fix a "labels.csv is invalid" error? | [Preparing Datasets](datasets.md#common-issues) |
| Should I split by patient or by sample? | This guide ([Group leakage](#group-leakage)) |
| Can I normalize data before creating splits? | This guide ([Learned preprocessing](#learned-preprocessing-and-transformation-order)) |
| What IDs are safe for medical data? | This guide ([Privacy-safe identifiers](#privacy-safe-identifiers)) |
| Will test metrics generalize to new data? | This guide ([Hidden test set guidelines](#hidden-test-set-guidelines)) |
| Can I use an external LLM with patient data? | This guide ([Privacy and data handling](#privacy-and-data-handling)) |

## Additional resources

- [Preparing Datasets](datasets.md) - Complete technical contract, file formats, and supported layouts
- [Running the Agent](running-agent.md) - How to start a run
- [Understanding Outputs](outputs.md) - What Agentomics produces
- Example datasets: Run `./scripts/download_example_dataset.sh --list`

## Getting help

If you encounter issues:

1. **For technical errors:** See [Common Issues](datasets.md#common-issues) and error-specific fixes
2. **For scientific validity questions:** Review this best practices guide
3. **For both:** Download and examine example datasets
4. [Open an issue](https://github.com/BioGeMT/agentomics-ml/issues) with your question
5. Email: martinekvlastimil95@gmail.com
