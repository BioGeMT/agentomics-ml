# Batch Experiments

Run multiple experiments across models and datasets with `orchestrator.sh`.

## Overview

The orchestrator script automates running Agentomics-ML across:

- Multiple LLM models
- Multiple datasets
- Consistent configuration

Useful for:
- Benchmarking different models
- Testing across datasets
- Systematic experiments

## Basic Usage

```bash
./orchestrator.sh
```

The script reads configuration and runs experiments in sequence.

## Configuration

Edit `orchestrator.sh` to configure:

### Models

```bash
MODELS=(
    "openai/gpt-4o"
    "anthropic/claude-3.5-sonnet"
    "openai/gpt-4-turbo"
)
```

### Datasets

```bash
DATASETS=(
    "breast_cancer"
    "gene_expression"
    "protein_classification"
)
```

### Common Parameters

```bash
ITERATIONS=10
VAL_METRIC="AUROC"
TIMEOUT=7200
```

## Experiment Matrix

The orchestrator runs all combinations:

```
Model 1 × Dataset 1 → Run 1
Model 1 × Dataset 2 → Run 2
Model 1 × Dataset 3 → Run 3
Model 2 × Dataset 1 → Run 4
...
```

## Output Organization

Results are organized by model and dataset:

```
outputs/
├── gpt-4o_breast_cancer_<timestamp>/
├── gpt-4o_gene_expression_<timestamp>/
├── claude-3.5-sonnet_breast_cancer_<timestamp>/
└── ...
```

## Spend Limits

Set maximum API spend:

```bash
MAX_SPEND=100.00  # USD
```

Experiments stop if spend limit is reached (OpenRouter only).

## Tags

Add tags for experiment tracking:

```bash
TAGS="benchmark-v1,paper-experiments"
```

Tags appear in W&B if logging is enabled.

## Timeout per Run

Set maximum time per experiment:

```bash
TIMEOUT=7200  # 2 hours per run
```

## Example Configuration

```bash
#!/bin/bash

# Models to test
MODELS=(
    "openai/gpt-4o"
    "anthropic/claude-3.5-sonnet"
)

# Datasets to use
DATASETS=(
    "dataset_a"
    "dataset_b"
)

# Run parameters
ITERATIONS=10
VAL_METRIC="AUROC"
TIMEOUT=3600

# Limits
MAX_SPEND=50.00

# Run all combinations
for model in "${MODELS[@]}"; do
    for dataset in "${DATASETS[@]}"; do
        ./run.sh \
            --model "$model" \
            --dataset "$dataset" \
            --iterations "$ITERATIONS" \
            --val-metric "$VAL_METRIC" \
            --timeout "$TIMEOUT"
    done
done
```

## Parallel Execution

For parallel runs on multiple machines:

1. Split the model/dataset combinations
2. Run different subsets on each machine
3. Combine results afterward

## Monitoring

### Progress

Each run outputs progress to console:

```
[1/6] Running: gpt-4o on breast_cancer
[2/6] Running: gpt-4o on gene_expression
...
```

### Logging

If W&B is configured, all runs are tracked:

- Compare runs in W&B dashboard
- Filter by tags
- Analyze trends

## Error Handling

### Single Run Failure

If one run fails, the orchestrator continues with the next combination.

### Retry Failed Runs

Re-run only failed experiments:

```bash
./orchestrator.sh --retry-failed
```

### Skip Completed

Skip already-completed runs:

```bash
./orchestrator.sh --skip-completed
```

## BioMLBench Integration

For benchmark datasets:

```bash
./orchestrator.sh --biomlbench
```

Uses the BioMLBench dataset collection for standardized evaluation.

## Best Practices

1. **Start Small** - Test with 1 model, 1 dataset first
2. **Set Timeouts** - Prevent runaway experiments
3. **Use Tags** - Track experiment versions
4. **Enable Logging** - Use W&B for comparison
5. **Monitor Spend** - Set spend limits for paid APIs

## Related

- [Running the Agent](../user-guide/running-agent.md) - Single run usage
- [CLI Options](../configuration/cli-options.md) - All available options
