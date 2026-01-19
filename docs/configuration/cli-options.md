# CLI Options

Complete reference for `run.sh` command-line options.

## Basic Options

| Option | Description | Default |
|--------|-------------|---------|
| `--model <name>` | LLM model to use | Interactive selection |
| `--dataset <name>` | Dataset name | Interactive selection |
| `--iterations <n>` | Number of iterations | 10 |
| `--val-metric <metric>` | Validation metric to optimize | ACC |
| `--timeout <seconds>` | Time limit for entire run | None |

## Deployment Options

| Option | Description |
|--------|-------------|
| `--pull-images` | Pull pre-built Docker images from Docker Hub |
| `--local` | Run without Docker (uses conda) |
| `--cpu-only` | Disable GPU acceleration |
| `--ollama` | Use local Ollama for LLM |

## Listing Options

| Option | Description |
|--------|-------------|
| `--list-models` | Show available LLM models |
| `--list-datasets` | Show prepared datasets |
| `--list-metrics` | Show available validation metrics |
| `--help` | Show help message |

## Advanced Options

| Option | Description |
|--------|-------------|
| `--user-prompt <text>` | Custom prompt for the agent |
| `--foundation-model-type <type>` | Pre-download foundation models (dna, rna, protein, molecule) |
| `--use-provisioning-key` | Use OpenRouter temporary API key |
| `--split-allowed-iterations <n>` | Iterations that can modify train/val split |
| `--exploration-iterations <n>` | Baseline exploration iterations |

## Examples

### Basic Run

```bash
./run.sh --model openai/gpt-4o --dataset breast_cancer --iterations 10
```

### Quick Start with Pre-built Images

```bash
./run.sh --pull-images
```

### Local Mode

```bash
./run.sh --local --model openai/gpt-4o --dataset my_data
```

### With Time Limit

```bash
./run.sh --timeout 3600 --model openai/gpt-4o --dataset my_data
```

### Custom Optimization Goal

```bash
./run.sh --user-prompt "Focus on interpretable models only" --model openai/gpt-4o
```

### Using Ollama

```bash
./run.sh --ollama --model llama3.1:70b
```

### CPU Only

```bash
./run.sh --cpu-only --model openai/gpt-4o --dataset my_data
```

### Pre-download Foundation Models

```bash
./run.sh --foundation-model-type protein --model openai/gpt-4o
```

## Validation Metrics

Available metrics for `--val-metric`:

**Classification:**

- `ACC` - Accuracy
- `AUROC` - Area Under ROC Curve
- `AUPRC` - Area Under Precision-Recall Curve
- `F1` - F1 Score
- `PRECISION` - Precision
- `RECALL` - Recall
- `MCC` - Matthews Correlation Coefficient
- `BALANCED_ACC` - Balanced Accuracy

**Regression:**

- `MSE` - Mean Squared Error
- `RMSE` - Root Mean Squared Error
- `MAE` - Mean Absolute Error
- `R2` - R-squared
- `PEARSON` - Pearson Correlation

## Environment Variables

CLI options can also be set via environment variables. See [Environment Variables](environment.md).

## Model Names

Model names are provider-specific:

- **OpenRouter:** `openai/gpt-4o`, `anthropic/claude-3.5-sonnet`, etc.
- **OpenAI:** `gpt-4o`, `gpt-4-turbo`, etc.
- **Anthropic:** `claude-3-5-sonnet-20241022`, etc.
- **Ollama:** `llama3.1:70b`, `mixtral:8x7b`, etc.

Use `--list-models` to see available models for your configured providers.

## Exit Codes

| Code | Meaning |
|------|---------|
| 0 | Success |
| 1 | General error |
| 2 | Invalid arguments |
| 130 | Interrupted (Ctrl+C) |
