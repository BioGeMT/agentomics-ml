# CLI Options

Complete reference for `run.sh` command-line options.

## Basic Options

| Option | Description | Default |
|--------|-------------|---------|
| `--model <name>` | LLM model to use | Interactive selection |
| `--dataset <name>` | Dataset name | Interactive selection |
| `--iterations <n>` | Number of iterations | Prompted in interactive mode (default 5) |
| `--val-metric <metric>` | Validation metric to optimize | Task-based default (`AUROC` for classification, `MAE` for regression) |
| `--timeout <seconds>` | Time limit for entire run | None |
| `--run-python-timeout <seconds>` | Timeout in seconds for each run_python tool execution - this will determine the maximum training time | `21600` (6 hours) |

The run stops when either the iteration count is reached or the timeout expires.

## Deployment Options

| Option | Description |
|--------|-------------|
| `--build-images` | Build Docker images locally |
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
| `--iteration-plan-model <name>` | LLM model used for generating the iteration plan (defaults to `--model`) |
| `--foundation-model-type <type>` | Pre-download foundation models (`dna`, `rna`, `protein`, `molecule`, `all`) |
| `--use-provisioning-key` | Use OpenRouter temporary API key |
| `--spend-limit <n>` | Spend limit for provisioning key (requires `--use-provisioning-key`) |
| `--split-allowed-iterations <n>` | Iterations that can modify train/val split (default 1) |
| `--exploration-iterations <n>` | Baseline exploration iterations (default 4) |
| `--run-python-timeout <seconds>` | Per-training timeout for `run_python` tool (default 21600) |

## Forking

| Option | Description |
|--------|-------------|
| `--fork-from-run <path>` | Path to a source `outputs/<run_id>` directory. Creates a new run branching off from a checkpoint in that run. Most other options are optional when forking — omitting them inherits from the source run config. |
| `--fork-from-iteration <n>` | Iteration number to fork from (default: latest). Only used with `--fork-from-run`. |
| `--fork-from-step <step>` | Step ID to fork from, e.g. `data_split` or `model_training` (default: latest checkpoint). Only used with `--fork-from-run`. |

When `--iterations`, `--split-allowed-iterations`, or `--exploration-iterations` are provided alongside `--fork-from-run`, they are interpreted as *additional* iterations from the fork point rather than absolute totals.

See [Forking a Run](../user-guide/forking.md) for a full guide and examples.

## Examples

### Basic Run

```bash
./run.sh --model openai/gpt-4 --dataset breast_cancer --iterations 10
```

### Quick Start with Pre-built Images

```bash
./run.sh
```

### Local Mode

```bash
./run.sh --local --model openai/gpt-4 --dataset my_data
```

### With Time Limit

```bash
./run.sh --timeout 3600 --model openai/gpt-4 --dataset my_data
```

### Custom Optimization Goal

```bash
./run.sh --user-prompt "Focus on interpretable models only" --model openai/gpt-4
```

### Using Ollama

```bash
./run.sh --ollama
```

### CPU Only

```bash
./run.sh --cpu-only --model openai/gpt-4 --dataset my_data
```

### Pre-download Foundation Models

```bash
./run.sh --foundation-model-type protein --model openai/gpt-4
```

### Run with locally built Docker images

```bash
./run.sh --build-images 
```

## Validation Metrics

Available metrics for `--val-metric`:

**Classification:**

- `ACC` - Accuracy
- `AUROC` - Area Under ROC Curve
- `AUPRC` - Area Under Precision-Recall Curve
- `F1` - F1 Score (macro)
- `LOG_LOSS` - Log loss
- `MCC` - Matthews Correlation Coefficient

**Regression:**

- `MSE` - Mean Squared Error
- `RMSE` - Root Mean Squared Error
- `MAE` - Mean Absolute Error
- `MAPE` - Mean Absolute Percentage Error
- `PEARSON` - Pearson Correlation
- `SPEARMAN` - Spearman Correlation
- `R2` - R-squared

## Environment Variables

API keys and logging settings come from environment variables or `.env`. See
[Environment Variables](environment.md).

## Model Names

Model names are provider-specific. Use `--list-models` to see available models for your configured providers.

## Exit Codes

| Code | Meaning |
|------|---------|
| 0 | Success |
| 1 | General error |
| 2 | Invalid arguments |
| 130 | Interrupted (Ctrl+C) |
