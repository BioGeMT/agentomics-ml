# CLI Options

Complete reference for `agentomics-run` command-line options.

The command launches the Agentomics Docker image and forwards these options to
the container. See [Installation](../getting-started/installation.md).

## Basic Options

| Option | Description | Default |
|--------|-------------|---------|
| `--model <name>` | LLM model to use | Interactive selection |
| `--provider <name>` | Provider to use when multiple providers are configured | Prompted if multiple providers are available in an interactive terminal |
| `--dataset <name>` | Dataset name | Interactive selection |
| `--iterations <n>` | Number of iterations | Prompted in interactive mode (default 5) |
| `--val-metric <metric>` | Validation metric to optimize | Task-based default (`AUROC` for classification, `MAE` for regression) |
| `--task-type <classification\|regression>` | Task type to use while preparing the selected raw dataset | Dataset config or interactive preparation prompt |
| `--timeout <seconds>` | Time limit for entire run | None |
| `--split-timeout <seconds>` | Time limit after which the agent may no longer change train/validation split | None |
| `--run-python-timeout <seconds>` | Timeout in seconds for each run_python tool execution - this will determine the maximum training time | `21600` (6 hours) |

The run stops when either the iteration count is reached or the timeout expires.

## Execution Options

| Option | Description |
|--------|-------------|
| `--cpu-only` | Disable GPU acceleration |
| `--test` | Run the integrated test suite |
| `--image <name>` | Docker image to launch (default: `biogemt/agentomics:<installed-package-version>`; use this option for an explicit override) |
| `--dev` | Build from the current local working tree before launching; cannot be combined with `--image` |
| `--datasets-dir <path>` | Host directory to read datasets from (default: `./datasets`) |
| `--workspace-dir <path>` | Host directory for this run's output (default: `./outputs/<agent_id>`) |

For Docker and Ollama setup, see [Installation](../getting-started/installation.md).

## Listing Options

| Option | Description |
|--------|-------------|
| `--list-models` | Show available LLM models |
| `--list-datasets` | Show available datasets |
| `--list-metrics` | Show available validation metrics |
| `--help` | Show help message |

## Advanced Options

| Option | Description |
|--------|-------------|
| `--user-prompt <text>` | Custom prompt for the agent |
| `--iteration-plan-model <name>` | LLM model used for generating the iteration plan (defaults to `--model`) |
| `--use-provisioning-key` | Use OpenRouter temporary API key |
| `--spend-limit <n>` | Spend limit for provisioning key (requires `--use-provisioning-key`) |
| `--verbosity <summary\|full>` | How much agent interaction detail is printed during the run (default: `full`) |
| `--disable-training-reporting` | Disable the TrainingReporter helper that emits structured training progress updates from the agent's training script |
| `--split-allowed-iterations <n>` | Iterations that can modify train/val split (default 1) |
| `--exploration-iterations <n>` | Baseline exploration iterations (default 4) |
| `--tags <tag...>` | Space-separated tags for W&B logging |
| `--conda-export-mode <full\|yaml>` | Best-iteration environment format: `full` stores `environment.yml` plus a packed `environment.tar.gz`; `yaml` stores only the portable YAML definition (default: `full`) |

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
agentomics-run --model openai/gpt-5.1-codex-max --dataset breast_cancer --iterations 10
```

### Interactive Run

```bash
agentomics-run
```

### With Time Limit

```bash
agentomics-run --timeout 3600 --model openai/gpt-5.1-codex-max --dataset my_data
```

### Custom Optimization Goal

```bash
agentomics-run --user-prompt "Focus on interpretable models only" --model openai/gpt-5.1-codex-max
```

### Using Ollama

```bash
export OLLAMA_BASE_URL=http://localhost:11434/v1
agentomics-run --provider ollama --model llama3.1 --dataset my_data
```

### CPU Only

```bash
agentomics-run --cpu-only --model openai/gpt-5.1-codex-max --dataset my_data
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
