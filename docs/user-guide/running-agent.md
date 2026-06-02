# Running the Agent

The main entry point for Agentomics-ML is `run.sh`. This guide covers both interactive and non-interactive usage.

## Interactive Mode

Running without arguments launches interactive mode:

```bash
./run.sh
```

Docker mode expects a `.env` file in the repo root (copy `.env.example`).

You'll be prompted to select:

1. **LLM Model** - Choose from available models
2. **Dataset** - Select a dataset
3. **Iterations** - Number of optimization cycles (default prompt: 5)

The validation metric is not prompted interactively; pass `--val-metric` to override the task-based default (`AUROC` for classification, `MAE` for regression).

## Non-Interactive Mode

Supply parameters directly to skip prompts:

```bash
./run.sh \
  --model openai/gpt-4 \
  --dataset breast_cancer \
  --iterations 10
```

For non-interactive fresh runs, provide at least `--model` and `--dataset`. If you omit `--iterations`, the default is 5.

## Common Options

| Option | Description | Example |
|--------|-------------|---------|
| `--model` | LLM model to use | `--model openai/gpt-4` |
| `--provider` | Provider to use when multiple providers are configured | `--provider openai` |
| `--dataset` | Dataset name | `--dataset my_data` |
| `--iterations` | Number of iterations | `--iterations 15` |
| `--val-metric` | Validation metric (optional, task-based default if omitted) | `--val-metric AUROC` |
| `--timeout` | Time limit in seconds | `--timeout 3600` |
| `--run-python-timeout` | Timeout in seconds for each run_python tool execution (see [CLI options](../configuration/cli-options.md)) | `--run-python-timeout 43200` |
| `--use-provisioning-key` | Use a provisioning key for OpenRouter | `--use-provisioning-key` |
| `--spend-limit` | Spend limit for provisioning key | `--spend-limit 25` |

## Listing Available Options

```bash
# List available models
./run.sh --list-models

# List available datasets
./run.sh --list-datasets

# List available metrics
./run.sh --list-metrics
```

## Deployment Flags

| Flag | Description |
|------|-------------|
| `--build-images` | Build Docker images locally |
| `--local` | Run without Docker (uses conda) |
| `--cpu-only` | Disable GPU acceleration |
| `--ollama` | Enable Docker host networking for a host Ollama server |

## Advanced Options

### Data Split and Exploration Controls

```bash
./run.sh --split-allowed-iterations 1 --exploration-iterations 4
```

`--split-allowed-iterations` controls how many early iterations are allowed to resplit
train/validation (ignored if you provide a `validation/` split). `--exploration-iterations`
controls how long the agent spends on baseline/exploration models.

### Time Limits

Set a deadline for the entire run:

```bash
./run.sh --timeout 7200  # 2 hour limit
```

Set timeout for each training execution (default is 6 hours):

```bash
./run.sh --run-python-timeout 43200  # 12 hours per training run
```

You can also set a separate split deadline:

```bash
./run.sh --split-timeout 3600  # stop allowing split changes after 1 hour
```

### Custom User Prompt

Override the default optimization goal:

```bash
./run.sh --user-prompt "Only use simple models like logistic regression"
```

See [Custom Prompts](../configuration/custom-prompts.md) for more details.

## Forking a Run

You can branch off from a checkpoint in an existing run and continue as an independent new run:

```bash
./run.sh \
  --fork-from-run outputs/my_source_run \
  --iterations 3
```

Most options (`--model`, `--user-prompt`, etc.) are optional when forking — they are inherited from the source run if omitted. `--iterations` means *N more from the fork point*, not a total. Dataset and validation metric are always inherited and cannot be changed.

See [Forking a Run](forking.md) for the full guide.

## Full Help

```bash
./run.sh --help
```

## What Happens During a Run

1. **Dataset Preparation** - Validates and prepares data in `datasets/` and `test_datasets/`
2. **Iterative Development** - Agent runs exploration, training, and evaluation cycles
3. **Snapshot Best Model** - Tracks the best-performing iteration
4. **Final Evaluation** - Tests on held-out test set (if provided)
5. **Output Results** - Saves everything to `outputs/<agent_id>/`

## Monitoring Progress

During execution, you'll see:

- Current iteration number
- Agent step (exploration, training, etc.)
- Validation metrics after each iteration
- Best iteration tracking

## Stopping a Run

Press `Ctrl+C` to stop. The agent will attempt to save current progress.

## Next Steps

- [CLI Options](../configuration/cli-options.md) - Complete flag reference
- [Understanding Outputs](outputs.md) - What the agent produces
- [Custom Prompts](../configuration/custom-prompts.md) - Customize agent behavior
