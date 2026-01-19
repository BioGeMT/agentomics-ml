# Running the Agent

The main entry point for Agentomics-ML is `run.sh`. This guide covers both interactive and non-interactive usage.

## Interactive Mode

Running without arguments launches interactive mode:

```bash
./run.sh
```

You'll be prompted to select:

1. **LLM Model** - Choose from available models
2. **Dataset** - Select a prepared dataset
3. **Iterations** - Number of optimization cycles (default: 10)
4. **Validation Metric** - Metric to optimize (ACC, AUROC, etc.)

## Non-Interactive Mode

Supply parameters directly to skip prompts:

```bash
./run.sh \
  --model openai/gpt-5.2-codex \
  --dataset breast_cancer \
  --iterations 10 \
  --val-metric ACC
```

## Common Options

| Option | Description | Example |
|--------|-------------|---------|
| `--model` | LLM model to use | `--model openai/gpt-5.2-codex` |
| `--dataset` | Dataset name | `--dataset my_data` |
| `--iterations` | Number of iterations | `--iterations 15` |
| `--val-metric` | Validation metric | `--val-metric AUROC` |
| `--timeout` | Time limit in seconds | `--timeout 3600` |

## Listing Available Options

```bash
# List available models
./run.sh --list-models

# List prepared datasets
./run.sh --list-datasets

# List available metrics
./run.sh --list-metrics
```

## Deployment Flags

| Flag | Description |
|------|-------------|
| `--pull-images` | Pull pre-built Docker images |
| `--local` | Run without Docker (uses conda) |
| `--cpu-only` | Disable GPU acceleration |
| `--ollama` | Use local Ollama models |

## Advanced Options

### Foundation Models

Pre-download domain-specific foundation models:

```bash
./run.sh --foundation-model-type dna
```

Available types: `dna`, `rna`, `protein`, `molecule`

### Time Limits

Set a deadline for the entire run:

```bash
./run.sh --timeout 7200  # 2 hour limit
```

### Custom User Prompt

Override the default optimization goal:

```bash
./run.sh --user-prompt "Only use simple models like logistic regression"
```

See [Custom Prompts](../configuration/custom-prompts.md) for more details.

## Full Help

```bash
./run.sh --help
```

## What Happens During a Run

1. **Dataset Preparation** - Validates and prepares data in `prepared_datasets/`
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
