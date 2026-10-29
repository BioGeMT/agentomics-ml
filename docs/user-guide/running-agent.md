# Running the Agent

The agent is launched through `agentomics-run`, which starts the Agentomics
Docker image and mounts the required datasets and workspace. See
[Installation](../getting-started/installation.md) for setup.

## Interactive Mode

With no run arguments (and an attached terminal), the agent prompts you for the
essentials:

```bash
agentomics-run
```

You'll be prompted to select:

1. **LLM Model** - Choose from available models
2. **Dataset** - Select a dataset
3. **Iterations** - Number of optimization cycles (default prompt: 5)

The validation metric is not prompted interactively; pass `--val-metric` to
override the task-based default (`AUROC` for classification, `MAE` for
regression).

## Non-Interactive Mode

Supply parameters directly to skip the prompts:

```bash
agentomics-run \
  --model openai/gpt-5.1-codex-max \
  --dataset breast_cancer \
  --iterations 10
```

For non-interactive fresh runs, provide at least `--model` and `--dataset`. If
you omit `--iterations`, the default is 5.

## Options

`agentomics-run` exposes many flags — model and provider selection, time limits, split
and exploration controls, forking, and more. See **[CLI Options](../configuration/cli-options.md)**
for the complete reference, or run:

```bash
agentomics-run --help
```

A few common ones:

```bash
# Set a deadline for the whole run (whichever of this or --iterations hits first)
agentomics-run --timeout 7200 --model openai/gpt-5.1-codex-max --dataset my_data

# Control how long the agent explores baselines / may re-split the data
agentomics-run --split-allowed-iterations 1 --exploration-iterations 4 ...

# Override the optimization goal
agentomics-run --user-prompt "Only use simple models like logistic regression" ...
```

See [Custom Prompts](../configuration/custom-prompts.md) for prompt tuning and
[Forking a Run](forking.md) for branching off an existing run's checkpoint.

## What Happens During a Run

1. **Dataset Preparation** - Validates the selected dataset and prepares the
   training/validation inputs inside the run workspace (`run/shared/`). Your
   source `datasets/` files are read-only and are never modified.
2. **Iterative Development** - The agent runs exploration, training, and
   evaluation cycles, scoring each iteration on the validation metric.
3. **Best-Model Snapshot** - The best-performing iteration is tracked and copied
   to `best_iteration_snapshot/`.
4. **Report Generation** - Per-iteration and final reports are written to
   `reports/markdown/` and `reports/pdf/`.

Results are written to the run's host workspace, `outputs/<agent_id>/` by
default, which is mounted at `/workspace` in the container. See
[Understanding Outputs](outputs.md) for the full layout.

To score a finished run against a labeled held-out set, run inference with
`--label-col`; see [Running Inference](inference.md).

## Monitoring Progress

During execution you'll see:

- Current iteration number
- Agent step (exploration, training, etc.)
- Validation metrics after each iteration
- Best-iteration tracking

## Stopping a Run

Press `Ctrl+C` to stop. The agent attempts to save current progress before
exiting.

## Next Steps

- [CLI Options](../configuration/cli-options.md) - Complete flag reference
- [Understanding Outputs](outputs.md) - What the agent produces
- [Custom Prompts](../configuration/custom-prompts.md) - Customize agent behavior
