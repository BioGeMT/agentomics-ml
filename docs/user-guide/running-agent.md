# Running the Agent

The agent is launched through `run.sh`. You can run it two ways:

- **Docker mode (recommended)** — `docker run ... biogemt/agentomics:latest [run.sh options]`
- **Local mode** — `./run.sh [options]` directly with Conda

Both accept the same `run.sh` options; in Docker mode they go after the image
name. See [Installation](../getting-started/installation.md) for the full
`docker run` invocation (mounts, env, GPU). The examples below show the
`./run.sh` form for brevity — prefix them with your `docker run ...` line to run
in a container.

## Interactive Mode

With no run arguments (and an attached terminal), the agent prompts you for the
essentials:

```bash
# Local
./run.sh

# Docker (note -it for an interactive terminal)
docker run --rm -it ... biogemt/agentomics:latest
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
./run.sh \
  --model openai/gpt-5.1-codex-max \
  --dataset breast_cancer \
  --iterations 10
```

For non-interactive fresh runs, provide at least `--model` and `--dataset`. If
you omit `--iterations`, the default is 5.

## Options

`run.sh` exposes many flags — model and provider selection, time limits, split
and exploration controls, forking, and more. See **[CLI Options](../configuration/cli-options.md)**
for the complete reference, or run:

```bash
./run.sh --help
```

A few common ones:

```bash
# Set a deadline for the whole run (whichever of this or --iterations hits first)
./run.sh --timeout 7200 --model openai/gpt-5.1-codex-max --dataset my_data

# Control how long the agent explores baselines / may re-split the data
./run.sh --split-allowed-iterations 1 --exploration-iterations 4 ...

# Override the optimization goal
./run.sh --user-prompt "Only use simple models like logistic regression" ...
```

See [Custom Prompts](../configuration/custom-prompts.md) for prompt tuning and
[Forking a Run](forking.md) for branching off an existing run's checkpoint.

## What Happens During a Run

1. **Dataset Preparation** - Validates the selected dataset and writes the
   training/validation inputs to `prepared_datasets/`.
2. **Iterative Development** - The agent runs exploration, training, and
   evaluation cycles, scoring each iteration on the validation metric.
3. **Best-Model Snapshot** - The best-performing iteration is tracked and copied
   to `best_iteration_snapshot/`.
4. **Report Generation** - Per-iteration and final reports are written to
   `reports/markdown/` and `reports/pdf/`.

Results are written to the run's workspace — `outputs/<agent_id>/` in local
mode, or the directory mounted at `/workspace` in Docker mode. See
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
