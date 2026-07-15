# Forking a Run

Forking lets you branch off from a completed checkpoint in an existing run and continue from that point as an independent new run. The forked run inherits the full history — trained models, data splits, conda environment — but any new iterations it produces are completely separate from the source.

## When to Use Forking

- **Try a different strategy mid-run**: fork after iteration 3 to explore a new direction without losing the work already done.
- **Extend a finished run**: add more iterations to a run that has already completed.
- **Compare branches**: fork the same checkpoint twice with different prompts or models to run a controlled comparison.

## Basic Usage

Point `--fork-from-run` at an `outputs/<run_id>` directory. The launcher mounts
the source workspace read-only in the container:

```bash
agentomics-run \
  --fork-from-run outputs/my_source_run \
  --model openai/gpt-5.1-codex-max \
  --iterations 3
```

The fork starts from the latest checkpoint in that run and continues for 3 more
iterations. The launcher mounts the source run read-only and writes the fork to
a new `outputs/<agent_id>/` directory, leaving the source run untouched.

## Choosing a Checkpoint

By default, the fork starts from the latest checkpoint (the end of the last completed iteration). You can pick an earlier point with `--fork-from-step` and/or `--fork-from-iteration`:

```bash
# Fork from the end of iteration 2
agentomics-run --fork-from-run outputs/my_run --fork-from-iteration 2

# Fork from after the data_split step in iteration 1
agentomics-run --fork-from-run outputs/my_run \
  --fork-from-iteration 1 \
  --fork-from-step data_split
```

If `--fork-from-iteration` is omitted, the most recent iteration for the given step is used. If both are omitted, the latest checkpoint overall is used.

## Iteration Counting

When `--iterations` is passed to a fork, it means **N additional iterations from the fork point**, not a total. If the source run completed 4 iterations and you pass `--iterations 2`, the fork will run iterations 4 and 5 (total 6).

If `--iterations` is omitted, the source run's original total is reused.

The same relative semantics apply to `--split-allowed-iterations` and `--exploration-iterations`.

## Inherited vs Overridable Options

Most run options are **optional** when forking — omitting them reuses the value from the source run:

| Option | Behaviour on fork |
|--------|------------------|
| `--model` | Inherited if omitted |
| `--provider` | Inherited if omitted |
| `--iterations` | Relative increment if provided; source total if omitted |
| `--split-allowed-iterations` | Relative increment if provided; inherited if omitted |
| `--exploration-iterations` | Relative increment if provided; inherited if omitted |
| `--run-python-timeout` | Inherited if omitted |
| `--timeout` | Not inherited — no run timeout unless passed (applies only to the forked run's own runtime) |
| `--split-timeout` | Not inherited — no split timeout unless passed (applies only to the forked run's own runtime) |
| `--user-prompt` | Inherited if omitted |
| `--tags` | Inherited if omitted |
| `--dataset` | **Always inherited, cannot be changed** |
| `--val-metric` | **Always inherited, cannot be changed** |

Dataset and validation metric are locked to keep all iterations comparable across the fork lineage.

## What the Fork Copies

When a fork is set up, the following happens before the new run starts:

1. The source workspace state is copied, excluding generated reports/logs and untracked Conda environments.
2. The git history in the run directory is checked out at the requested checkpoint — files added in later commits are removed.
3. Absolute paths stored in step outputs are rewritten to point to the new workspace.
4. The shared conda environment is renamed for the new run ID and updated from the stored `environment.yml`.

The forked run then continues from that state exactly as if the original run had stopped there.

**Note on supplementary materials**: Forked runs reference the same dataset directory as the source run. Any changes to dataset files (including `supplementary/`) will be visible to the fork.

Split directories may contain symbolic links pointing to container paths. Forks
run in the same Docker layout, so those links continue to resolve.

## Example: Extend a Completed Run

```bash
# Original run finished after 5 iterations
agentomics-run \
  --fork-from-run outputs/finished_run \
  --iterations 5   # run 5 more, for a total of 10
```

## Example: Compare Two Strategies from the Same Checkpoint

```bash
# Fork A: aggressive regularization
agentomics-run \
  --fork-from-run outputs/base_run \
  --fork-from-iteration 3 \
  --user-prompt "Focus on heavily regularized models to reduce overfitting" \
  --iterations 4

# Fork B: ensemble approach
agentomics-run \
  --fork-from-run outputs/base_run \
  --fork-from-iteration 3 \
  --user-prompt "Try ensemble methods combining multiple base learners" \
  --iterations 4
```

Both forks start from identical state at iteration 3 and produce independent outputs you can compare directly.

## Related

- [CLI Options](../configuration/cli-options.md#forking) — Complete fork flag reference
- [Understanding Outputs](outputs.md) — What each run produces
- [Running the Agent](running-agent.md) — General usage guide
