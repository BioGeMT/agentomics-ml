# Local Development

Install Agentomics in editable mode (one time setup) from the repository root:

```bash
pip install -e .
```

## Running tests

See [How to run tests in CONTRIBUTING.md](https://github.com/BioGeMT/Agentomics-ML/blob/main/CONTRIBUTING.md#how-to-run-tests).

## Running application changes locally

Use `--dev` with any Docker-backed command.
These must be run from the repository root.

```bash
agentomics-run --dev --dataset <name>
agentomics-inference --dev <options>
agentomics-retrain --dev <options>
```

## How it works

Before starting the requested operation, it builds the current working tree as `agentomics:dev` and automatically uses it

```bash
docker build --build-arg REPOSITORY_SOURCE=. -t agentomics:dev .
```

The build includes uncommitted and untracked files except those excluded by
`.dockerignore`. It runs every time, while Docker's layer cache keeps unchanged
layers fast.

`--dev` cannot be combined with `--image`. Commands without `--dev` continue to
use the image matching the installed Agentomics version (and pull the corresponding image).

## A/B testing

`scripts/ab_test.py` runs Agentomics variants side by side. A variant is a
commit plus `agentomics-run` options, so variants can differ in code, in
options, or in both. Copy the example config, edit it, and run it from the
repository root:

```bash
cp scripts/ab_test_example.toml my_test.toml
python scripts/ab_test.py my_test.toml
```

```toml
replicates = 1

[shared]
commit = "main"
dataset = "breast_cancer"
provider = "codex"
iterations = 3

[[variants]]
model = "gpt-5.6-luna"

[[variants]]
model = "gpt-5.6-sol"
user-prompt = "Develop a classical machine learning model (no neural networks) that generalizes well to new unseen data."
cpu-only = true

[[variants]]
commit = "ab-orchestrator"
model = "gpt-5.6-luna"
```

`[shared]` holds the options every variant gets, and each `[[variants]]` block
is one variant; the first is variant 0. A variant's options override
`[shared]`. `commit` is a commit hash, branch, or tag and is required, either
in `[shared]` or in every variant. Every other key is an `agentomics-run`
option without the leading `--`; `true` passes a switch such as `cpu-only`.
`replicates` defaults to 1 and must come before `[shared]`.

The script builds a Docker image for each commit once and runs each variant
with that commit's code `replicates` times. Runs alternate between variants:
replicate 1 of every variant, then replicate 2, and so on. Each run writes its
workspace and log to `outputs/ab_tests/<config name>-<timestamp>/`. The script
sets `--image`, `--workspace-dir`, and `--tags` itself.

When [W&B is configured](../configuration/environment.md), the runs go to
their own W&B project, `ab-<config name>-<timestamp>`, are grouped by variant
(`variant-<index>`), and are tagged `variant:<index>` and `commit:<commit>`.
Runs on commits from before `WANDB_RUN_GROUP` support are not grouped.
