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