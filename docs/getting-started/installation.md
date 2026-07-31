# Installation

Agentomics-ML runs in Docker. Everything—including Conda environments, dataset
preparation, and agent-generated code—runs inside the container. You drive it
through the installed `agentomics-run` command, which launches the container,
mounts the files each workflow needs, and collects the results.

## Requirements

- [Docker](https://docs.docker.com/get-docker/) installed and running
- Python 3.11+ to install the CLI

## Setup

Install the CLI from PyPI, then run it. No repository clone is required — the
container image is pulled automatically on the first run:

```bash
python3 -m pip install agentomics

# Provide at least one provider key, exported or in a ./.env file, e.g.:
export OPENROUTER_API_KEY=...

agentomics-run --dataset my_dataset
```

With no run arguments, the run is interactive, prompting for model, dataset, and
iterations.

The launcher takes care of the container plumbing for you:

- Datasets are read from `./datasets`; select one with `--dataset <name>`.
- Results are written to `./outputs/<agent_id>/` (a fresh directory per run).
- Output files are owned by you, not root.
- A `.env` file in the current directory and exported provider keys are passed
  through automatically.
- GPU access is enabled by default; pass `--cpu-only` to disable it (see
  [GPU Settings](../developer/gpu-settings.md)).
- For the `codex` provider, `~/.codex` is mounted read-only automatically.

By default the launcher uses the image matching the installed package version:
`biogemt/agentomics:<installed-package-version>`. Use `--image <name>` only to
select another image explicitly, such as a locally built development image.

### Building the image yourself

```bash
git clone https://github.com/BioGeMT/Agentomics-ML.git
cd Agentomics-ML
docker build -t agentomics .
```

The build uses the repository's `main` branch by default. To build the image
from another branch, pass its Git URL and branch with `REPOSITORY_SOURCE`:

```bash
docker build \
  --build-arg REPOSITORY_SOURCE=https://github.com/BioGeMT/Agentomics-ML.git#my-branch \
  -t agentomics .
```

Contributors testing changes from a working tree should use `--dev`; see
[Local Development](../developer/development.md).

## Ollama (Local LLMs)

Run with local models using Ollama for privacy or offline use.

### Requirements

- [Ollama](https://ollama.ai/) installed and running on the host

Set `OLLAMA_BASE_URL` and select the `ollama` provider. The launcher enables
host networking automatically so the container can reach the Ollama server:

```bash
export OLLAMA_BASE_URL=http://localhost:11434/v1
agentomics-run --provider ollama --model <ollama-model> --dataset <dataset>
```

---

## CPU-Only Mode

Disable GPU acceleration with `--cpu-only`:

```bash
agentomics-run --cpu-only
```

## Next Steps

- [Running the Agent](../user-guide/running-agent.md) - Learn all agentomics-run options
- [LLM Providers](../configuration/providers.md) - Configure different LLM providers
- [GPU Settings](../developer/gpu-settings.md) - NVIDIA GPU setup
