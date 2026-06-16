# Installation

Agentomics-ML runs in two modes:

- **Docker mode (recommended)** — run the pre-built image. Everything (conda
  environments, dataset preparation, the agent) runs inside the container.
- **Local mode** — run `./run.sh` directly on your machine with Conda.

## Docker Mode

### Requirements

- [Docker](https://docs.docker.com/get-docker/) installed and running

### Setup

The repository is baked into the image, so no clone is needed — only a `.env`
file with your API key, a folder with your dataset, and an output directory:

```bash
# .env with at least one provider key, e.g.:
# OPENROUTER_API_KEY=...

mkdir -p outputs/my_run_1

docker run --rm -it \
  --env-file .env \
  -v "$(pwd)/datasets:/repository/datasets" \
  -v "$(pwd)/outputs/my_run_1:/workspace" \
  -e HOST_UID=$(id -u) -e HOST_GID=$(id -g) \
  biogemt/agentomics:latest \
  --dataset my_dataset
```

Everything after the image name is passed to `run.sh`. With no run arguments,
the run is interactive, prompting for model, dataset, and iterations.

!!! note "One run per output mount"
    `/workspace` holds the contents of a **single** run
    (`best_iteration_snapshot/`, `run/`, `reports/`, ...) — not an `outputs/`
    parent. Mount a fresh, empty directory for each run (e.g.
    `outputs/my_run_2`) so runs don't overwrite each other. This differs from
    local mode, where runs are auto-organized under `outputs/<agent_id>/`.

- `-v ...:/repository/datasets` makes your datasets visible to the agent. Select
  one with `--dataset <name>`.
- `-v ...:/workspace` is where this run's results are written. After the run,
  find them in the mounted host directory (`./outputs/my_run_1`).
- `HOST_UID`/`HOST_GID` make the output files owned by you instead of root.
- For GPU runs, add `--gpus all` (see [GPU Settings](../developer/gpu-settings.md)).
- For the `codex` provider, also mount your codex auth:
  `-v ~/.codex:/mnt/codex-host:ro`.

### Building the image yourself

```bash
git clone https://github.com/BioGeMT/Agentomics-ML.git
cd Agentomics-ML
docker build -t agentomics .
```

Then use `agentomics` in place of `biogemt/agentomics:latest` above.

---

## Local Mode (No Docker)

Runs `run.sh` directly with Conda. Environments are created automatically on
first run, and results are organized under `outputs/<agent_id>/`.

!!! warning "Security Notice"
    In local mode, agent-generated code executes directly on your machine.
    Prefer Docker mode unless you are in an already-isolated environment
    (e.g., a container or Google Colab).

### Requirements

- [Conda](https://docs.conda.io/en/latest/miniconda.html) installed

### Setup

```bash
git clone https://github.com/BioGeMT/Agentomics-ML.git
cd Agentomics-ML

# Set your API key: export it, or put it in a .env file in the repo root
# (loaded automatically)
export OPENROUTER_API_KEY="your-key-here"

./run.sh
```

---

## Google Colab

The easiest way to try Agentomics-ML without any local setup.

[:material-google: Open in Google Colab](https://colab.research.google.com/drive/1rxsGsIwxrE49E4rjzNh920s66UdG34xF?usp=sharing){ .md-button .md-button--primary }

The Colab notebook runs Agentomics-ML in local mode.

---

## Ollama (Local LLMs)

Run with local models using Ollama for privacy or offline use.

### Requirements

- [Ollama](https://ollama.ai/) installed and running on the host

### Local Mode

```bash
export OLLAMA_BASE_URL=http://localhost:11434/v1
./run.sh --provider ollama --model <ollama-model> --dataset <dataset>
```

### Docker Mode

Add `--network host` so the container can reach the Ollama server on the host:

```bash
docker run --rm -it --network host \
  --env-file .env \
  -e OLLAMA_BASE_URL=http://localhost:11434/v1 \
  -v "$(pwd)/datasets:/repository/datasets" \
  -v "$(pwd)/outputs/my_run_1:/workspace" \
  biogemt/agentomics:latest \
  --provider ollama --model <ollama-model> --dataset <dataset>
```

---

## CPU-Only Mode

Disable GPU acceleration with `--cpu-only` (works in both modes):

```bash
./run.sh --cpu-only
```

In Docker mode, simply omitting `--gpus` also keeps the run on CPU.

---

## Comparison Table

| Mode | Requirements | Isolation | Best For |
|------|--------------|-----------|----------|
| Docker | Docker | High | Recommended default |
| Local | Conda | None | Development, already-isolated machines |
| Google Colab | Browser | Colab VM | Trying it out |

## Next Steps

- [Running the Agent](../user-guide/running-agent.md) - Learn all run.sh options
- [LLM Providers](../configuration/providers.md) - Configure different LLM providers
- [GPU Settings](../developer/gpu-settings.md) - NVIDIA GPU setup
