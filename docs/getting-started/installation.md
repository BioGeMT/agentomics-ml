# Installation

Agentomics-ML supports multiple deployment options. Choose the one that best fits your needs.

## Prerequisites

All installation methods require:

```bash
git clone https://github.com/BioGeMT/agentomics-ml.git
cd agentomics-ml
```

Running Agentomics also requires a configured [LLM provider](../configuration/providers.md). External providers may charge for usage and receive data-derived context; use local Ollama when that context must stay on your machine.

## Docker with Pre-built Images

**Default mode** - Downloads pre-built images from Docker Hub.

### Requirements

- [Docker](https://docs.docker.com/get-docker/) installed and running

### Setup

```bash
# Create a .env file (required for Docker mode)
cp .env.example .env
# Edit .env and set at least one provider key

# Run with pre-built images
./run.sh
```

The images will be downloaded automatically on first run. All subsequent runs will use the cached images.

---

## Docker with Local Build

**Alternative mode** - Builds Docker images locally.

### Requirements

- [Docker](https://docs.docker.com/get-docker/) installed and running

### Setup

```bash
# Create a .env file (required for Docker mode)
cp .env.example .env
# Edit .env and set at least one provider key

# Run while building images locally
./run.sh --build-images
```

With `--build-images`, the Docker images are built locally before the run starts. This takes a few minutes but only needs to be repeated when dependencies or Dockerfiles change.

---

## Local Mode (No Docker)

**For development or Google Colab** - Runs directly with conda.

!!! warning "Security Notice"
    Local mode executes code without containerization. Only use in secure environments like Google Colab or your own isolated container.

### Requirements

- [Conda](https://docs.conda.io/en/latest/miniconda.html) installed

### Setup

```bash
# Set your API key (export or .env)
export OPENROUTER_API_KEY="your-key-here"

# Run in local mode
./run.sh --local
```

Conda environments will be created automatically.

---

## Google Colab

The easiest way to try Agentomics-ML without any local setup.

[:material-google: Open in Google Colab](https://colab.research.google.com/drive/1rxsGsIwxrE49E4rjzNh920s66UdG34xF?usp=sharing){ .md-button .md-button--primary }

The Colab notebook uses local mode automatically.

---

## Ollama (Local LLMs)

Run with local models using Ollama for privacy or offline use.

### Requirements

- [Ollama](https://ollama.ai/) installed and running
- Docker (recommended) or conda

### Docker Mode Setup

1. Ensure Ollama is running on the host.
2. Make the Ollama provider selectable and choose it explicitly:

    ```bash
    export OLLAMA_BASE_URL=http://localhost:11434/v1
    ./run.sh --ollama --provider ollama --model <ollama-model> --dataset <dataset>
    ```

Docker mode connects to the URL configured in `src/utils/providers/configured_providers.yaml`
(default: `http://localhost:11434/v1`) and uses host networking when `--ollama` is passed.

### Local Mode Setup

For local mode, set the Ollama base URL in `src/utils/providers/configured_providers.yaml`
to `http://localhost:11434/v1`, then run:

```bash
export OLLAMA_BASE_URL=http://localhost:11434/v1
./run.sh --local --provider ollama --model <ollama-model> --dataset <dataset>
```

---

## CPU-Only Mode

Disable GPU acceleration:

```bash
./run.sh --cpu-only
```

Works with both Docker and local modes.

---

## Comparison Table

| Mode | Docker Required | First-run work | Generated-code isolation | Best For |
|------|-----------------|----------------|--------------------------|----------|
| Docker + Pull Images | Yes | Downloads several-GB images | Container | Quick start |
| Docker + Local Build | Yes | Builds images locally | Container | Custom builds |
| Local Mode | No | Creates Conda environments | None | Development, Colab |
| Google Colab | No | Notebook-managed setup | None | Trying it out |
| Ollama | Depends | Downloads local models | Varies | Privacy, offline |

**Note on privacy:** Generated-code isolation (Docker vs local Conda) affects filesystem access. LLM provider choice (external vs local Ollama) affects where data-derived context is sent. Both are independent security decisions.

## Next Steps

- [Running the Agent](../user-guide/running-agent.md) - Learn all run.sh options
- [LLM Providers](../configuration/providers.md) - Configure different LLM providers
- [GPU Settings](../developer/gpu-settings.md) - NVIDIA GPU setup
