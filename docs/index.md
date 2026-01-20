# Agentomics-ML

**Autonomous AI agent for supervised machine learning model development on omics data**

[:material-rocket-launch: Quick Start](getting-started/quick-start.md){ .md-button .md-button--primary }
[:material-file-document: Preprint](https://arxiv.org/abs/2506.05542){ .md-button }
[:material-web: Website](https://agentomicsml.com/){ .md-button }

---

## What is Agentomics-ML?

Agentomics-ML is an autonomous AI agent that develops machine learning models for omics data. Given a raw dataset, it produces:

- **A trained model** ready to run inference on new data
- **A detailed report** summarizing the model development process and evaluation metrics

## How It Works

Agentomics-ML works like an ML engineer:

1. **Explores data** before designing a model
2. **Considers domain information** from dataset descriptions
3. **Chooses proper data representation** (encoding, normalization, feature selection)
4. **Designs and trains models**, including custom neural networks
5. **Works iteratively**, reacting to issues like overfitting and underfitting based on validation metrics
6. **Produces working scripts** with their conda environments

## Key Features

| Feature | Description |
|---------|-------------|
| **Any LLM** | Works with OpenAI, Anthropic, OpenRouter, or local models via Ollama |
| **Any Dataset** | Supports classification or regression datasets in CSV format |
| **Secure Execution** | Docker containers with read-only access to code and isolated execution |
| **Reproducible** | Outputs include trained models, scripts, and conda environments |

## Deployment Options

Choose the setup that works best for you:

| Mode | Description | Best For |
|------|-------------|----------|
| [Docker + Pull Images](getting-started/installation.md#docker-with-pre-built-images) | Fastest setup - pulls pre-built images | Getting started quickly |
| [Docker + Local Build](getting-started/installation.md#docker-with-local-build) | Build images locally | Custom modifications |
| [Local Mode](getting-started/installation.md#local-mode-no-docker) | No Docker, uses conda directly | Development, Google Colab |
| [Local LLMs](configuration/providers.md#ollama-local-models) | Run with Ollama | Privacy, offline use |

## Main Scripts

| Script | Purpose |
|--------|---------|
| `run.sh` | Run the full agent workflow |
| `train.sh` | Re-train a model with new data |
| `inference.sh` | Run predictions on new data |

## Quick Example

```bash
# Set your API key
export OPENROUTER_API_KEY="your-key-here"

# Run the agent (pulls Docker images automatically)
./run.sh --pull-images
```

The agent will guide you through selecting a model, dataset, and run parameters interactively.

## Try It Now

[:material-google: Google Colab Demo](https://colab.research.google.com/drive/1rxsGsIwxrE49E4rjzNh920s66UdG34xF?usp=sharing){ .md-button .md-button--primary }

## License

Agentomics-ML is open source. See the [GitHub repository](https://github.com/BioGeMT/Agentomics-ML) for details.
