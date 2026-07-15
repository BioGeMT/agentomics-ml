# Agentomics-ML

**Autonomous AI agent for supervised machine learning model development on omics data**

[:material-rocket-launch: Quick Start](getting-started/quick-start.md){ .md-button .md-button--primary }
[:material-file-document: Preprint](https://www.biorxiv.org/content/10.64898/2026.01.27.702049v1){ .md-button }
[:material-web: Website](https://agentomicsml.com/){ .md-button }

---

## What is Agentomics-ML?

Agentomics-ML is an autonomous AI agent that develops machine learning models for omics data. Given a dataset, it produces:

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
| **Any LLM** | Works with OpenAI, Anthropic, OpenRouter, Codex/ChatGPT OAuth, or local models via Ollama |
| **Any Dataset** | Supports folder-based inputs for classification or regression tasks |
| **Secure Execution** | Docker containers with read-only access to code and isolated execution |
| **Reproducible** | Outputs include trained models, scripts, and conda environments |

## Deployment

Agentomics runs in Docker. The Python commands launch the pre-built image and
mount only the files needed by each workflow. [Local LLMs](configuration/providers.md#ollama-local-models)
remain available through Ollama running on the host.

## Main Commands

| Command | Purpose |
|--------|---------|
| `agentomics-run` | Run the full agent workflow |
| `agentomics-retrain` | Re-train a model with new data |
| `agentomics-inference` | Run predictions on new data |
| `agentomics-check-dataset` | Validate a dataset's format before a run |

## Quick Example

```bash
export OPENROUTER_API_KEY="your-key-here"

agentomics-run
```

The agent will guide you through selecting a model, dataset, and run parameters interactively.

## License

MIT. See the [LICENSE](https://github.com/BioGeMT/Agentomics-ML/blob/main/LICENSE).
