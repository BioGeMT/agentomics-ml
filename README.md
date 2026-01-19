# Agentomics-ML

**Autonomous AI agent for supervised machine learning model development on omics data**

[![Documentation](https://img.shields.io/badge/docs-GitHub%20Pages-blue)](https://biogemT.github.io/Agentomics-ML/)
[![arXiv](https://img.shields.io/badge/arXiv-2506.05542-b31b1b.svg)](https://arxiv.org/abs/2506.05542)
[![Try in Colab](https://colab.research.google.com/assets/colab-badge.svg)](https://colab.research.google.com/drive/1rxsGsIwxrE49E4rjzNh920s66UdG34xF?usp=sharing)

Given a raw dataset, Agentomics-ML autonomously generates a trained model ready for inference and a detailed report summarizing the development process.

## Quick Start

```bash
git clone https://github.com/BioGeMT/Agentomics-ML.git
cd Agentomics-ML
export OPENROUTER_API_KEY="your-key-here"
./run.sh --pull-images
```

## Documentation

For complete documentation, visit **[biogemT.github.io/Agentomics-ML](https://biogemT.github.io/Agentomics-ML/)**

- [Quick Start Guide](https://biogemT.github.io/Agentomics-ML/getting-started/quick-start/)
- [Installation Options](https://biogemT.github.io/Agentomics-ML/getting-started/installation/) (Docker, Local, Colab, Ollama)
- [User Guide](https://biogemT.github.io/Agentomics-ML/user-guide/running-agent/)
- [Configuration](https://biogemT.github.io/Agentomics-ML/configuration/cli-options/)
- [How It Works](https://biogemT.github.io/Agentomics-ML/how-it-works/architecture/)

## Key Features

| Feature | Description |
|---------|-------------|
| **Any LLM** | OpenAI, Anthropic, OpenRouter, or local models via Ollama |
| **Any Dataset** | Classification or regression in CSV format |
| **Secure** | Docker containers with isolated execution |
| **Reproducible** | Outputs include models, scripts, and conda environments |

## Main Scripts

| Script | Purpose |
|--------|---------|
| `run.sh` | Run the full agent workflow |
| `train.sh` | Re-train a model with new data |
| `inference.sh` | Run predictions on new data |

## Links

- [Documentation](https://biogemT.github.io/Agentomics-ML/)
- [Preprint](https://arxiv.org/abs/2506.05542)
- [Website](https://agentomicsml.com/)
- [Google Colab Demo](https://colab.research.google.com/drive/1rxsGsIwxrE49E4rjzNh920s66UdG34xF?usp=sharing)
