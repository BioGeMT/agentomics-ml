# Agentomics-ML

**Autonomous AI agent for supervised machine learning model development on biomedical datasets**

[![Try in Colab](https://colab.research.google.com/assets/colab-badge.svg)](https://colab.research.google.com/drive/1rxsGsIwxrE49E4rjzNh920s66UdG34xF?usp=sharing)

Given a raw CSV dataset, Agentomics-ML autonomously generates a trained model ready for inference and a detailed report summarizing the development process.

<img src="docs/assets/agentomics-overview.png" alt="Agentomics-ML overview" width="50%">

**Reviewers:** See the [ismb branch](https://github.com/BioGeMT/Agentomics-ML/tree/ismb) for the Bioinformatics submission materials.

## Quick Start

```bash
git clone https://github.com/BioGeMT/Agentomics-ML.git
cd Agentomics-ML
cp .env.example .env
# Edit .env and set at least one API key (OPENROUTER_API_KEY or OPENAI_API_KEY)

# Download example datasets
./download_example_datasets.sh

./run.sh --pull-images
```

Recommended model: `gpt-5.1-codex-max`.

Results are saved to `outputs/<agent_id>/`. PDF reports live in `outputs/<agent_id>/pdf_reports/`.

## Documentation

For complete documentation, visit **https://biogemt.github.io/agentomics-ml/**

## Key Features

- Any LLM: OpenAI, OpenRouter, or local models via Ollama
- Any dataset: classification or regression in CSV format
- Secure: Docker containers with isolated execution
- Reproducible: outputs include models, scripts, and conda environments

## Main Scripts

- `run.sh`: run the full agent workflow
- `train.sh`: re-train a model with new data
- `inference.sh`: run predictions on new data

## Roadmap

- More data types (beyond CSV datasets)
- Remote GPU support for GCP
- Better local model support and configuration

## License

MIT. See `LICENSE`.

## Links

- [Documentation](https://biogemt.github.io/Agentomics-ML/)
- [Preprint](https://arxiv.org/abs/2506.05542)
- [Website](https://agentomicsml.com/)
- [Google Colab Demo](https://colab.research.google.com/drive/1rxsGsIwxrE49E4rjzNh920s66UdG34xF?usp=sharing)
