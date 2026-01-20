# Agentomics-ML

**Autonomous AI agent for supervised machine learning model development on omics data**

[![Try in Colab](https://colab.research.google.com/assets/colab-badge.svg)](https://colab.research.google.com/drive/1rxsGsIwxrE49E4rjzNh920s66UdG34xF?usp=sharing)

Given a raw dataset, Agentomics-ML autonomously generates a trained model ready for inference and a detailed report summarizing the development process.

## Quick Start

```bash
git clone https://github.com/BioGeMT/Agentomics-ML.git
cd Agentomics-ML
cp .env.example .env
# Edit .env and set at least one API key (OPENROUTER_API_KEY, OPENAI_API_KEY, or ANTHROPIC_API_KEY)

# Download example datasets
conda env create -f competitors/environment_datasets.yaml
conda activate agentomics-datasets
python src/utils/create_datasets.py
conda deactivate

./run.sh --pull-images
```

## Documentation

For complete documentation, visit **https://biogemt.github.io/agentomics-ml/**

## Key Features

- Any LLM: OpenAI, Anthropic, OpenRouter, or local models via Ollama
- Any dataset: classification or regression in CSV format
- Secure: Docker containers with isolated execution
- Reproducible: outputs include models, scripts, and conda environments

## Main Scripts

- `run.sh`: run the full agent workflow
- `train.sh`: re-train a model with new data
- `inference.sh`: run predictions on new data

## Links

- [Documentation](https://biogemt.github.io/Agentomics-ML/)
- [Preprint](https://arxiv.org/abs/2506.05542)
- [Website](https://agentomicsml.com/)
- [Google Colab Demo](https://colab.research.google.com/drive/1rxsGsIwxrE49E4rjzNh920s66UdG34xF?usp=sharing)
