# Agentomics
NEWS: *Agentomics has been accepted into the ISMB 2026 Proceedings*

## Autonomous agentic system for supervised machine learning model development.

Made for biomedical data, Agentomics outperformed human experts and created new state-of-the-art models for problems in Protein Engineering, Drug Discovery, and Regulatory Genomics.


How it works
1) Input is a folder-based dataset split + optional data description
2) Agentomics autonomously experiments with various ML models and strategies
3) Output is a trained model ready for inference and a detailed PDF report summarizing the development process and achieved metrics

For more details see: [preprint](https://www.biorxiv.org/content/10.64898/2026.01.27.702049v1)

<p align="center">
  <img src="docs/assets/agentomics-overview.png" alt="agentomics overview" width="50%">
</p>

<!-- ## Try the DEMO -->
<!-- [![Try in Colab](https://colab.research.google.com/assets/colab-badge.svg)](https://colab.research.google.com/drive/1rxsGsIwxrE49E4rjzNh920s66UdG34xF?usp=sharing) -->
## Quick Start

```bash
git clone https://github.com/BioGeMT/agentomics-ml.git
cd agentomics-ml
cp .env.example .env
# Edit .env and set at least one supported provider credential

# Download example dataset
./scripts/download_example_dataset.sh

./run.sh
```

Recommended model: `gpt-5.1-codex-max`

Outputs are saved to `outputs/<agent_id>/`, including PDF reports in `outputs/<agent_id>/reports/pdf`.

### Installation Requirements

Agentomics can be run either:
- **(Recommended)** with [Docker](https://www.docker.com/)
- **Locally** with [Conda](https://docs.conda.io/)

### API Calls

Agentomics can be run via:
- your **local Codex subscription** via `codex login`
- a **supported provider API key** such as OpenRouter, OpenAI, Anthropic, or a configured OpenAI-compatible provider
- **local Ollama models** for offline/private runs


## Documentation

For more details visit **https://biogemt.github.io/agentomics-ml/**

## Key Features
- Generic: Agentomics can use folder-based inputs for classification and regression tasks.
- Secure: Agents execute code securely in Docker with read-only mounts to your file system and are only allowed to write in a Docker Volume.
- Reproducible: Outputs include models, scripts, and conda environments needed to run inference or re-train models with one bash command.
- Trustworthy: If you provide a test set, Agentomics fully abstracts LLMs from accessing it, allowing you to rely on programmaticly computed and reported test set metrics.
- Foundation models: Agentomics can leverage foundation models from huggingface for both embeddings and fine-tuning.
- Various LLM providers: OpenAI, Anthropic, OpenRouter, local Codex auth, local models via Ollama, or custom OpenAI-compatible providers
- Reliability: Thanks to our functional validators, Agentomics creates a working model 100% of the time (when using recommended settings).

## Run Output Structure Example

Each completed run is written to `outputs/<agent_id>/`. The key paths are:

```text
outputs/<agent_id>/
├── best_iteration_snapshot/
│   ├── model_training/
│   │   ├── train.py
│   │   └── training_artifacts/
│   ├── model_inference/
│   │   └── inference.py
│   ├── environment.yml
│   └── runtime_info/
├── run/
│   ├── shared/
│   │   ├── config.json
│   │   └── splits/
│   └── iteration_*/
└── reports/
    ├── markdown/
    └── pdf/
```

Use `best_iteration_snapshot/` for inference or re-training. `run/` keeps the
full iterative workspace, and `reports/` contains the human-readable summaries.

## Roadmap
Agentomics is in active development. We welcome any raised Issues and suggestions. You can also [Email Us](mailto:martinekvlastimil95@gmail.com).

Features coming soon:
- Run forking and continuing
- Better local model support and configuration
- Remote GPU support for GCP

## Reproducing publication results
See the [ismb_submission branch](https://github.com/BioGeMT/agentomics-ml/tree/ismb_post_review) README for instructions.

## Citation

If you use **Agentomics** in your work, please cite:

Martinek *et al.* (2026). 
*Agentomics: An Agentic System that Autonomously Develops Novel State-of-the-Art Solutions for Biomedical Machine Learning Tasks*.
bioRxiv (preprint) https://www.biorxiv.org/content/10.64898/2026.01.27.702049v1

## License

MIT. See `LICENSE`.
