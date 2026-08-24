# Agentomics
NEWS:
- *Agentomics to be presented at ISMB 2026*
- *Agentomics has been published in Bioinformatics journal* [link to published paper](https://academic.oup.com/bioinformatics/article/42/Supplement_1/btag250/8726289)
- *Agentomics now supports any data type and supplementary material*

## Autonomous agentic system for supervised machine learning model development.

Made for biomedical data, Agentomics outperformed human experts and created new state-of-the-art models for problems in Protein Engineering, Drug Discovery, and Regulatory Genomics.


How it works
1) Input is a folder-based dataset split + optional data description
2) Agentomics autonomously experiments with various ML models and strategies
3) Output is a trained model ready for inference and a detailed PDF report summarizing the development process and achieved metrics

For more details see: [link to published paper](https://academic.oup.com/bioinformatics/article/42/Supplement_1/btag250/8726289)

<p align="center">
  <img src="https://raw.githubusercontent.com/BioGeMT/Agentomics-ML/main/docs/assets/agentomics-overview.png" alt="agentomics overview" width="50%">
</p>

## Quick Start

Install [Docker](https://www.docker.com/), then:

```bash
pip install agentomics
```
Set at least one supported provider credential. For example:
OPENROUTER_API_KEY or OPENAI_API_KEY or ANTHROPIC_API_KEY
```bash
export OPENROUTER_API_KEY=...   
```
Download an example dataset into ./datasets

To see all available examples add the --list option
```bash
agentomics-download-dataset
```
Start an Agentomics run and follow instructions
```bash
agentomics-run
```

Recommended model: `gpt-5.1-codex-max`

Outputs are saved to `outputs/<agent_id>/`, including PDF reports in `outputs/<agent_id>/reports/pdf`.

See [Installation](https://biogemt.github.io/agentomics-ml/getting-started/installation/),
[Datasets](https://biogemt.github.io/agentomics-ml/user-guide/datasets/),
[CLI Options](https://biogemt.github.io/agentomics-ml/configuration/cli-options/), and
[Running Inference](https://biogemt.github.io/agentomics-ml/user-guide/inference/) for details.

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
- Various LLM providers: OpenAI, OpenRouter, or local models via Ollama
- Reliability: Thanks to our functional validators, Agentomics creates a working model 100% of the time (when using recommended settings).
- Supplementary materials: Give the agent dataset-specific papers, helper scripts, and foundation-model guidance. See the [supplementary materials documentation](https://biogemt.github.io/agentomics-ml/user-guide/datasets/#supplementary-optional).

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
│   └── runtime_info/
│       └── environment.yml
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
- Better local model support and configuration
- Remote GPU support for GCP

## Citation

If you use **Agentomics** in your work, please cite:

Martinek *et al.* (2026). 
*Agentomics: An Agentic System that Autonomously Develops Novel State-of-the-Art Solutions for Biomedical Machine Learning Tasks*.
Bioinformatics (https://doi.org/10.1093/bioinformatics/btag250)

## License

MIT. See `LICENSE`.
