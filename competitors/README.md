# Agentomics Competitors Runner

Utilities for benchmarking the BioMLBench agents on Agentomics datasets with OpenRouter models.

## Files

- `config.yaml` – central configuration (repo URL, OpenRouter credentials, global time/step limits, dataset list, optional per-agent overrides).
- `setup.sh` – clones the patched BioMLBench fork, installs it, and materialises Agentomics tasks/data under the clone (requires `yq`, `jq`, Python with `pandas` and `scikit-learn`).
- `scripts/setup_tasks.py` – helper invoked by `setup.sh` to scaffold task configs, prepare data splits, and generate simple RMSE graders.
- `run_competitors.py` – drives `biomlbench run-agent` for every agent × dataset pair declared in the config.
- `results/` – output directory populated with submissions/artefacts from each run.

## Workflow

1. Edit `config.yaml` to set your OpenRouter key/model, execution limits, agent names, and dataset names.
2. Run `./setup.sh` (idempotent) to clone BioMLBench, install it, and generate Agentomics task scaffolding + prepared data inside the clone.
3. Execute `python run_competitors.py` to run all agents on all datasets specified in config.

## Results Structure

All outputs are saved under `results/{dataset}_{agent}/`:

- `run.log` - Combined stdout/stderr from the benchmark runner
- `agentomics_{dataset}_{agent}_{timestamp}/` - BioMLBench run directory containing:
  - `submission/` - Agent's final submission files
  - `agent_code/` - Complete agent workspace with all generated code
  - `logs/` - Detailed execution logs from the agent
