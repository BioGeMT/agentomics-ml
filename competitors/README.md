# Agentomics Competitors Runner

Utilities for benchmarking the BioMLBench agents on Agentomics datasets with OpenRouter models.

## Files

- `config.yaml` – central configuration (repo URL, OpenRouter credentials, global time/step limits, dataset list, optional per-agent overrides).
- `setup.sh` – clones the patched BioMLBench fork, installs it, and materialises Agentomics tasks/data under the clone (requires `yq`, `jq`, Python with `pandas` and `scikit-learn`).
- `scripts/setup_tasks.py` – helper invoked by `setup.sh` to scaffold task configs, prepare data splits, and generate simple RMSE graders.
- `run_competitors.py` – drives `biomlbench run-agent` for every agent × dataset pair, evaluates submissions, tests inference reproducibility, and logs to W&B.
- `evaluation.py` – computes classification metrics using Agentomics evaluation logic and tests inference script reproducibility.
- `results/` – output directory populated with submissions/artefacts from each run.

## Workflow

1. Edit `config.yaml` to set your OpenRouter key/model, execution limits, agent names, and dataset names.
2. Set W&B credentials in `.env`: `WANDB_API_KEY`, `WANDB_PROJECT_NAME`, `WANDB_ENTITY`.
3. Run `./setup.sh` (idempotent) to create conda environment (`biomlbench-agents`), clone BioMLBench, install dependencies, and generate Agentomics task scaffolding.
4. Activate the environment: `conda activate biomlbench-agents`
5. Execute `python run_competitors.py` to run all agents on all datasets specified in config.

## Results Structure

All outputs are saved under `results/{dataset}_{agent}/`:

- `run.log` - Combined stdout/stderr from biomlbench run-agent
- `run_artifacts/` - Copied from BioMLBench runs directory:
  - `submission/submission.csv` - Agent's predictions
  - `submission/inference.py` - Inference script (if created by agent)
  - `submission/environment.yaml` - Dependencies (if created by agent)
- `metrics.json` - Computed classification metrics (ACC, AUC, F1, etc.)
- `metrics_results.csv` - Predictions in normalized format
- `metrics_test.csv` - Ground truth labels
- `label_mapping.json` - Label encoding mapping
- `inference_stage.json` - Inference reproducibility test result (missing/exists/runs/matches)
- `replay/` - Inference replay outputs for validation

## Evaluation

After each agent completes:
1. **Metrics computation** - Submission evaluated using Agentomics metrics (ACC, AUC, F1, etc.)
2. **Inference replay** - Tests if agent created reproducible inference script with dependencies
3. **W&B logging** - All metrics and inference stage logged to Weights & Biases for tracking
