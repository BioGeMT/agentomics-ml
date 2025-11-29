# Agentomics Competitors

Run the BioMLBench agents on agentomics datasets with OpenRouter models.

## Workflow

1. Edit `config.yaml` to set your OpenRouter key/model, execution limits, agent names, and dataset names.
2. Set W&B credentials in `.env`: `WANDB_API_KEY`, `WANDB_PROJECT_NAME`, `WANDB_ENTITY`.
3. Run `./setup.sh` to create conda environment (`biomlbench-agents`), clone BioMLBench, install dependencies, and generate Agentomics tasks.
4. Activate the environment: `conda activate biomlbench-agents`
5. Execute `python run_competitors.py` to run all agents on all datasets specified in config.

## Results 

All outputs are saved under `results/{dataset}_{agent}/`:

- `run.log` - from biomlbench run-agent
- `run_artifacts/` - from BioMLBench runs directory:
  - `submission/submission.csv` - Agent's predictions
  - `submission/inference.py` 
  - `submission/environment.yaml` 
- `metrics.json` - Computed classification metrics (ACC, AUC, F1, etc.)
- `inference_stage.json` - Inference reproducibility result (missing/exists/runs/matches)
- `replay/` - Inference replay outputs

