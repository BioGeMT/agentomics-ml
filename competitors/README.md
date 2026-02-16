# Agentomics Competitors

Run BioMLBench agents (including `zeroshot`) on Agentomics and BioMLBench datasets.
All commands below assume your current directory is `competitors/`.

## Prerequisites

If you need to generate Agentomics datasets first (genomic benchmarks, miRBench):

```bash
./scripts/prepare_agentomics_datasets.sh
```

Set credentials before running:

- In `competitors/config.yaml`:
  - If `enable_cost_tracking: false`, set `openrouter_key` to a real OpenRouter key.
  - If `enable_cost_tracking: true`, set `provisioning_key` to a real provisioning key.
- In `agentomics-ml/.env` set:
  - `WANDB_API_KEY`
  - `WANDB_PROJECT_NAME`
  - `WANDB_ENTITY`

If key setup is wrong, BioMLBench calls fail with `401 Missing Authentication header`.

## Setup

```bash
# Build only zeroshot image
./setup.sh --name zeroshot

# Or build all agents
./setup.sh
```

Then activate runtime env:

```bash
conda activate biomlbench-agents
```

## Prepare BioMLBench Datasets

Only full BioMLBench task IDs need explicit preparation (`polarishub/...`, `proteingym-dms/...`):

```bash
conda activate biomlbench-agents
./scripts/prepare_biomlbench_tasks.sh
```

Prepared BioMLBench data is stored in `competitors/data/`.
Task IDs are read from `competitors/config.yaml` (single source of truth).
For `polarishub/...`, the script first regenerates `leaderboard.csv` from Polaris Hub web tables,
then runs `biomlbench prepare` through a proxy-aware wrapper (`trust_env=True`, extended timeout).

Prepare only one group:

```bash
./scripts/prepare_biomlbench_tasks.sh proteingym
./scripts/prepare_biomlbench_tasks.sh polaris
```

Show prepared status:

```bash
./scripts/list_prepared_biomlbench_tasks.sh
```

## Run Zeroshot

```bash
# Runs zeroshot on every dataset listed in competitors/config.yaml
python run_competitors.py --agents zeroshot

# Runs zeroshot only on selected datasets
python run_competitors.py \
  --agents zeroshot \
  --datasets proteingym-dms/SPIKE_SARS2_Starr_2020_binding
```

Important behavior:

- If `--datasets` is not passed, the runner executes all datasets from `competitors/config.yaml`.
- `ago2_clash_hejret`-style names map to `agentomics/<name>`.
- `polarishub/...` and `proteingym-dms/...` are used directly as BioMLBench task IDs.
- ProteinGym support here is currently `zeroshot` only.
- For `proteingym-dms/...` with `zeroshot`, post-processing retrains/evaluates per fold and reports averaged metrics.

## Results

Outputs are saved under `competitors/results/{dataset}_{agent}_{timestamp}/`:

- `run.log`: output from `biomlbench run-agent`
- `run_artifacts/`: copied BioMLBench run directory
- `metrics.json`: local evaluation metrics
- `grade.json`: `biomlbench grade-sample` output
- `inference_stage.json`: inference reproducibility stage
- `duration.json`: run duration
- `cost.json`: cost data when `enable_cost_tracking: true`
