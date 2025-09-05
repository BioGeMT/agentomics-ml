set -euo pipefail

mkdir -p workspace/runs
mkdir -p workspace/datasets

if ! conda env list | grep -q "agentomics-prepare-env"; then
    conda env create -f environment_prepare.yaml -q
fi

mkdir -p prepared_datasets
conda run -n agentomics-prepare-env python src/prepare_datasets.py --prepare-all

if ! conda env list | grep -q "agentomics-env"; then
    conda env create -f environment.yaml -q
fi

# Activate environment and run with interactive terminal support
eval "$(conda shell.bash hook)"
conda activate agentomics-env
python src/run_agent_interactive.py
