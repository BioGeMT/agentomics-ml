set -euo pipefail

AGENTOMICS_ARGS=""
while [[ $# -gt 0 ]]; do
    case $1 in
        --model)
            AGENTOMICS_ARGS="$AGENTOMICS_ARGS --model $2"
            shift 2
            ;;
        --dataset)
            AGENTOMICS_ARGS="$AGENTOMICS_ARGS --dataset $2"
            shift 2
            ;;
        --iterations)
            AGENTOMICS_ARGS="$AGENTOMICS_ARGS --iterations $2"
            shift 2
            ;;
        --val-metric)
            AGENTOMICS_ARGS="$AGENTOMICS_ARGS --val-metric $2"
            shift 2
            ;;
        *)
            shift
            ;;
    esac
done

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
python src/run_agent_interactive.py $AGENTOMICS_ARGS

cp -r workspace/snapshots/. outputs/best_run_files/
cp -r workspace/reports/. outputs/reports/
