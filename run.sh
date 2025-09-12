set -euo pipefail

AGENTOMICS_ARGS=()
LOCAL_MODE=false

while [[ $# -gt 0 ]]; do
    case $1 in
        --model)
            AGENTOMICS_ARGS+=(--model "$2")
            shift 2
            ;;
        --dataset)
            AGENTOMICS_ARGS+=(--dataset "$2")
            shift 2
            ;;
        --iterations)
            AGENTOMICS_ARGS+=(--iterations "$2")
            shift 2
            ;;
        --val-metric)
            AGENTOMICS_ARGS+=(--val-metric "$2")
            shift 2
            ;;
        --user-prompt)
            AGENTOMICS_ARGS+=(--user-prompt "$2")
            shift 2
            ;;
        --local)
            LOCAL_MODE=true
            shift
            ;;
        *)
            shift
            ;;
    esac
done

if [ "$LOCAL_MODE" = true ]; then
    RED='\033[0;31m'
    NOCOLOR='\033[0m'
    echo -e "${RED}Running in local mode - this is only recommended if you run in a non-vulnerable environment!${NOCOLOR}"
    echo "For Docker mode (secure run), re-run without the --local flag."
    
    if ! conda env list | grep -q "agentomics-prepare-env"; then
        conda env create -f environment_prepare.yaml -q
    fi

    mkdir -p prepared_datasets
    conda run -n agentomics-prepare-env python src/prepare_datasets.py --prepare-all

    if ! conda env list | grep -q "agentomics-env"; then
        conda env create -f environment.yaml -q
    fi

    eval "$(conda shell.bash hook)"
    conda activate agentomics-env
    python src/run_agent_interactive.py "${AGENTOMICS_ARGS[@]}"

    mkdir -p outputs/best_run_files outputs/reports
    cp -r workspace/snapshots/. outputs/best_run_files/
    cp -r workspace/reports/. outputs/reports/
else
    docker build -t agentomics_prepare_img -f Dockerfile.prepare .
    docker run \
        --rm \
        -it \
        --name agentomics_prepare_cont \
        -v $(pwd):/repository \
        agentomics_prepare_img

    docker build -t agentomics_img .
    docker volume create temp_agentomics_volume

    docker run \
        -it \
        --rm \
        --name agentomics_cont \
        -v $(pwd):/repository:ro \
        -v temp_agentomics_volume:/workspace \
        agentomics_img "${AGENTOMICS_ARGS[@]}"

    # Copy best-run files and report
    mkdir -p outputs/best_run_files outputs/reports
    docker run --rm -v temp_agentomics_volume:/source -v $(pwd)/outputs:/dest busybox cp -r /source/snapshots/. /dest/best_run_files/

    # Copy reports from all iterations
    docker run --rm -v temp_agentomics_volume:/source -v $(pwd)/outputs:/dest busybox cp -r /source/reports/. /dest/reports/

    docker volume rm temp_agentomics_volume
fi
