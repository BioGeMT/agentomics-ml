#!/usr/bin/env bash

# Ensure we are running under bash even if invoked via sh/zsh
if [ -z "${BASH_VERSION:-}" ]; then
  exec /usr/bin/env bash "$0" "$@"
fi

set -euo pipefail

AGENTOMICS_ARGS=()
LOCAL_MODE=false
TEST_MODE=false
CPU_ONLY=false
OLLAMA=false

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
        --ollama)
            OLLAMA=true
            shift
            ;;
        --test)
            TEST_MODE=true
            shift
            ;;
        --cpu-only)
            CPU_ONLY=true
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
    python src/run_agent_interactive.py ${AGENTOMICS_ARGS+"${AGENTOMICS_ARGS[@]}"}

    mkdir -p outputs/best_run_files outputs/reports
    cp -r workspace/snapshots/. outputs/best_run_files/
    cp -r workspace/reports/. outputs/reports/
else
    docker build -t agentomics_prepare_img -f Dockerfile.prepare .
    docker run \
        -u $(id -u):$(id -g) \
        --rm \
        -it \
        --name agentomics_prepare_cont \
        -v "$(pwd)":/repository \
        agentomics_prepare_img

    docker build -t agentomics_img .
    docker volume create temp_agentomics_volume

    GPU_FLAGS=()
    if [ "$CPU_ONLY" = false ]; then
        GPU_FLAGS+=(--gpus all)
        GPU_FLAGS+=(--env NVIDIA_VISIBLE_DEVICES=all)
    fi
    OLLAMA_FLAGS=()
    if [ "$OLLAMA" = true ]; then
        OLLAMA_FLAGS+=(--add-host=host.docker.internal:host-gateway)
    fi

    if [ "$TEST_MODE" = true ]; then
        docker run \
            -it \
            --rm \
            --name agentomics_test_cont \
            ${GPU_FLAGS[@]+"${GPU_FLAGS[@]}"} \
            ${OLLAMA_FLAGS[@]+"${OLLAMA_FLAGS[@]}"} \
            -v "$(pwd)/src":/repository/src:ro \
            -v "$(pwd)/test":/repository/test:ro \
            -v "$(pwd)/prepared_datasets":/repository/prepared_datasets:ro \
            -v "$(pwd)/.env":/repository/.env:ro \
            -v temp_agentomics_volume:/workspace \
            --entrypoint /opt/conda/envs/agentomics-env/bin/python \
            agentomics_img -m test.run_all_tests
    else
        docker run \
            -it \
            --rm \
            --name agentomics_cont \
            ${GPU_FLAGS[@]+"${GPU_FLAGS[@]}"} \
            ${OLLAMA_FLAGS[@]+"${OLLAMA_FLAGS[@]}"} \
            -v "$(pwd)/src":/repository/src:ro \
            -v "$(pwd)/prepared_datasets":/repository/prepared_datasets:ro \
            -v "$(pwd)/.env":/repository/.env:ro \
            -v temp_agentomics_volume:/workspace \
            agentomics_img ${AGENTOMICS_ARGS+"${AGENTOMICS_ARGS[@]}"}

        # Copy best-run files and report
        mkdir -p outputs/best_run_files outputs/reports
        docker run --rm -u $(id -u):$(id -g) -v temp_agentomics_volume:/source -v $(pwd)/outputs:/dest busybox cp -r /source/snapshots/. /dest/best_run_files/

        # Copy reports from all iterations
        docker run --rm -u $(id -u):$(id -g) -v temp_agentomics_volume:/source -v $(pwd)/outputs:/dest busybox cp -r /source/reports/. /dest/reports/
    fi

    docker volume rm temp_agentomics_volume
fi
