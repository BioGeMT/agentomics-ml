#!/usr/bin/env bash

# Ensure we are running under bash even if invoked via sh/zsh
if [ -z "${BASH_VERSION:-}" ]; then
  exec /usr/bin/env bash "$0" "$@"
fi

set -euo pipefail

# Color definitions
RED='\033[0;31m'
GREEN='\033[0;32m'
NOCOLOR='\033[0m'

AGENTOMICS_ARGS=()
LOCAL_MODE=false
TEST_MODE=false
CPU_ONLY=false
OLLAMA=false

show_help() {
    cat << EOF
Usage: ./run.sh [OPTIONS]

Orchestrates the Agentomics training and evaluation process. By default, it runs in Docker containers.
Use --local to run with a local Conda environment.

Required Arguments (for non-interactive runs):
  --model <name>      The LLM model name (e.g., 'openai/gpt-4').
  --dataset <name>    The short identifier for the prepared dataset (e.g., 'breast_cancer').
  --iterations <N>    Number of iterations to run the agent (e.g., 5).
  --split-allowed-iterations <N>    Number of initial iterations that are allowed to (re)split the data into train/validation (e.g., 1).
  --val-metric <name> The metric to optimize (e.g., 'ACC').
  --user-prompt <str> The main prompt/goal for the agent.
                      (Default: "Create the best possible machine learning model that will generalize to new unseen data.")

Operational Flags:
  --local             Run the project using local Conda environments instead of Docker.
  --test              Run the project's integrated test suite.
                      (Note: Only supported in Docker mode, not in local Conda mode.)
  --cpu-only          Force Docker/Conda to run using CPU only (skip GPU configuration).
  --ollama            Enable support for an Ollama server running on the host machine.
  -h, --help          Show this help message and exit.

Listing Flags (Run the script with only one of these):
  --list-models       List models available via the configured provider and exit.
  --list-datasets     List all prepared datasets and exit.
  --list-metrics      List all available validation metrics and exit.

Environment:
  API keys read from 'src/utils/providers/configured_providers.yaml' must be set as
  environment variables in your host environment (e.g., in a shell session or .env file)
  to be injected into the Docker container.

Output:
  Results are copied from the temporary workspace to the local 'outputs/<AGENT_ID>' directory.
EOF
}

while [[ $# -gt 0 ]]; do
    case $1 in
        -h|--help)
            show_help
            exit 0
            ;;
        --list-models)
            AGENTOMICS_ARGS+=(--list-models)
            shift
            ;;
        --list-datasets)
            AGENTOMICS_ARGS+=(--list-datasets)
            shift
            ;;
        --list-metrics)
            AGENTOMICS_ARGS+=(--list-metrics)
            shift
            ;;
        --root-privileges)
            AGENTOMICS_ARGS+=(--root-privileges)
            shift
            ;;
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
        --split-allowed-iterations)
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
          # Catch unrecognized arguments
          if [[ "$1" == -* ]]; then
                echo -e "${RED}Error: Unrecognized argument or flag: $1${NOCOLOR}" >&2
                echo "Please run ./run.sh --help for the available arguments." >&2
                exit 1
            fi
            shift
            ;;
    esac
done

if [ "$LOCAL_MODE" = true ]; then
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

    AGENT_ID=$(python src/utils/create_user.py)
    export AGENT_ID
    python src/run_agent_interactive.py ${AGENTOMICS_ARGS+"${AGENTOMICS_ARGS[@]}"}
    export PYTHONPATH=./src
    python src/run_logging/evaluate_log_test.py --workspace-dir ../workspace

    mkdir -p outputs/${AGENT_ID}/best_run_files outputs/${AGENT_ID}/reports
    cp -r ../workspace/snapshots/${AGENT_ID}/. outputs/${AGENT_ID}/best_run_files/
    cp -r ../workspace/reports/${AGENT_ID}/. outputs/${AGENT_ID}/reports/
else
    echo "Building the run image"
    docker build --progress=quiet -t agentomics_img -f Dockerfile .
    echo "Build done"
    AGENT_ID=$(docker run --rm -u $(id -u):$(id -g) -v "$(pwd)":/repository:ro --entrypoint \
               /opt/conda/envs/agentomics-env/bin/python agentomics_img /repository/src/utils/create_user.py)

    echo "Building the data preparation image"
    docker build --progress=quiet -t agentomics_prepare_img -f Dockerfile.prepare .
    echo "Build done"
    docker run \
        -u $(id -u):$(id -g) \
        --rm \
        -it \
        --name agentomics_prepare_cont_${AGENT_ID} \
        -v "$(pwd)":/repository \
        agentomics_prepare_img

    docker volume create temp_agentomics_volume_${AGENT_ID}
    trap "docker volume rm temp_agentomics_volume_${AGENT_ID}" EXIT

    GPU_FLAGS=()
    if [ "$CPU_ONLY" = false ]; then
        GPU_FLAGS+=(--gpus all)
        GPU_FLAGS+=(--env NVIDIA_VISIBLE_DEVICES=all)
    fi
    OLLAMA_FLAGS=()
    if [ "$OLLAMA" = true ]; then
        OLLAMA_FLAGS+=(--add-host=host.docker.internal:host-gateway)
    fi

    PROVIDERS_CONFIG_FILE="src/utils/providers/configured_providers.yaml"
    API_KEY_NAMES=$(grep -E 'apikey:' "$PROVIDERS_CONFIG_FILE" | grep -o '\${[^}]*}' | tr -d '${}' | sort -u)
    DOCKER_API_KEY_ENV_VARS=()
    for KEY_NAME in $API_KEY_NAMES; do
        if [ -n "${!KEY_NAME:-}" ]; then
            DOCKER_API_KEY_ENV_VARS+=(-e "$KEY_NAME=${!KEY_NAME}")
            echo "Adding API key env var to docker: $KEY_NAME"
        fi
    done

    if [ "$TEST_MODE" = true ]; then
        docker run \
            -it \
            --rm \
            --name agentomics_test_cont_${AGENT_ID} \
            --env-file $(pwd)/.env \
            -e AGENT_ID=${AGENT_ID} \
            ${GPU_FLAGS[@]+"${GPU_FLAGS[@]}"} \
            ${OLLAMA_FLAGS[@]+"${OLLAMA_FLAGS[@]}"} \
            ${DOCKER_API_KEY_ENV_VARS[@]+"${DOCKER_API_KEY_ENV_VARS[@]}"} \
            -v "$(pwd)/src":/repository/src:ro \
            -v "$(pwd)/test":/repository/test:ro \
            -v "$(pwd)/prepared_datasets":/repository/prepared_datasets:ro \
            -v temp_agentomics_volume_${AGENT_ID}:/workspace \
            --entrypoint /opt/conda/envs/agentomics-env/bin/python \
            agentomics_img -m test.run_all_tests
    else
        docker run \
            -it \
            --rm \
            --name agentomics_cont_${AGENT_ID} \
            --env-file $(pwd)/.env \
            -e AGENT_ID=${AGENT_ID} \
            ${GPU_FLAGS[@]+"${GPU_FLAGS[@]}"} \
            ${OLLAMA_FLAGS[@]+"${OLLAMA_FLAGS[@]}"} \
            ${DOCKER_API_KEY_ENV_VARS[@]+"${DOCKER_API_KEY_ENV_VARS[@]}"} \
            -v "$(pwd)/src":/repository/src:ro \
            -v "$(pwd)/prepared_datasets":/repository/prepared_datasets:ro \
            -v temp_agentomics_volume_${AGENT_ID}:/workspace \
            agentomics_img ${AGENTOMICS_ARGS+"${AGENTOMICS_ARGS[@]}"}

        #TODO only run this if test set exists
        echo "Running final evaluation on test set"
        docker run \
            --rm \
            --name agentomics_test_eval_cont_${AGENT_ID} \
            --env-file $(pwd)/.env \
            -e AGENT_ID=${AGENT_ID} \
            -e PYTHONPATH=/repository/src \
            -v "$(pwd)/src":/repository/src:ro \
            -v "$(pwd)/prepared_datasets":/repository/prepared_datasets:ro \
            -v "$(pwd)/prepared_test_sets":/repository/prepared_test_sets:ro \
            -v temp_agentomics_volume_${AGENT_ID}:/workspace \
            --entrypoint /opt/conda/envs/agentomics-env/bin/python \
            agentomics_img src/run_logging/evaluate_log_test.py

        mkdir -p outputs/${AGENT_ID}/best_run_files outputs/${AGENT_ID}/reports

        # Copy best-run files and report
        docker run --rm -u $(id -u):$(id -g) -v temp_agentomics_volume_${AGENT_ID}:/source -v $(pwd)/outputs/${AGENT_ID}:/dest busybox cp -r /source/snapshots/${AGENT_ID}/. /dest/best_run_files/

        docker run --rm -u $(id -u):$(id -g) -v temp_agentomics_volume_${AGENT_ID}:/source -v $(pwd)/outputs/${AGENT_ID}:/dest busybox cp -r /source/reports/${AGENT_ID}/. /dest/reports/
        
        echo -e "${GREEN}Run finished. Report and files can be found in outputs/${AGENT_ID}${NOCOLOR}"
        echo -e "${GREEN}To run inference on new data, use ./inference.sh --agent-dir outputs/${AGENT_ID} --input <path_to_input_csv> --output <path_to_output_csv>${NOCOLOR}"

    fi

    docker volume rm temp_agentomics_volume_${AGENT_ID}
fi
