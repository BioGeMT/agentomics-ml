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
CPU_ONLY=true
OLLAMA=false
USE_PROVISIONING_KEY=false
SPEND_LIMIT=10
FORK_FROM_RUN=""
FORK_FROM_STEP=""
FORK_FROM_ITERATION=""

show_help() {
    cat << EOF
Usage: ./run.sh [OPTIONS]

Orchestrates the Agentomics training and evaluation process. By default, it runs in Docker containers.
Use --local to run with a local Conda environment.

Required Arguments (for non-fork runs):
  --model <name>      The LLM model name (e.g., 'openai/gpt-4').
  --dataset <name>    The dataset identifier (e.g., 'breast_cancer').
                      Auto-detected when forking.
  --iterations <N>    Number of iterations to run the agent (e.g., 5).
  --timeout <int>   Amount of seconds the agent is allowed to run for. This or --iterations will dictate the duration, whichever will expire first. 
  --split-allowed-iterations <N>    Number of initial iterations that are allowed to (re)split the data into train/validation (e.g., 1).
  --val-metric <name> The metric to optimize (e.g., 'ACC').
                      Auto-detected when forking.
  --user-prompt <str> The main prompt/goal for the agent.
                      Auto-inherited from source when forking.
                      (Default: "Create the best possible machine learning model that will generalize to new unseen data.")

Operational Flags:
  --local             Run the project using local Conda environments instead of Docker.
  --test              Run the project's integrated test suite.
                      (Note: Only supported in Docker mode, not in local Conda mode.)
  --cpu-only          Force Docker/Conda to run using CPU only (skip GPU configuration).
  --ollama            Enable support for an Ollama server running on the host machine.
  --use-provisioning-key  Use OpenRouter provisioning key to create temporary API key and log costs.
  --spend-limit <N>   Only applies when --use-provisioning-key is passed. Spend limit for a temporary key (default: 10).
  --tags              (Optional) Space separated tags for Weights and Biases logging.
  -h, --help          Show this help message and exit.

Forking Flags (Resume from a previous run):
  --fork-from-run <run_id>      The run ID to fork from (e.g., 'melodic_recipe_grind').
  --fork-from-step <step_name>  The step to fork from. Options: 
                                  data_exploration, data_split, data_representation,
                                  model_architecture, model_training, model_inference,
                                  prediction_exploration
  --fork-from-iteration <N>     (Optional) Specific iteration to fork from. If omitted, 
                                uses the latest iteration.

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
        --timeout)
            AGENTOMICS_ARGS+=(--timeout "$2")
            shift 2
            ;;
        --split-allowed-iterations)
            AGENTOMICS_ARGS+=(--split-allowed-iterations "$2")
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
        --tags)
            AGENTOMICS_ARGS+=(--tags)
            shift
            while [[ $# -gt 0 && "$1" != -* ]]; do
                AGENTOMICS_ARGS+=("$1")
                shift
            done
            ;;
        --local)
            LOCAL_MODE=true
            shift
            ;;
        --ollama)
            OLLAMA=true
            shift
            ;;
        --use-provisioning-key)
            USE_PROVISIONING_KEY=true
            shift
            ;;
        --spend-limit)
            SPEND_LIMIT="$2"
            shift 2
            ;;
        --test)
            TEST_MODE=true
            shift
            ;;
        --cpu-only)
            CPU_ONLY=true
            shift
            ;;
        --fork-from-run)
            FORK_FROM_RUN="$2"
            AGENTOMICS_ARGS+=(--fork-from-run "$2")
            shift 2
            ;;
        --fork-from-step)
            FORK_FROM_STEP="$2"
            AGENTOMICS_ARGS+=(--fork-from-step "$2")
            shift 2
            ;;
        --fork-from-iteration)
            FORK_FROM_ITERATION="$2"
            AGENTOMICS_ARGS+=(--fork-from-iteration "$2")
            shift 2
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

# Validate fork arguments
if [ -n "$FORK_FROM_RUN" ] || [ -n "$FORK_FROM_STEP" ]; then
    if [ -z "$FORK_FROM_RUN" ] || [ -z "$FORK_FROM_STEP" ]; then
        echo -e "${RED}Error: Both --fork-from-run and --fork-from-step must be specified together.${NOCOLOR}" >&2
        exit 1
    fi
    
    # Validate step name against predefined list
    VALID_STEPS=("data_exploration" "data_split" "data_representation" "model_architecture" "model_training" "model_inference" "prediction_exploration")
    if [[ ! " ${VALID_STEPS[@]} " =~ " ${FORK_FROM_STEP} " ]]; then
        echo -e "${RED}Error: Invalid step name '${FORK_FROM_STEP}'.${NOCOLOR}" >&2
        echo "Valid steps are: ${VALID_STEPS[*]}" >&2
        exit 1
    fi
    
    if [ -z "$FORK_FROM_ITERATION" ]; then
        echo "Fork mode enabled: forking from run '$FORK_FROM_RUN' at step '$FORK_FROM_STEP' (latest iteration)"
    else
        echo "Fork mode enabled: forking from run '$FORK_FROM_RUN' at step '$FORK_FROM_STEP' (iteration $FORK_FROM_ITERATION)"
    fi
fi

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

    # Generate AGENT_ID
    if [ -n "$FORK_FROM_RUN" ]; then
        # Generate fork-based ID: source_run_fork_step_[iterN]_timestamp
        # Using _fork_ instead of _forked_from_ to save space (8 chars per fork)
        TIMESTAMP=$(date +%Y%m%d_%H%M%S)
        if [ -n "$FORK_FROM_ITERATION" ]; then
            AGENT_ID="${FORK_FROM_RUN}_fork_${FORK_FROM_STEP}_iter${FORK_FROM_ITERATION}_${TIMESTAMP}"
        else
            AGENT_ID="${FORK_FROM_RUN}_fork_${FORK_FROM_STEP}_${TIMESTAMP}"
        fi
        # Safety check: truncate if exceeds 240 chars (leaving room for path prefixes)
        if [ ${#AGENT_ID} -gt 240 ]; then
            AGENT_ID="${AGENT_ID:0:240}"
            echo "[WARNING] Fork ID truncated to 240 chars to avoid filesystem limits"
        fi
        echo "Generated fork agent ID: $AGENT_ID"
    else
        AGENT_ID=$(python src/utils/create_user.py)
        echo "Generated new agent ID: $AGENT_ID"
    fi
    export AGENT_ID
    
    # If forking, copy source run data into the workspace
    if [ -n "$FORK_FROM_RUN" ]; then
        echo "Copying source run data for forking..."
        
        # Check if source run exists in outputs
        if [ ! -d "outputs/${FORK_FROM_RUN}" ]; then
            echo -e "${RED}Error: Source run '${FORK_FROM_RUN}' not found in outputs/${NOCOLOR}"
            echo "Make sure the source run exists before trying to fork from it."
            exit 1
        fi
        
        # Check for required files
        if [ ! -d "outputs/${FORK_FROM_RUN}/storage/.agentomics_storage" ]; then
            echo -e "${RED}Error: Content storage not found for run '${FORK_FROM_RUN}'${NOCOLOR}"
            echo "This run needs to be re-run with the updated system to support forking."
            exit 1
        fi
        
        # Copy .agentomics_storage (contains everything: snapshots, objects, configs)
        if [ -d "outputs/${FORK_FROM_RUN}/storage/.agentomics_storage" ]; then
            cp -r outputs/${FORK_FROM_RUN}/storage/.agentomics_storage ../workspace/
            echo "  Copied content-addressed storage (including config)"
        fi
    fi
    
    timeout $TIME_LIMIT_SECS python src/run_agent_interactive.py ${AGENTOMICS_ARGS+"${AGENTOMICS_ARGS[@]}"}
    if [ $? -eq 124 ]; then
        echo "Timed out after $TIME_LIMIT_SECS"
    fi
    
    # Function to copy local outputs (runs even on failure)
    copy_local_outputs() {
        local exit_code=$1
        
        if [ $exit_code -eq 0 ]; then
            # SUCCESS: Copy everything
            echo "Copying outputs from local workspace..."
            
            mkdir -p outputs/${AGENT_ID}/best_run_files outputs/${AGENT_ID}/reports outputs/${AGENT_ID}/storage
            
            # Copy content-addressed storage (now created with correct permissions by Python)
            echo "Copying content-addressed storage..."
            if [ -d "../workspace/.agentomics_storage" ]; then
                cp -r ../workspace/.agentomics_storage outputs/${AGENT_ID}/storage/
            fi
            
            # Copy all output files
            if [ -d "../workspace/snapshots/${AGENT_ID}" ]; then
                cp -r ../workspace/snapshots/${AGENT_ID}/. outputs/${AGENT_ID}/best_run_files/
            fi
            
            if [ -d "../workspace/reports/${AGENT_ID}" ]; then
                cp -r ../workspace/reports/${AGENT_ID}/. outputs/${AGENT_ID}/reports/
            fi
            
            echo -e "${GREEN}Run finished successfully. Outputs can be found in outputs/${AGENT_ID}${NOCOLOR}"
        else
            # FAILURE: Only copy .agentomics_storage (sufficient for forking)
            echo "Run failed. Copying step snapshots for recovery..."
            
            mkdir -p outputs/${AGENT_ID}/storage
            
            if [ -d "../workspace/.agentomics_storage" ]; then
                # Copy .agentomics_storage (now created with correct permissions by Python)
                cp -r ../workspace/.agentomics_storage outputs/${AGENT_ID}/storage/
                echo "Snapshots saved to outputs/${AGENT_ID}/storage/"
            else
                echo "Warning: No snapshots found"
            fi
            
            echo -e "${RED}Run failed with exit code ${exit_code}${NOCOLOR}"
            echo -e "Step snapshots saved to outputs/${AGENT_ID}/storage/"
            echo -e "You can fork from any completed step using: --fork-from-run ${AGENT_ID} --fork-from-step <step_name>"
        fi
    }
    
    # Set up trap to ensure outputs are copied even on failure
    trap 'copy_local_outputs $?' EXIT
    
    export PYTHONPATH=./src
    python src/run_logging/evaluate_log_test.py --workspace-dir ../workspace
    
    # If we get here, run succeeded
    trap - EXIT  # Disable trap
    copy_local_outputs 0
else
    echo "Building the run image"
    docker build -t agentomics_img -f Dockerfile .
    echo "Build done"
    
    # Generate AGENT_ID
    if [ -n "$FORK_FROM_RUN" ]; then
        # Generate fork-based ID: source_run_fork_step_[iterN]_timestamp
        # Using _fork_ instead of _forked_from_ to save space (8 chars per fork)
        TIMESTAMP=$(date +%Y%m%d_%H%M%S)
        if [ -n "$FORK_FROM_ITERATION" ]; then
            AGENT_ID="${FORK_FROM_RUN}_fork_${FORK_FROM_STEP}_iter${FORK_FROM_ITERATION}_${TIMESTAMP}"
        else
            AGENT_ID="${FORK_FROM_RUN}_fork_${FORK_FROM_STEP}_${TIMESTAMP}"
        fi
        # Safety check: truncate if exceeds 240 chars (leaving room for path prefixes)
        if [ ${#AGENT_ID} -gt 240 ]; then
            AGENT_ID="${AGENT_ID:0:240}"
            echo "[WARNING] Fork ID truncated to 240 chars to avoid filesystem limits"
        fi
        echo "Generated fork agent ID: $AGENT_ID"
    else
        AGENT_ID=$(docker run --rm -u $(id -u):$(id -g) -v "$(pwd)":/repository:ro --entrypoint \
                   /opt/conda/envs/agentomics-env/bin/python agentomics_img /repository/src/utils/create_user.py)
        echo "Generated new agent ID: $AGENT_ID"
    fi

    echo "Building the data preparation image"
    docker build -t agentomics_prepare_img -f Dockerfile.prepare .
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

    TEMP_API_KEY_HASH=""
    if [ "$USE_PROVISIONING_KEY" = true ]; then
        if ! conda env list | grep -q "^agentomics-env "; then
            echo "Creating agentomics-env conda environment"
            conda env create -f environment.yaml -q
        fi
        echo "Creating temporary API key with spend limit: $SPEND_LIMIT"
        API_KEY_OUTPUT=$(PYTHONPATH="$(pwd)/src" conda run -n agentomics-env python src/utils/api_keys_utils.py create --name "agentomics_run_$(date +%s)" --limit "$SPEND_LIMIT")
        TEMP_API_KEY=$(echo "$API_KEY_OUTPUT" | cut -d',' -f1)
        TEMP_API_KEY_HASH=$(echo "$API_KEY_OUTPUT" | cut -d',' -f2)
        export OPENROUTER_API_KEY="$TEMP_API_KEY"
    fi

    GPU_FLAGS=()
    if [ "$CPU_ONLY" = false ]; then
        GPU_FLAGS+=(--gpus all)
        GPU_FLAGS+=(--env NVIDIA_VISIBLE_DEVICES=all)
    fi

    # If forking, copy source run data into the volume
    if [ -n "$FORK_FROM_RUN" ]; then
        echo "Copying source run data for forking..."
        
        # Check if source run exists in outputs
        if [ ! -d "outputs/${FORK_FROM_RUN}" ]; then
            echo -e "${RED}Error: Source run '${FORK_FROM_RUN}' not found in outputs/${NOCOLOR}"
            echo "Make sure the source run exists before trying to fork from it."
            exit 1
        fi
        
        # Copy the content-addressed storage (includes objects, snapshots, and configs)
        if [ ! -d "outputs/${FORK_FROM_RUN}/storage/.agentomics_storage" ]; then
            echo -e "${RED}Error: Content storage not found for run '${FORK_FROM_RUN}'${NOCOLOR}"
            echo "This run needs to be re-run with the updated system to support forking."
            exit 1
        fi
        
        # Copy .agentomics_storage (contains everything: snapshots, objects, configs)
        docker run --rm \
            -v "$(pwd)/outputs/${FORK_FROM_RUN}/storage":/source:ro \
            -v temp_agentomics_volume_${AGENT_ID}:/dest \
            busybox cp -r /source/.agentomics_storage /dest/
        echo "  Copied content-addressed storage (including config)"
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

    if [ "$USE_PROVISIONING_KEY" = true ]; then
        DOCKER_API_KEY_ENV_VARS+=(-e "OPENROUTER_API_KEY=${OPENROUTER_API_KEY}")
    fi

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
            -v "$(pwd)/prepared_test_sets":/repository/prepared_test_sets:ro \
            -v temp_agentomics_volume_${AGENT_ID}:/workspace \
            --entrypoint /opt/conda/envs/agentomics-env/bin/python \
            agentomics_img -m test.run_all_tests
    else
        # Function to copy files from volume (runs even on failure)
        copy_outputs_from_volume() {
            local exit_code=$1
            
            if [ $exit_code -eq 0 ]; then
                # SUCCESS: Copy everything
                echo "Copying outputs from Docker volume..."
                
                mkdir -p outputs/${AGENT_ID}/best_run_files outputs/${AGENT_ID}/reports outputs/${AGENT_ID}/run_files outputs/${AGENT_ID}/extras outputs/${AGENT_ID}/storage

                # Copy content-addressed storage
                echo "Copying content-addressed storage..."
                docker run --rm -u $(id -u):$(id -g) \
                    -v temp_agentomics_volume_${AGENT_ID}:/source \
                    -v $(pwd)/outputs/${AGENT_ID}:/dest \
                    busybox sh -c "if [ -d /source/.agentomics_storage ]; then cp -r /source/.agentomics_storage /dest/storage/; fi"

                # Copy all output files (with existence checks)
                docker run --rm -v temp_agentomics_volume_${AGENT_ID}:/workspace busybox sh -c "
                    [ -d /workspace/snapshots/${AGENT_ID} ] && chmod -R a+rX /workspace/snapshots/${AGENT_ID}/ || true
                    [ -d /workspace/runs/${AGENT_ID} ] && chmod -R a+rX /workspace/runs/${AGENT_ID}/ || true
                "

                docker run --rm -u $(id -u):$(id -g) -v temp_agentomics_volume_${AGENT_ID}:/source -v $(pwd)/outputs/${AGENT_ID}:/dest busybox sh -c "
                    [ -d /source/snapshots/${AGENT_ID} ] && cp -r /source/snapshots/${AGENT_ID}/. /dest/best_run_files/ || echo 'No snapshots directory'
                "

                docker run --rm -u $(id -u):$(id -g) -v temp_agentomics_volume_${AGENT_ID}:/source -v $(pwd)/outputs/${AGENT_ID}:/dest busybox sh -c "
                    [ -d /source/runs/${AGENT_ID} ] && cp -r /source/runs/${AGENT_ID}/. /dest/run_files/ || echo 'No runs directory'
                "

                docker run --rm -u $(id -u):$(id -g) -v temp_agentomics_volume_${AGENT_ID}:/source -v $(pwd)/outputs/${AGENT_ID}:/dest busybox sh -c "
                    [ -d /source/reports/${AGENT_ID} ] && cp -r /source/reports/${AGENT_ID}/. /dest/reports/ || echo 'No reports directory'
                "

                docker run --rm -u $(id -u):$(id -g) -v temp_agentomics_volume_${AGENT_ID}:/source -v $(pwd)/outputs/${AGENT_ID}:/dest busybox sh -c "
                    [ -d /source/extras ] && cp -r /source/extras/. /dest/extras/ || echo 'No extras directory'
                "
                
                # If we used a provisioning key, log costs and clean up the temporary key
                if [ "$USE_PROVISIONING_KEY" = true ]; then
                    echo "Logging costs and cleaning up temporary API key"
                    CONFIG_PATH="outputs/${AGENT_ID}/best_run_files/config.json"
                    PYTHONPATH="$(pwd)/src" conda run -n agentomics-env python src/utils/api_keys_utils.py cleanup-and-log --config-path "$CONFIG_PATH" --api-key-hash "$TEMP_API_KEY_HASH"
                fi

                echo -e "${GREEN}Run finished successfully. Outputs can be found in outputs/${AGENT_ID}${NOCOLOR}"
                echo -e "${GREEN}To run inference on new data, use ./inference.sh --agent-dir outputs/${AGENT_ID} --input <path_to_input_csv> --output <path_to_output_csv>${NOCOLOR}"
            else
                # FAILURE: Only copy .agentomics_storage (sufficient for forking)
                echo "Run failed. Copying step snapshots for recovery..."
                
                mkdir -p outputs/${AGENT_ID}/storage

                docker run --rm -u $(id -u):$(id -g) \
                    -v temp_agentomics_volume_${AGENT_ID}:/source \
                    -v $(pwd)/outputs/${AGENT_ID}:/dest \
                    busybox sh -c "if [ -d /source/.agentomics_storage ]; then cp -r /source/.agentomics_storage /dest/storage/; echo 'Snapshots saved to outputs/${AGENT_ID}/storage/'; else echo 'Warning: No snapshots found'; fi"
                
                echo -e "${RED}Run failed with exit code ${exit_code}${NOCOLOR}"
                echo -e "Step snapshots saved to outputs/${AGENT_ID}/storage/"
                echo -e "You can fork from any completed step using: --fork-from-run ${AGENT_ID} --fork-from-step <step_name>"
            fi
            
            # Clean up volume
            docker volume rm temp_agentomics_volume_${AGENT_ID} 2>/dev/null || true
        }

        # Set up trap to ensure outputs are copied even on failure
        trap 'copy_outputs_from_volume $?' EXIT

        # Run the main agent
        docker run \
            --rm \
            -it \
            --name agentomics_cont_${AGENT_ID} \
            --env-file $(pwd)/.env \
            -e AGENT_ID=${AGENT_ID} \
            ${GPU_FLAGS[@]+"${GPU_FLAGS[@]}"} \
            ${OLLAMA_FLAGS[@]+"${OLLAMA_FLAGS[@]}"} \
            ${DOCKER_API_KEY_ENV_VARS[@]+"${DOCKER_API_KEY_ENV_VARS[@]}"} \
            -v "$(pwd)/src":/repository/src:ro \
            -v "$(pwd)/prepared_datasets":/repository/prepared_datasets:ro \
            -v "$(pwd)/prepared_test_sets":/repository/prepared_test_sets:ro \
            -v temp_agentomics_volume_${AGENT_ID}:/workspace \
            agentomics_img ${AGENTOMICS_ARGS+"${AGENTOMICS_ARGS[@]}"}

        echo "Running final evaluation on test set"
        docker run \
            --rm \
            --name agentomics_test_eval_cont_${AGENT_ID} \
            --env-file $(pwd)/.env \
            -e AGENT_ID=${AGENT_ID} \
            -e PYTHONPATH=/repository/src \
            ${GPU_FLAGS[@]+"${GPU_FLAGS[@]}"} \
            -v "$(pwd)/src":/repository/src:ro \
            -v "$(pwd)/prepared_datasets":/repository/prepared_datasets:ro \
            -v "$(pwd)/prepared_test_sets":/repository/prepared_test_sets:ro \
            -v temp_agentomics_volume_${AGENT_ID}:/workspace \
            --entrypoint /opt/conda/envs/agentomics-env/bin/python \
            agentomics_img src/run_logging/evaluate_log_test.py
        
        # If we get here, both runs succeeded
        trap - EXIT  # Disable trap
        copy_outputs_from_volume 0

    fi
fi
