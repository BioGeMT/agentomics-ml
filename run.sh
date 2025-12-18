#!/usr/bin/env bash

source "./bash_helpers.sh"

AGENTOMICS_ARGS=()
LOCAL_MODE=false
TEST_MODE=false
CPU_ONLY=false
OLLAMA=false
USE_PROVISIONING_KEY=false
SPEND_LIMIT=10
TIMEOUT_SECS=""
MODEL_NAME=""
DATASET_NAME=""
VAL_METRIC=""
LIST_MODE=false
FOUNDATION_MODEL_TYPE=""
PULL_IMAGES=false
DOCKERHUB_USERNAME="biogemt"

show_help() {
    cat << EOF
Usage: ./run.sh [OPTIONS]

Orchestrates the Agentomics training and evaluation process. By default, it runs in Docker containers.
Use --local to run with a local Conda environment.

Required Arguments (for non-interactive runs):
  --model <name>      The LLM model name (e.g., 'openai/gpt-4').
  --dataset <name>    The short identifier for the prepared dataset (e.g., 'breast_cancer').
  --iterations <N>    Number of iterations to run the agent (recommended more than 5).
  --timeout <int>   Amount of seconds the agent is allowed to run for. This or --iterations will dictate the duration, whichever will expire first. (recommended
  ~480s)
  --split-allowed-iterations <N>    Number of initial iterations that are allowed to (re)split the data into train/validation (e.g., 1).
  --exploration-iterations <N>     Number of initial iterations that should focus on baseline/exploration models (e.g., 4).
  --val-metric <name> The metric to optimize (e.g., 'ACC').
  --user-prompt <str> The main prompt/goal for the agent.
                      (Default: "Create the best possible machine learning model that will generalize to new unseen data.")

Operational Flags:
  --local             Run the project using local Conda environments instead of Docker.
  --test              Run the project's integrated test suite.
                      (Note: Only supported in Docker mode, not in local Conda mode.)
  --cpu-only          Force Docker/Conda to run using CPU only (skip GPU configuration).
  --ollama            Enable support for an Ollama server running on the host machine.
  --pull-images       Pull prebuilt Docker images from Docker Hub instead of building locally (uses biogemt images).
  --foundation-model-type <dna|rna|molecule|protein>
                      Enable foundation models of a specific type (Docker only). When omitted, no foundation models are used or pre-downloaded.
  --use-provisioning-key  Use OpenRouter provisioning key to create temporary API key and log costs.
  --spend-limit <N>   Only applies when --use-provisioning-key is passed. Spend limit for a temporary key (default: 10).
  --tags              (Optional) Space separated tags for Weights and Biases logging.
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
            LIST_MODE=true
            shift
            ;;
        --list-datasets)
            AGENTOMICS_ARGS+=(--list-datasets)
            LIST_MODE=true
            shift
            ;;
        --list-metrics)
            AGENTOMICS_ARGS+=(--list-metrics)
            LIST_MODE=true
            shift
            ;;
        --root-privileges)
            AGENTOMICS_ARGS+=(--root-privileges)
            shift
            ;;
        --model)
            require_opt_value "$1" "${2:-}"
            AGENTOMICS_ARGS+=(--model "$2")
            MODEL_NAME="$2"
            shift 2
            ;;
        --dataset)
            require_opt_value "$1" "${2:-}"
            AGENTOMICS_ARGS+=(--dataset "$2")
            DATASET_NAME="$2"
            shift 2
            ;;
        --iterations)
            require_opt_value "$1" "${2:-}"
            AGENTOMICS_ARGS+=(--iterations "$2")
            shift 2
            ;;
        --timeout)
            require_opt_value "$1" "${2:-}"
            AGENTOMICS_ARGS+=(--timeout "$2")
            TIMEOUT_SECS="$2"
            shift 2
            ;;
        --split-allowed-iterations)
            require_opt_value "$1" "${2:-}"
            AGENTOMICS_ARGS+=(--split-allowed-iterations "$2")
            shift 2
            ;;
        --exploration-iterations)
            AGENTOMICS_ARGS+=(--exploration-iterations "$2")
            shift 2
            ;;
        --val-metric)
            require_opt_value "$1" "${2:-}"
            AGENTOMICS_ARGS+=(--val-metric "$2")
            VAL_METRIC="$2"
            shift 2
            ;;
        --user-prompt)
            require_opt_value "$1" "${2:-}"
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
            require_opt_value "$1" "${2:-}"
            SPEND_LIMIT="$2"
            shift 2
            ;;
        --pull-images)
            PULL_IMAGES=true
            shift
            ;;
        --foundation-model-type)
            require_opt_value "$1" "${2:-}"
            FOUNDATION_MODEL_TYPE="$2"
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

IS_INTERACTIVE_RUN=false
if [[ -t 0 && ${#AGENTOMICS_ARGS[@]} -eq 0 && "$LOCAL_MODE" = false && "$TEST_MODE" = false ]]; then
    IS_INTERACTIVE_RUN=true
fi

if [ "$LOCAL_MODE" = true ]; then
    need_cmd conda
    need_cmd python
    if [ "$TEST_MODE" = true ]; then
        die "--test is only supported in Docker mode (remove --test or remove --local)"
    fi
    if [[ "$LIST_MODE" = false ]] && ! has_tty; then
        if [[ -z "$MODEL_NAME" || -z "$DATASET_NAME" || -z "$VAL_METRIC" ]]; then
            die "Non-interactive runs require --model, --dataset, and --val-metric (or run in an interactive terminal)"
        fi
    fi

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
    if [[ -n "$TIMEOUT_SECS" ]]; then
        need_cmd timeout
        [[ "$TIMEOUT_SECS" =~ ^[0-9]+$ ]] || die "--timeout must be an integer number of seconds (got: $TIMEOUT_SECS)"
        set +e
        timeout "$TIMEOUT_SECS" python src/run_agent_interactive.py ${AGENTOMICS_ARGS+"${AGENTOMICS_ARGS[@]}"}
        exit_code=$?
        set -e
        if [[ "$exit_code" -eq 124 ]]; then
            echo "Timed out after $TIMEOUT_SECS seconds"
        elif [[ "$exit_code" -ne 0 ]]; then
            exit "$exit_code"
        fi
    else
        python src/run_agent_interactive.py ${AGENTOMICS_ARGS+"${AGENTOMICS_ARGS[@]}"}
    fi

    if [[ "$LIST_MODE" = true ]]; then
        exit 0
    fi

    export PYTHONPATH=./src
    python src/run_logging/evaluate_log_test.py --workspace-dir ../workspace

    mkdir -p outputs/${AGENT_ID}/best_run_files outputs/${AGENT_ID}/reports
    cp -r ../workspace/snapshots/${AGENT_ID}/. outputs/${AGENT_ID}/best_run_files/
    cp -r ../workspace/reports/${AGENT_ID}/. outputs/${AGENT_ID}/reports/
else
    need_cmd docker
    if ! docker info >/dev/null 2>&1; then
        die "Docker is not running or not accessible (start Docker and retry)"
    fi
    if [[ "$LIST_MODE" = false ]] && ! has_tty; then
        if [[ -z "$MODEL_NAME" || -z "$DATASET_NAME" || -z "$VAL_METRIC" ]]; then
            die "Non-interactive runs require --model, --dataset, and --val-metric (or run in an interactive terminal)"
        fi
    fi

    if [[ "$IS_INTERACTIVE_RUN" = true && "$PULL_IMAGES" = false ]]; then
        echo ""
        echo "Docker images"
        echo "Select how to obtain Docker images:"
        echo "  1) build locally"
        echo "  2) pull prebuilt (biogemt)"
        echo ""
        read -r -p "Enter choice [2]: " images_choice
        images_choice="${images_choice:-2}"
        case "$images_choice" in
            1) PULL_IMAGES=false;;
            2) PULL_IMAGES=true;;
            *) echo -e "${RED}Error: Invalid choice.${NOCOLOR}" >&2; exit 1;;
        esac
    fi

    if [ -n "$FOUNDATION_MODEL_TYPE" ] && [[ "$FOUNDATION_MODEL_TYPE" != "dna" && "$FOUNDATION_MODEL_TYPE" != "rna" && "$FOUNDATION_MODEL_TYPE" != "molecule" && "$FOUNDATION_MODEL_TYPE" != "protein" ]]; then
        echo -e "${RED}Error: Invalid --foundation-model-type '$FOUNDATION_MODEL_TYPE'. Allowed: dna, rna, molecule, protein.${NOCOLOR}" >&2
        exit 1
    fi

    if [[ "$IS_INTERACTIVE_RUN" = true && -z "$FOUNDATION_MODEL_TYPE" ]]; then
        echo ""
        echo "Foundation models (optional)"
        echo "Select which foundation model type should be pre-downloaded and made available to the agent:"
        echo "  1) none"
        echo "  2) DNA"
        echo "  3) RNA"
        echo "  4) Molecule"
        echo "  5) Protein"
        echo ""
        read -r -p "Enter choice [1]: " fm_choice
        fm_choice="${fm_choice:-1}"
        case "$fm_choice" in
            1) FOUNDATION_MODEL_TYPE="";;
            2) FOUNDATION_MODEL_TYPE="dna";;
            3) FOUNDATION_MODEL_TYPE="rna";;
            4) FOUNDATION_MODEL_TYPE="molecule";;
            5) FOUNDATION_MODEL_TYPE="protein";;
            *) echo -e "${RED}Error: Invalid choice.${NOCOLOR}" >&2; exit 1;;
        esac
    fi

    FOUNDATION_MODEL_FLAGS=()
    DOCKER_BUILD_ARGS=()
    if [ -n "$FOUNDATION_MODEL_TYPE" ]; then
        DOCKER_BUILD_ARGS+=(--build-arg "FOUNDATION_MODEL_TYPE=$FOUNDATION_MODEL_TYPE")
        FOUNDATION_MODEL_FLAGS+=(-e "FOUNDATION_MODEL_TYPE=$FOUNDATION_MODEL_TYPE")
    fi

    AGENTOMICS_IMAGE="agentomics_img"
    PREPARE_IMAGE="agentomics_prepare_img"

    if [ "$PULL_IMAGES" = true ]; then
        FM_TAG="NONE"
        if [ -n "$FOUNDATION_MODEL_TYPE" ]; then
            FM_TAG="$(echo "$FOUNDATION_MODEL_TYPE" | tr '[:lower:]' '[:upper:]')"
        fi
        AGENTOMICS_IMAGE="${DOCKERHUB_USERNAME}/agentomics:FM-${FM_TAG}-latest"
        PREPARE_IMAGE="${DOCKERHUB_USERNAME}/agentomics-prepare:latest"

        echo "Pulling the run image"
        docker pull "$AGENTOMICS_IMAGE"
        echo "Pulling the data preparation image"
        docker pull "$PREPARE_IMAGE"
    else
        echo "Building the run image"
        docker build -t "$AGENTOMICS_IMAGE" -f Dockerfile ${DOCKER_BUILD_ARGS[@]+"${DOCKER_BUILD_ARGS[@]}"} .
        echo "Build done"

        echo "Building the data preparation image"
        docker build -t "$PREPARE_IMAGE" -f Dockerfile.prepare .
        echo "Build done"
    fi
    AGENT_ID=$(docker run --rm -u $(id -u):$(id -g) -v "$(pwd)":/repository:ro --entrypoint \
               /opt/conda/envs/agentomics-env/bin/python "$AGENTOMICS_IMAGE" /repository/src/utils/create_user.py)
    docker run \
        -u $(id -u):$(id -g) \
        --rm \
        -it \
        -e PYTHONWARNINGS=ignore \
        --name agentomics_prepare_cont_${AGENT_ID} \
        -v "$(pwd)":/repository \
        "$PREPARE_IMAGE"

    docker volume create temp_agentomics_volume_${AGENT_ID}
    cleanup() {
        docker volume rm temp_agentomics_volume_${AGENT_ID} >/dev/null 2>&1 || true
    }
    trap cleanup EXIT

    TEMP_API_KEY_HASH=""
    if [ "$USE_PROVISIONING_KEY" = true ]; then
        need_cmd conda
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

    if [ "$CPU_ONLY" = false ]; then
        if ! docker_has_gpu; then
            warn "GPU not available (nvidia-smi not found or Docker lacks GPU support)"
            warn "Automatically switching to CPU-only mode"
            warn "To suppress this warning, use --cpu-only flag"
            CPU_ONLY=true
        fi
    fi

    GPU_FLAGS=()
    if [ "$CPU_ONLY" = false ]; then
        GPU_FLAGS+=(--gpus all)
        GPU_FLAGS+=(--env NVIDIA_VISIBLE_DEVICES=all)
        info "GPU mode enabled"
    else
        info "Running in CPU-only mode"
    fi
    OLLAMA_FLAGS=()
    if [ "$OLLAMA" = true ]; then
        OLLAMA_FLAGS+=(--add-host=host.docker.internal:host-gateway)
    fi

    ENV_FILE_PATH="$(pwd)/.env"
    [[ -f "$ENV_FILE_PATH" ]] || die "Env file not found: $ENV_FILE_PATH (create it from .env.example)"
    ENV_FILE_ARGS=(--env-file "$ENV_FILE_PATH")

    PROVIDERS_CONFIG_FILE="src/utils/providers/configured_providers.yaml"
    [[ -f "$PROVIDERS_CONFIG_FILE" ]] || die "Missing providers config: $PROVIDERS_CONFIG_FILE"
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
            ${ENV_FILE_ARGS[@]+"${ENV_FILE_ARGS[@]}"} \
            -e AGENT_ID=${AGENT_ID} \
            -e PYTHONWARNINGS=ignore \
            ${FOUNDATION_MODEL_FLAGS[@]+"${FOUNDATION_MODEL_FLAGS[@]}"} \
            ${GPU_FLAGS[@]+"${GPU_FLAGS[@]}"} \
            ${OLLAMA_FLAGS[@]+"${OLLAMA_FLAGS[@]}"} \
            ${DOCKER_API_KEY_ENV_VARS[@]+"${DOCKER_API_KEY_ENV_VARS[@]}"} \
            -v "$(pwd)/src":/repository/src:ro \
            -v "$(pwd)/test":/repository/test:ro \
            -v "$(pwd)/prepared_datasets":/repository/prepared_datasets:ro \
            -v temp_agentomics_volume_${AGENT_ID}:/workspace \
            --entrypoint /opt/conda/envs/agentomics-env/bin/python \
            "$AGENTOMICS_IMAGE" -m test.run_all_tests
    else
        docker run \
            --rm \
            -it \
            --name agentomics_cont_${AGENT_ID} \
            ${ENV_FILE_ARGS[@]+"${ENV_FILE_ARGS[@]}"} \
            -e AGENT_ID=${AGENT_ID} \
            -e PYTHONWARNINGS=ignore \
            ${FOUNDATION_MODEL_FLAGS[@]+"${FOUNDATION_MODEL_FLAGS[@]}"} \
            ${GPU_FLAGS[@]+"${GPU_FLAGS[@]}"} \
            ${OLLAMA_FLAGS[@]+"${OLLAMA_FLAGS[@]}"} \
            ${DOCKER_API_KEY_ENV_VARS[@]+"${DOCKER_API_KEY_ENV_VARS[@]}"} \
            -v "$(pwd)/src":/repository/src:ro \
            -v "$(pwd)/prepared_datasets":/repository/prepared_datasets:ro \
            -v temp_agentomics_volume_${AGENT_ID}:/workspace \
            "$AGENTOMICS_IMAGE" ${AGENTOMICS_ARGS+"${AGENTOMICS_ARGS[@]}"}

        if [ "$LIST_MODE" = true ]; then
            exit 0
        fi
        ARTIFACT_PATH="/workspace/snapshots/${AGENT_ID}"

        if ! docker run --rm -v temp_agentomics_volume_${AGENT_ID}:/workspace busybox test -d ${ARTIFACT_PATH}; then
            echo -e "${RED}Agent didn't produce any valid model, skipping testing evaluation.${NOCOLOR}" >&2

            docker volume rm temp_agentomics_volume_${AGENT_ID} || true
            exit 1
        fi

        echo "Running final evaluation on test set"
        docker run \
            --rm \
            --name agentomics_test_eval_cont_${AGENT_ID} \
            ${ENV_FILE_ARGS[@]+"${ENV_FILE_ARGS[@]}"} \
            -e AGENT_ID=${AGENT_ID} \
            -e PYTHONPATH=/repository/src \
            -e PYTHONWARNINGS=ignore \
            ${FOUNDATION_MODEL_FLAGS[@]+"${FOUNDATION_MODEL_FLAGS[@]}"} \
            ${GPU_FLAGS[@]+"${GPU_FLAGS[@]}"} \
            -v "$(pwd)/src":/repository/src:ro \
            -v "$(pwd)/prepared_datasets":/repository/prepared_datasets:ro \
            -v "$(pwd)/prepared_test_sets":/repository/prepared_test_sets:ro \
            -v temp_agentomics_volume_${AGENT_ID}:/workspace \
            --entrypoint /opt/conda/envs/agentomics-env/bin/python \
            "$AGENTOMICS_IMAGE" src/run_logging/evaluate_log_test.py

        mkdir -p outputs/${AGENT_ID}/best_run_files outputs/${AGENT_ID}/reports outputs/${AGENT_ID}/run_files outputs/${AGENT_ID}/extras

        docker run --rm -v temp_agentomics_volume_${AGENT_ID}:/workspace busybox chmod -R a+rX /workspace/snapshots/${AGENT_ID}/ /workspace/runs/${AGENT_ID}/

        # Copy run files and report
        docker run --rm -u $(id -u):$(id -g) -v temp_agentomics_volume_${AGENT_ID}:/source -v $(pwd)/outputs/${AGENT_ID}:/dest busybox cp -r /source/snapshots/${AGENT_ID}/. /dest/best_run_files/

        docker run --rm -u $(id -u):$(id -g) -v temp_agentomics_volume_${AGENT_ID}:/source -v $(pwd)/outputs/${AGENT_ID}:/dest busybox cp -r /source/runs/${AGENT_ID}/. /dest/run_files/

        docker run --rm -u $(id -u):$(id -g) -v temp_agentomics_volume_${AGENT_ID}:/source -v $(pwd)/outputs/${AGENT_ID}:/dest busybox cp -r /source/reports/${AGENT_ID}/. /dest/reports/

        docker run --rm -u $(id -u):$(id -g) -v temp_agentomics_volume_${AGENT_ID}:/source -v $(pwd)/outputs/${AGENT_ID}:/dest busybox cp -r /source/extras/. /dest/extras/

        if [ "$USE_PROVISIONING_KEY" = true ]; then
            echo "Logging costs and cleaning up temporary API key"
            CONFIG_PATH="outputs/${AGENT_ID}/best_run_files/config.json"
            PYTHONPATH="$(pwd)/src" conda run -n agentomics-env python src/utils/api_keys_utils.py cleanup-and-log --config-path "$CONFIG_PATH" --api-key-hash "$TEMP_API_KEY_HASH"
        fi

        echo -e "${GREEN}Run finished. Report and files can be found in outputs/${AGENT_ID}${NOCOLOR}"
        echo -e "${GREEN}To run inference on new data, use ./inference.sh --agent-dir outputs/${AGENT_ID} --input <path_to_input_csv> --output <path_to_output_csv>${NOCOLOR}"

    fi
fi
