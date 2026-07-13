#!/usr/bin/env bash

# Get the absolute directory of this script
AGENTOMICS_DIR="$(cd "$(dirname "$0")" && pwd)"

source "$AGENTOMICS_DIR/scripts/bash_helpers.sh"

cd "$AGENTOMICS_DIR" || die "Cannot cd into repository directory: $AGENTOMICS_DIR"

AGENTOMICS_ARGS=()
LOCAL_MODE=false
TEST_MODE=false
CPU_ONLY=false
OLLAMA=false
USE_PROVISIONING_KEY=false
SPEND_LIMIT=10
TIMEOUT_SECS=""
MODEL_NAME=""
PREFERRED_PROVIDER=""
DATASET_NAME=""
TASK_TYPE=""
VAL_METRIC=""
LIST_MODE=false
DOCTOR_MODE=false
DOCTOR_JSON=false
FOUNDATION_MODELS_TYPE=""
VERBOSITY="full"
ALL_ITERATIONS_TEST=false
BUILD_IMAGES=false
DOCKERHUB_USERNAME="biogemt"
FORK_FROM_RUN=""
FORK_FROM_STEP=""
FORK_FROM_ITERATION=""

show_help() {
    cat <<'EOF'
Usage: ./run.sh [OPTIONS]

Orchestrates the Agentomics training and evaluation process. By default, it runs in Docker containers.
Use --local to run with a local Conda environment.

Required Arguments (for non-interactive runs):
  --model <name>      The LLM model name (e.g., 'openai/gpt-4').
  --dataset <name>    The short identifier for the dataset (e.g., 'breast_cancer').
                      Can be replaced by --fork-from-run, which inherits the dataset from the source run.

Optional Arguments:
  --iteration-plan-model <name>
                      The LLM model used for generating the iteration plan (e.g., 'openai/gpt-5.4').
                      If not provided, defaults to the same model as --model.
  --provider <name>   When multiple api keys are provided, provider override (e.g., 'openai', 'openrouter').
  --task-type <classification|regression>
                      Task type used when preparing a dataset selected with --dataset.
                      If omitted and the dataset metadata does not define it, preparation prompts in interactive mode.
  --iterations <N>    Number of iterations to run the agent (recommended more than 5).
                      For forked runs, omitting it keeps the source run's total iteration limit.
                      Providing it means N additional iterations from the fork point.
  --timeout <int>     Amount of seconds the agent is allowed to run for. This or --iterations will dictate the duration,
                      whichever expires first (recommended ~480s).
  --run-python-timeout <int>
                      Timeout in seconds for each run_python tool execution — this determines the maximum training time
                      (default: 21600, i.e. 6 hours).
  --split-allowed-iterations <N>
                      Number of initial iterations that are allowed to (re)split the data into train/validation (e.g., 1).
                      For forked runs, omitting it keeps the source run's split-allowed limit.
                      Providing it means N more split-allowed iterations from the fork point.
  --exploration-iterations <N>
                      Number of initial iterations that should focus on baseline/exploration models (e.g., 4).
                      For forked runs, omitting it keeps the source run's exploration limit.
                      Providing it means N more exploration iterations from the fork point.
  --val-metric <name> Metric to optimize. Defaults: AUROC (classification), MAE (regression).
  --user-prompt <str> The main prompt/goal for the agent.
                      (Default: "Develop a machine learning model that generalizes well to new unseen data.")

Forking:
  --fork-from-run <path>  Path to the source run workspace directory (the 'outputs/<run_id>' folder).
                          Creates a new independent run branching off from the given checkpoint.
                          When forking, most run arguments (--model, --iterations, --user-prompt, etc.)
                          are optional — omitting them reuses the values from the source run's config.
                          Two arguments are always inherited and cannot be overridden:
                            --dataset    (tied to the data the source run was trained on)
                            --val-metric (must stay consistent to compare iterations across the fork)
  --fork-from-step <step> Only used with --fork-from-run. Step ID to fork from (e.g. 'model_training').
                          Defaults to the latest completed step or iteration end checkpoint in the source run.
  --fork-from-iteration <N>
                          Only used with --fork-from-run. Iteration to fork from.
                          Defaults to the latest iteration containing the specified step or iteration end checkpoint.

Operational Flags:
  --local             Run the project using local Conda environments instead of Docker.
  --test              Run the project's integrated test suite.
                      (Note: Only supported in Docker mode, not in local Conda mode.)
  --all-iterations-test
                      After the run finishes, evaluate every archived iteration on the held-out test set.
  --cpu-only          Force Docker/Conda to run using CPU only (skip GPU configuration).
  --ollama            Enable support for an Ollama server running on the host machine.
  --build-images      Build Docker images locally instead of pulling prebuilt biogemt images from Docker Hub.
  --foundation-models-type <dna|rna|molecule|protein|all>
                      Enable foundation models of a specific type. Use 'all' to download all types.
                      When omitted on a fresh run, no foundation models are used.
                      When omitted on a forked run, the source run's foundation model type is reused.
  --use-provisioning-key  Use OpenRouter provisioning key to create temporary API key and log costs.
  --spend-limit <N>   Only applies when --use-provisioning-key is passed. Spend limit for a temporary key (default: 10).
  --disable-training-reporting
                      Disable the TrainingReporter helper that emits structured best-effort training updates (enabled by default).
  --conda-export-mode <full|yaml>
                      How to capture the best-iteration environment. 'full' (default) copies the conda env into the
                      snapshot and reuses it directly at test time (fast, large). 'yaml' stores only environment.yml
                      and rebuilds the env at test time (portable, slower).
  --verbosity <summary|full>
                      Control how much agent interaction detail is printed during the run (default: full).
  --tags              (Optional) Space separated tags for Weights and Biases logging.
  -h, --help          Show this help message and exit.

Diagnostic Flags:
  --doctor            Run pre-flight environment and provider checks without starting a run.
                      Validates Docker/Conda, provider credentials, disk space, and dataset presence.
                      Use with --local, --cpu-only, --ollama to check specific configurations.
                      Add --json for machine-readable output.

Listing Flags (Run the script with only one of these):
  --list-models       List models available via the configured provider and exit.
  --list-datasets     List available datasets and exit.
  --list-metrics      List all available validation metrics and exit.

Environment:
  API keys read from 'src/utils/providers/configured_providers.yaml' must be set as
  environment variables in your host environment (e.g., in a shell session or .env file)
  to be injected into the Docker container.
  For the 'codex' provider, run `codex login` on the host so `~/.codex/auth.json`
  is available to the launcher.

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
        --doctor)
            DOCTOR_MODE=true
            shift
            ;;
        --json)
            DOCTOR_JSON=true
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
        --iteration-plan-model)
            require_opt_value "$1" "${2:-}"
            AGENTOMICS_ARGS+=(--iteration-plan-model "$2")
            shift 2
            ;;
        --provider)
            require_opt_value "$1" "${2:-}"
            AGENTOMICS_ARGS+=(--provider "$2")
            PREFERRED_PROVIDER="$2"
            shift 2
            ;;
        --dataset)
            require_opt_value "$1" "${2:-}"
            AGENTOMICS_ARGS+=(--dataset "$2")
            DATASET_NAME="$2"
            shift 2
            ;;
        --task-type)
            require_opt_value "$1" "${2:-}"
            AGENTOMICS_ARGS+=(--task-type "$2")
            TASK_TYPE="$2"
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
        --run-python-timeout)
            require_opt_value "$1" "${2:-}"
            AGENTOMICS_ARGS+=(--run-python-timeout "$2")
            shift 2
            ;;
        --split-timeout)
            AGENTOMICS_ARGS+=(--split-timeout "$2")
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
        --verbosity)
            require_opt_value "$1" "${2:-}"
            VERBOSITY="$2"
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
        --fork-from-run)
            require_opt_value "$1" "${2:-}"
            FORK_FROM_RUN="$2"
            shift 2
            ;;
        --fork-from-step)
            require_opt_value "$1" "${2:-}"
            FORK_FROM_STEP="$2"
            shift 2
            ;;
        --fork-from-iteration)
            require_opt_value "$1" "${2:-}"
            FORK_FROM_ITERATION="$2"
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
        --use-provisioning-key)
            USE_PROVISIONING_KEY=true
            shift
            ;;
        --spend-limit)
            require_opt_value "$1" "${2:-}"
            SPEND_LIMIT="$2"
            shift 2
            ;;
        --build-images)
            BUILD_IMAGES=true
            shift
            ;;
        --foundation-models-type)
            require_opt_value "$1" "${2:-}"
            FOUNDATION_MODELS_TYPE="$2"
            shift 2
            ;;
        --test)
            TEST_MODE=true
            shift
            ;;
        --all-iterations-test)
            ALL_ITERATIONS_TEST=true
            shift
            ;;
        --disable-training-reporting)
            AGENTOMICS_ARGS+=(--disable-training-reporting)
            shift
            ;;
        --conda-export-mode)
            require_opt_value "$1" "${2:-}"
            AGENTOMICS_ARGS+=(--conda-export-mode "$2")
            shift 2
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

if ! is_valid_foundation_models_type "$FOUNDATION_MODELS_TYPE"; then
    die "Invalid --foundation-models-type '$FOUNDATION_MODELS_TYPE'. Allowed: dna, rna, molecule, protein, all."
fi
if [[ "$VERBOSITY" != "summary" && "$VERBOSITY" != "full" ]]; then
    die "Invalid --verbosity '$VERBOSITY'. Allowed: summary, full."
fi

EFFECTIVE_FOUNDATION_MODELS_TYPE="$(resolve_effective_string_field "$FOUNDATION_MODELS_TYPE" "$FORK_FROM_RUN" "foundation_models_type")"
EFFECTIVE_DATASET_NAME="$(resolve_effective_string_field "$DATASET_NAME" "$FORK_FROM_RUN" "dataset" true)"

if ! is_valid_foundation_models_type "$EFFECTIVE_FOUNDATION_MODELS_TYPE"; then
    die "Invalid foundation model type '$EFFECTIVE_FOUNDATION_MODELS_TYPE' in the effective run configuration. Allowed: dna, rna, molecule, protein, all."
fi

if [[ -n "$FORK_FROM_RUN" && -n "$DATASET_NAME" && "$DATASET_NAME" != "$EFFECTIVE_DATASET_NAME" ]]; then
    warn "--dataset '$DATASET_NAME' is ignored for forked runs. Using '$EFFECTIVE_DATASET_NAME' from the source run config."
fi

# Run doctor checks if requested and exit early
if [ "$DOCTOR_MODE" = true ]; then
    DEPLOYMENT_MODE="docker"
    if [ "$LOCAL_MODE" = true ]; then
        DEPLOYMENT_MODE="local"
    fi

    export PYTHONPATH="$(pwd)/src"

    DOCTOR_ARGS="--repo-root $(pwd) --deployment-mode $DEPLOYMENT_MODE"

    if [ "$CPU_ONLY" = true ]; then
        DOCTOR_ARGS="$DOCTOR_ARGS --cpu-only"
    fi

    if [ -n "$PREFERRED_PROVIDER" ]; then
        DOCTOR_ARGS="$DOCTOR_ARGS --provider-name $PREFERRED_PROVIDER"
    fi

    PROVIDER_ARG="None"
    if [ -n "$PREFERRED_PROVIDER" ]; then
        PROVIDER_ARG="$PREFERRED_PROVIDER"
    fi

    CPU_ARG="False"
    if [ "$CPU_ONLY" = true ]; then
        CPU_ARG="True"
    fi

    JSON_ARG="False"
    if [ "$DOCTOR_JSON" = true ]; then
        JSON_ARG="True"
    fi

    python3 - "$DEPLOYMENT_MODE" "$PROVIDER_ARG" "$CPU_ARG" "$JSON_ARG" << 'DOCTOR_SCRIPT'
import sys
from pathlib import Path
from utils.doctor import run_doctor

output, exit_code = run_doctor(
    repo_root=Path.cwd(),
    deployment_mode=sys.argv[1],
    provider_name=sys.argv[2] if sys.argv[2] != "None" else None,
    cpu_only=sys.argv[3] == "True",
    json_output=sys.argv[4] == "True"
)
print(output)
sys.exit(exit_code)
DOCTOR_SCRIPT

    exit $?
fi


if [ "$LOCAL_MODE" = true ]; then
    need_cmd conda
    need_cmd python
    if [ "$TEST_MODE" = true ]; then
        die "--test is only supported in Docker mode (remove --test or remove --local)"
    fi
    if [[ "$LIST_MODE" = false ]] && ! has_tty; then
        if [[ -z "$FORK_FROM_RUN" && ( -z "$MODEL_NAME" || -z "$DATASET_NAME" ) ]]; then
            die "Non-interactive runs require --model and --dataset (or --fork-from-run) (or run in an interactive terminal)"
        fi
    fi

    if ! conda env list | grep -q "agentomics-env"; then
        conda env create -f envs/environment.yaml -q
    else
        conda env update -n agentomics-env -f envs/environment.yaml -q
    fi

    eval "$(conda shell.bash hook)"
    conda activate agentomics-env

    AGENT_ID=$(python src/utils/agent_id.py)
    export AGENT_ID
    export AGENTOMICS_VERBOSITY="$VERBOSITY"

    WORKSPACE_ROOT="$(dirname "$AGENTOMICS_DIR")/workspace"
    WORKSPACE_CACHE_DIR="$WORKSPACE_ROOT/cache"
    WORKSPACE_DIR="$WORKSPACE_ROOT/runs/$AGENT_ID"
    WORKSPACE_TEST_DATASETS_DIR="$(pwd)/test_datasets"
    mkdir -p "$WORKSPACE_CACHE_DIR" "$(dirname "$WORKSPACE_DIR")"
    rm -rf "$WORKSPACE_DIR"
    mkdir -p "$WORKSPACE_DIR"

    if [ "$CPU_ONLY" = true ]; then
        export CUDA_VISIBLE_DEVICES=""
    fi

    if [ -n "$EFFECTIVE_FOUNDATION_MODELS_TYPE" ]; then
        FOUNDATION_MODELS_YAML_PATH="$(pwd)/foundation_models/models.yaml"
        [[ -f "$FOUNDATION_MODELS_YAML_PATH" ]] || die "Foundation models YAML not found: $FOUNDATION_MODELS_YAML_PATH"
        export HF_HOME="$WORKSPACE_CACHE_DIR/foundation_models"
        mkdir -p "$HF_HOME"
        AGENTOMICS_ARGS+=(--foundation-models-type "$EFFECTIVE_FOUNDATION_MODELS_TYPE")
        AGENTOMICS_ARGS+=(--foundation-models-yaml "$FOUNDATION_MODELS_YAML_PATH")
    fi

    echo -e "${RED}Running in local mode - this is only recommended if you run in a non-vulnerable environment!${NOCOLOR}"
    echo "For Docker mode (secure run), re-run without the --local flag."
    
    if ! conda env list | grep -q "^agent_start_env "; then
        conda env create -f envs/environment_agent.yaml -q
    fi
    START_ENV_PKG_PATH="$WORKSPACE_CACHE_DIR/agent_start_env.tar"
    if [[ ! -f "$START_ENV_PKG_PATH" ]]; then
        echo "Packing agent start environment to ${START_ENV_PKG_PATH}"
        conda run -n agent_start_env conda-pack --format tar -o "$START_ENV_PKG_PATH"
    fi
    export START_ENV_PKG="$START_ENV_PKG_PATH"

    if [ -n "$EFFECTIVE_FOUNDATION_MODELS_TYPE" ]; then
        FOUNDATION_MODELS_MARKER="$HF_HOME/.downloaded_${EFFECTIVE_FOUNDATION_MODELS_TYPE}"
        if [[ ! -f "$FOUNDATION_MODELS_MARKER" ]]; then
            conda run -n agentomics-env python src/utils/download_foundation_models.py \
                --foundation-models-type "$EFFECTIVE_FOUNDATION_MODELS_TYPE" \
                --models-yaml "$FOUNDATION_MODELS_YAML_PATH"
            touch "$FOUNDATION_MODELS_MARKER"
        fi
    fi

    TEMP_API_KEY_HASH=""
    if [ "$USE_PROVISIONING_KEY" = true ]; then
        echo "Creating temporary API key with spend limit: $SPEND_LIMIT"
        API_KEY_OUTPUT=$(PYTHONPATH="$(pwd)/src" conda run -n agentomics-env python src/utils/api_keys.py create --name "agentomics_run_$(date +%s)" --limit "$SPEND_LIMIT")
        TEMP_API_KEY=$(echo "$API_KEY_OUTPUT" | cut -d',' -f1)
        TEMP_API_KEY_HASH=$(echo "$API_KEY_OUTPUT" | cut -d',' -f2)
        export OPENROUTER_API_KEY="$TEMP_API_KEY"
    fi

    AGENTOMICS_ARGS+=(--workspace-dir "$WORKSPACE_DIR" --datasets-dir "$(pwd)/datasets")

    if [ -n "$FORK_FROM_RUN" ]; then
        FORK_FROM_RUN_ABS="$(cd "$FORK_FROM_RUN" && pwd)"
        build_setup_fork_args "$FORK_FROM_RUN_ABS" "$WORKSPACE_DIR" "$AGENT_ID" "$FORK_FROM_STEP" "$FORK_FROM_ITERATION"
        PYTHONPATH="$(pwd)/src" conda run -n agentomics-env python src/runtime/setup_fork.py "${SETUP_FORK_ARGS[@]}"
    fi

    RUN_EXIT_CODE=0
    if [[ -n "$TIMEOUT_SECS" ]]; then
        need_cmd timeout
        [[ "$TIMEOUT_SECS" =~ ^[0-9]+$ ]] || die "--timeout must be an integer number of seconds (got: $TIMEOUT_SECS)"
        set +e
        timeout "$TIMEOUT_SECS" python src/run_agent_interactive.py ${AGENTOMICS_ARGS+"${AGENTOMICS_ARGS[@]}"}
        RUN_EXIT_CODE=$?
        set -e
        if [[ "$RUN_EXIT_CODE" -eq 124 ]]; then
            echo "Timed out after $TIMEOUT_SECS seconds"
        elif [[ "$RUN_EXIT_CODE" -ne 0 ]]; then
            warn "Run process exited with code ${RUN_EXIT_CODE}. Exporting available run state before exiting."
        fi
    else
        set +e
        python src/run_agent_interactive.py ${AGENTOMICS_ARGS+"${AGENTOMICS_ARGS[@]}"}
        RUN_EXIT_CODE=$?
        set -e
        if [[ "$RUN_EXIT_CODE" -ne 0 ]]; then
            warn "Run process exited with code ${RUN_EXIT_CODE}. Exporting available run state before exiting."
        fi
    fi

    if [[ "$LIST_MODE" = true ]]; then
        exit 0
    fi

    RUN_SUCCEEDED=true
    if [[ -f "${WORKSPACE_DIR}/best_iteration_snapshot/runtime_info/iteration_metadata.json" ]]; then
        CONFIG_PATH="${WORKSPACE_DIR}/run/shared/config.json"
        if [[ ! -f "${CONFIG_PATH}" ]]; then
            die "Config not found: ${CONFIG_PATH}"
        fi
        export PYTHONPATH=./src
        conda run -n agentomics-env python src/run_logging/test_evaluation.py \
            --workspace-dir "$WORKSPACE_DIR" \
            --test-datasets-dir "$WORKSPACE_TEST_DATASETS_DIR"
    else
        RUN_SUCCEEDED=false
    fi

    mkdir -p "outputs/${AGENT_ID}"
    tar -c --exclude='./run/shared/.conda' -C "${WORKSPACE_DIR}" . | tar -x -C "outputs/${AGENT_ID}/"

    if [[ "$RUN_SUCCEEDED" = true ]]; then
        PYTHONPATH="$(pwd)/src" conda run -n agentomics-env python -m runtime.iteration_reports --agent-dir "outputs/${AGENT_ID}"

        if [ "$USE_PROVISIONING_KEY" = true ]; then
            echo "Logging costs and cleaning up temporary API key"
            PYTHONPATH="$(pwd)/src" conda run -n agentomics-env python src/utils/api_keys.py cleanup-and-log --config-path "outputs/${AGENT_ID}/run/shared/config.json" --api-key-hash "$TEMP_API_KEY_HASH"
        fi

        write_outputs_readme "${AGENT_ID}"
        PYTHONPATH="$(pwd)/src" conda run -n agentomics-env python src/runtime/generate_final_reports.py \
            --agent-dir "outputs/${AGENT_ID}" \
            --test-datasets-dir "$WORKSPACE_TEST_DATASETS_DIR"

        if [ "$ALL_ITERATIONS_TEST" = true ]; then
            echo "Running held-out test evaluation for all archived iterations"
            PYTHONPATH="$(pwd)/src" conda run -n agentomics-env \
                python -m runtime.stealth_test_evaluation \
                --agent-dir "outputs/${AGENT_ID}" \
                --test-datasets-dir "$WORKSPACE_TEST_DATASETS_DIR"
        fi

        echo "PDF reports ready at: outputs/${AGENT_ID}/reports/pdf/"
        echo -e "${GREEN}Run finished. Report and files can be found in outputs/${AGENT_ID}${NOCOLOR}"
        echo -e "${GREEN}To run inference on new data, use ./scripts/inference.sh --agent-dir outputs/${AGENT_ID} --input <path_to_input_folder> --output <path_to_output_csv>${NOCOLOR}"
    else
        PYTHONPATH="$(pwd)/src" conda run -n agentomics-env python -m runtime.iteration_reports --agent-dir "outputs/${AGENT_ID}"
        warn "Agent didn't produce any valid best iteration snapshot. Exported run artifacts to outputs/${AGENT_ID}."
        write_outputs_readme "${AGENT_ID}"
    fi

    if [[ "$RUN_EXIT_CODE" -ne 0 ]]; then
        exit "$RUN_EXIT_CODE"
    fi
    if [[ "$RUN_SUCCEEDED" = false ]]; then
        exit 1
    fi
else
    need_cmd docker
    if ! docker info >/dev/null 2>&1; then
        die "Docker is not running or not accessible (start Docker and retry). Alternatively, run with --local argument(./run.sh --local), if you are running in a non-vulnerable environment."
    fi
    if [[ "$LIST_MODE" = false ]] && ! has_tty; then
        if [[ -z "$FORK_FROM_RUN" && ( -z "$MODEL_NAME" || -z "$DATASET_NAME" ) ]]; then
            die "Non-interactive runs require --model and --dataset (or --fork-from-run) (or run in an interactive terminal)"
        fi
    fi


    DOCKER_BUILD_ARGS=()
    if [ -n "$EFFECTIVE_FOUNDATION_MODELS_TYPE" ]; then
        DOCKER_BUILD_ARGS+=(--build-arg "FOUNDATION_MODEL_TYPE=$EFFECTIVE_FOUNDATION_MODELS_TYPE")
        AGENTOMICS_ARGS+=(--foundation-models-type "$EFFECTIVE_FOUNDATION_MODELS_TYPE")
        AGENTOMICS_ARGS+=(--foundation-models-yaml /foundation_models/models.yaml)
    fi

    AGENTOMICS_IMAGE="agentomics_img"

    if [ "$BUILD_IMAGES" = true ]; then
        echo "Building the run image"
        docker build -t "$AGENTOMICS_IMAGE" -f Dockerfile ${DOCKER_BUILD_ARGS[@]+"${DOCKER_BUILD_ARGS[@]}"} .
        echo "Build done"
    else
        FM_TAG="NONE"
        if [ -n "$EFFECTIVE_FOUNDATION_MODELS_TYPE" ]; then
            FM_TAG="$(echo "$EFFECTIVE_FOUNDATION_MODELS_TYPE" | tr '[:lower:]' '[:upper:]')"
        fi
        AGENTOMICS_IMAGE="${DOCKERHUB_USERNAME}/agentomics:FM-${FM_TAG}-latest"

        echo "Pulling the run image"
        docker pull "$AGENTOMICS_IMAGE"
    fi
    AGENT_ID=$(docker run --rm -u $(id -u):$(id -g) -v "$(pwd)":/repository:ro --entrypoint \
               /opt/conda/envs/agentomics-env/bin/python "$AGENTOMICS_IMAGE" /repository/src/utils/agent_id.py)

    printless_command docker volume create temp_agentomics_volume_${AGENT_ID}
    cleanup_volume_on_finish

    TEMP_API_KEY_HASH=""
    if [ "$USE_PROVISIONING_KEY" = true ]; then
        need_cmd conda
        if ! conda env list | grep -q "^agentomics-env "; then
            echo "Creating agentomics-env conda environment"
            conda env create -f envs/environment.yaml -q
        fi
        echo "Creating temporary API key with spend limit: $SPEND_LIMIT"
        API_KEY_OUTPUT=$(PYTHONPATH="$(pwd)/src" conda run -n agentomics-env python src/utils/api_keys.py create --name "agentomics_run_$(date +%s)" --limit "$SPEND_LIMIT")
        TEMP_API_KEY=$(echo "$API_KEY_OUTPUT" | cut -d',' -f1)
        TEMP_API_KEY_HASH=$(echo "$API_KEY_OUTPUT" | cut -d',' -f2)
        export OPENROUTER_API_KEY="$TEMP_API_KEY"
    fi

    OLLAMA_FLAGS=()
    if [ "$OLLAMA" = true ]; then
        OLLAMA_FLAGS+=(--network="host")
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

    if [ "$LIST_MODE" = false ]; then
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
        fi
    fi

    CODEX_DOCKER_FLAGS=()
    HOST_CODEX_DIR="${HOME}/.codex"
    if [[ -d "$HOST_CODEX_DIR" ]] && ([[ -z "$PREFERRED_PROVIDER" ]] || [[ "$PREFERRED_PROVIDER" == "codex" ]]); then
        CODEX_DOCKER_FLAGS+=(-v "${HOST_CODEX_DIR}:/mnt/codex-host:ro")
        CODEX_DOCKER_FLAGS+=(-e "CODEX_AUTH_FILE=/tmp/codex/auth.json")
        CODEX_DOCKER_FLAGS+=(-e "CODEX_MODELS_CACHE_FILE=/tmp/codex/models_cache.json")
    fi

    AGENTOMICS_ARGS+=(--workspace-dir /workspace --datasets-dir /repository/datasets)
    TEST_EXEC=(--entrypoint /opt/conda/envs/agentomics-env/bin/python "$AGENTOMICS_IMAGE" -m test.run_all_tests)
    RUN_EXEC=("$AGENTOMICS_IMAGE" "${AGENTOMICS_ARGS[@]}")
    if [[ ${#CODEX_DOCKER_FLAGS[@]} -gt 0 ]]; then
        CODEX_BOOTSTRAP='
            mkdir -p /tmp/codex
            if [[ -f /mnt/codex-host/auth.json ]]; then cp -f /mnt/codex-host/auth.json /tmp/codex/auth.json; fi
            if [[ -f /mnt/codex-host/models_cache.json ]]; then cp -f /mnt/codex-host/models_cache.json /tmp/codex/models_cache.json; fi
            exec "$@"
        '
        TEST_EXEC=(
            --entrypoint /bin/bash
            "$AGENTOMICS_IMAGE"
            -lc "$CODEX_BOOTSTRAP"
            --
            /opt/conda/envs/agentomics-env/bin/python
            -m
            test.run_all_tests
        )
        RUN_EXEC=(
            --entrypoint /bin/bash
            "$AGENTOMICS_IMAGE"
            -lc "$CODEX_BOOTSTRAP"
            --
            /opt/conda/envs/agentomics-env/bin/python
            /repository/src/run_agent_interactive.py
            "${AGENTOMICS_ARGS[@]}"
        )
    fi

    if [ "$TEST_MODE" = true ]; then
        docker run \
            -it \
            --rm \
            --name agentomics_test_cont_${AGENT_ID} \
            ${ENV_FILE_ARGS[@]+"${ENV_FILE_ARGS[@]}"} \
            -e AGENT_ID=${AGENT_ID} \
            -e AGENTOMICS_VERBOSITY=${VERBOSITY} \
            -e PYTHONWARNINGS=ignore \
            ${EFFECTIVE_FOUNDATION_MODELS_TYPE:+-e FOUNDATION_MODELS_TYPE=${EFFECTIVE_FOUNDATION_MODELS_TYPE}} \
            ${GPU_FLAGS[@]+"${GPU_FLAGS[@]}"} \
            ${OLLAMA_FLAGS[@]+"${OLLAMA_FLAGS[@]}"} \
            ${DOCKER_API_KEY_ENV_VARS[@]+"${DOCKER_API_KEY_ENV_VARS[@]}"} \
            ${CODEX_DOCKER_FLAGS[@]+"${CODEX_DOCKER_FLAGS[@]}"} \
            -v "$(pwd)/src":/repository/src:ro \
            -v "$(pwd)/test":/repository/test:ro \
            -v "$(pwd)/scripts":/repository/scripts:ro \
            -v "$(pwd)/run.sh":/repository/run.sh:ro \
            -v "$(pwd)/datasets":/repository/datasets:ro \
            -v temp_agentomics_volume_${AGENT_ID}:/workspace \
            ${TEST_EXEC[@]+"${TEST_EXEC[@]}"}
    else
        if [ -n "$FORK_FROM_RUN" ]; then
            FORK_FROM_RUN_ABS="$(cd "$FORK_FROM_RUN" && pwd)"
            FORK_MOUNT_FLAGS=(-v "${FORK_FROM_RUN_ABS}:/fork_source:ro")

            build_setup_fork_args /fork_source /workspace "$AGENT_ID" "$FORK_FROM_STEP" "$FORK_FROM_ITERATION"
            docker run --rm \
                -e PYTHONPATH=/repository/src \
                ${FORK_MOUNT_FLAGS[@]+"${FORK_MOUNT_FLAGS[@]}"} \
                -v "$(pwd)/src":/repository/src:ro \
                -v temp_agentomics_volume_${AGENT_ID}:/workspace \
                --entrypoint /opt/conda/envs/agentomics-env/bin/python \
                "$AGENTOMICS_IMAGE" /repository/src/runtime/setup_fork.py "${SETUP_FORK_ARGS[@]}"
        fi

        set +e
        docker run \
            --rm \
            -it \
            --name agentomics_cont_${AGENT_ID} \
            ${ENV_FILE_ARGS[@]+"${ENV_FILE_ARGS[@]}"} \
            -e AGENT_ID=${AGENT_ID} \
            -e AGENTOMICS_VERBOSITY=${VERBOSITY} \
            -e PYTHONWARNINGS=ignore \
            ${GPU_FLAGS[@]+"${GPU_FLAGS[@]}"} \
            ${OLLAMA_FLAGS[@]+"${OLLAMA_FLAGS[@]}"} \
            ${DOCKER_API_KEY_ENV_VARS[@]+"${DOCKER_API_KEY_ENV_VARS[@]}"} \
            ${CODEX_DOCKER_FLAGS[@]+"${CODEX_DOCKER_FLAGS[@]}"} \
            -v "$(pwd)/src":/repository/src:ro \
            -v "$(pwd)/datasets":/repository/datasets:ro \
            -v temp_agentomics_volume_${AGENT_ID}:/workspace \
            ${RUN_EXEC[@]+"${RUN_EXEC[@]}"}
        RUN_EXIT_CODE=$?
        set -e

        if [ "$LIST_MODE" = true ]; then
            exit "$RUN_EXIT_CODE"
        fi
        if [[ "$RUN_EXIT_CODE" -ne 0 ]]; then
            warn "Run container exited with code ${RUN_EXIT_CODE}. Exporting available run state before exiting."
        fi
        RUN_SUCCEEDED=true
        if docker run --rm -v temp_agentomics_volume_${AGENT_ID}:/workspace busybox test -f "/workspace/best_iteration_snapshot/runtime_info/iteration_metadata.json"; then
            echo "Running final evaluation on test set"
            docker run \
                --rm \
                --name agentomics_test_eval_cont_${AGENT_ID} \
                ${ENV_FILE_ARGS[@]+"${ENV_FILE_ARGS[@]}"} \
                -e PYTHONPATH=/repository/src \
                -e PYTHONWARNINGS=ignore \
                ${GPU_FLAGS[@]+"${GPU_FLAGS[@]}"} \
                -v "$(pwd)/src":/repository/src:ro \
                -v "$(pwd)/test_datasets":/repository/test_datasets:ro \
                -v temp_agentomics_volume_${AGENT_ID}:/workspace \
                --entrypoint /opt/conda/envs/agentomics-env/bin/python \
                "$AGENTOMICS_IMAGE" src/run_logging/test_evaluation.py \
                    --workspace-dir /workspace \
                    --test-datasets-dir /repository/test_datasets
        else
            RUN_SUCCEEDED=false
        fi

        mkdir -p "outputs/${AGENT_ID}"
        docker run --rm -v temp_agentomics_volume_${AGENT_ID}:/workspace busybox chmod -R a+rX /workspace/ || true
        docker run --rm -v temp_agentomics_volume_${AGENT_ID}:/source busybox tar -c --exclude='./run/shared/.conda' -C /source . | tar -x -C "outputs/${AGENT_ID}/"

        if [[ "$RUN_SUCCEEDED" = true ]]; then
            docker run --rm \
              -u "$(id -u):$(id -g)" \
              -e PYTHONPATH=/repository/src \
              -v "$(pwd)/src":/repository/src:ro \
              -v "$(pwd)/outputs/${AGENT_ID}":/agent_out \
              --entrypoint /opt/conda/envs/agentomics-env/bin/python \
              "$AGENTOMICS_IMAGE" -m runtime.iteration_reports --agent-dir /agent_out

            MPLCONFIGDIR_IN_CONTAINER="/tmp/mplconfig"
            docker run --rm \
              -u "$(id -u):$(id -g)" \
              -e MPLCONFIGDIR="$MPLCONFIGDIR_IN_CONTAINER" \
              -e PYTHONPATH=/repository/src \
              -v "$(pwd)":/repository \
              -v "$(pwd)/outputs/${AGENT_ID}":/agent_out \
              --entrypoint /opt/conda/envs/agentomics-env/bin/python \
              "$AGENTOMICS_IMAGE" /repository/src/runtime/generate_final_reports.py \
                --agent-dir /agent_out \
                --test-datasets-dir /repository/test_datasets

            echo "PDF reports ready at: outputs/${AGENT_ID}/reports/pdf/"

            if [ "$USE_PROVISIONING_KEY" = true ]; then
                echo "Logging costs and cleaning up temporary API key"
                docker run --rm \
                  -u "$(id -u):$(id -g)" \
                  ${ENV_FILE_ARGS[@]+"${ENV_FILE_ARGS[@]}"} \
                  ${DOCKER_API_KEY_ENV_VARS[@]+"${DOCKER_API_KEY_ENV_VARS[@]}"} \
                  -e PYTHONPATH=/repository/src \
                  -v "$(pwd)/src":/repository/src:ro \
                  -v "$(pwd)/outputs/${AGENT_ID}":/agent_out \
                  --entrypoint /opt/conda/envs/agentomics-env/bin/python \
                  "$AGENTOMICS_IMAGE" /repository/src/utils/api_keys.py cleanup-and-log \
                    --config-path /agent_out/run/shared/config.json \
                    --api-key-hash "$TEMP_API_KEY_HASH"
            fi

            if [ "$ALL_ITERATIONS_TEST" = true ]; then
                echo "Running held-out test evaluation for all archived iterations"
                docker run --rm \
                  ${ENV_FILE_ARGS[@]+"${ENV_FILE_ARGS[@]}"} \
                  -e PYTHONPATH=/repository/src \
                  -e PYTHONWARNINGS=ignore \
                  ${GPU_FLAGS[@]+"${GPU_FLAGS[@]}"} \
                  -v "$(pwd)/src":/repository/src:ro \
                  -v "$(pwd)/test_datasets":/repository/test_datasets:ro \
                  -v "$(pwd)/outputs/${AGENT_ID}":/agent_out \
                  --entrypoint /opt/conda/envs/agentomics-env/bin/python \
                  "$AGENTOMICS_IMAGE" -m runtime.stealth_test_evaluation \
                    --agent-dir /agent_out \
                    --test-datasets-dir /repository/test_datasets
            fi

            write_outputs_readme "${AGENT_ID}"

            echo -e "${GREEN}Run finished. Report and files can be found in outputs/${AGENT_ID}${NOCOLOR}"
            echo -e "${GREEN}To run inference on new data, use ./scripts/inference.sh --agent-dir outputs/${AGENT_ID} --input <path_to_input_folder> --output <path_to_output_csv>${NOCOLOR}"
        else
            docker run --rm \
              -u "$(id -u):$(id -g)" \
              -e PYTHONPATH=/repository/src \
              -v "$(pwd)/src":/repository/src:ro \
              -v "$(pwd)/outputs/${AGENT_ID}":/agent_out \
              --entrypoint /opt/conda/envs/agentomics-env/bin/python \
              "$AGENTOMICS_IMAGE" -m runtime.iteration_reports --agent-dir /agent_out
            warn "Agent didn't produce any valid best iteration snapshot. Exported run files for later continuation to outputs/${AGENT_ID}."
            write_outputs_readme "${AGENT_ID}"
        fi

        if [[ "$RUN_EXIT_CODE" -ne 0 ]]; then
            exit "$RUN_EXIT_CODE"
        fi
        if [[ "$RUN_SUCCEEDED" = false ]]; then
            exit 1
        fi

      fi
  fi
