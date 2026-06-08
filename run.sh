#!/usr/bin/env bash

# Get the absolute directory of this script
AGENTOMICS_DIR="$(cd "$(dirname "$0")" && pwd)"

source "$AGENTOMICS_DIR/scripts/bash_helpers.sh"

cd "$AGENTOMICS_DIR" || die "Cannot cd into repository directory: $AGENTOMICS_DIR"

AGENTOMICS_ARGS=()
TEST_MODE=false
CPU_ONLY=false
USE_PROVISIONING_KEY=false
SPEND_LIMIT=10
TIMEOUT_SECS=""
MODEL_NAME=""
PREFERRED_PROVIDER=""
DATASET_NAME=""
TASK_TYPE=""
VAL_METRIC=""
LIST_MODE=false
VERBOSITY="full"
FORK_FROM_RUN=""
FORK_FROM_STEP=""
FORK_FROM_ITERATION=""

show_help() {
    cat <<'EOF'
Usage: ./run.sh [OPTIONS]

Orchestrates the Agentomics training and evaluation process.

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
  --test              Run the project's integrated test suite.
  --cpu-only          Force Conda to run using CPU only (skip GPU configuration).
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
  Results are written to the workspace directory: 'outputs/<AGENT_ID>' by default, or
  \$AGENTOMICS_WORKSPACE_DIR if set (e.g. '/workspace' inside the Docker image — mount a
  host directory there to retrieve the results).
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
        --use-provisioning-key)
            USE_PROVISIONING_KEY=true
            shift
            ;;
        --spend-limit)
            require_opt_value "$1" "${2:-}"
            SPEND_LIMIT="$2"
            shift 2
            ;;
        --test)
            TEST_MODE=true
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

if [[ "$VERBOSITY" != "summary" && "$VERBOSITY" != "full" ]]; then
    die "Invalid --verbosity '$VERBOSITY'. Allowed: summary, full."
fi

EFFECTIVE_DATASET_NAME="$(resolve_effective_string_field "$DATASET_NAME" "$FORK_FROM_RUN" "dataset" true)"

if [[ -n "$FORK_FROM_RUN" && -n "$DATASET_NAME" && "$DATASET_NAME" != "$EFFECTIVE_DATASET_NAME" ]]; then
    warn "--dataset '$DATASET_NAME' is ignored for forked runs. Using '$EFFECTIVE_DATASET_NAME' from the source run config."
fi

if [[ "$LIST_MODE" = false && "$TEST_MODE" = false ]] && ! has_tty; then
    if [[ -z "$FORK_FROM_RUN" && ( -z "$MODEL_NAME" || -z "$DATASET_NAME" ) ]]; then
        die "Non-interactive runs require --model and --dataset (or --fork-from-run) (or run in an interactive terminal)"
    fi
fi

ensure_conda_env "agentomics-env" "$AGENTOMICS_DIR/envs/environment.yaml"

eval "$(conda shell.bash hook)"
conda activate agentomics-env

AGENT_ID=$(python src/utils/agent_id.py)
export AGENT_ID
export AGENTOMICS_VERBOSITY="$VERBOSITY"

WORKSPACE_DIR="${AGENTOMICS_WORKSPACE_DIR:-$AGENTOMICS_DIR/outputs/$AGENT_ID}"
mkdir -p "$WORKSPACE_DIR"
AGENTOMICS_ARGS+=(--workspace-dir "$WORKSPACE_DIR")

# allow git when running as root in container to avoid "dubious ownership" errors.
if [[ "$(id -u)" -eq 0 ]]; then
    git config --system --add safe.directory '*'
fi

PREPARED_DATASETS_DIR="$AGENTOMICS_DIR/prepared_datasets"
mkdir -p "$PREPARED_DATASETS_DIR"

if [ -n "$EFFECTIVE_DATASET_NAME" ]; then
    PREPARE_ARGS=(
        --dataset-dir "$AGENTOMICS_DIR/datasets/$EFFECTIVE_DATASET_NAME"
        --prepared-datasets-dir "$PREPARED_DATASETS_DIR"
    )
    if [ -n "$TASK_TYPE" ]; then
        PREPARE_ARGS+=(--task-type "$TASK_TYPE")
    fi
    python src/prepare_datasets.py "${PREPARE_ARGS[@]}"
else
    python src/prepare_datasets.py --prepare-all \
        --datasets-dir "$AGENTOMICS_DIR/datasets" \
        --prepared-datasets-dir "$PREPARED_DATASETS_DIR"
fi

AGENTOMICS_ARGS+=(--prepared-datasets-dir "$PREPARED_DATASETS_DIR")

if [[ ! -f "${START_ENV_PKG:-}" ]]; then
    AGENTOMICS_CACHE_DIR="$AGENTOMICS_DIR/.cache"
    mkdir -p "$AGENTOMICS_CACHE_DIR"
    local_pack="$AGENTOMICS_CACHE_DIR/agent_start_env.tar"
    if [[ ! -f "$local_pack" ]]; then
        ensure_conda_env "agent_start_env" "$AGENTOMICS_DIR/envs/environment_agent.yaml"
        conda run -n agent_start_env conda-pack --format tar -o "$local_pack"
    fi
    export START_ENV_PKG="$local_pack"
fi

if [ "$TEST_MODE" = true ]; then
    set +e
    PYTHONPATH="$AGENTOMICS_DIR/src" python -m test.run_all_tests \
        --workspace-dir "$WORKSPACE_DIR" \
        --prepared-datasets-dir "$PREPARED_DATASETS_DIR"
    TEST_EXIT=$?
    set -e
    exit "$TEST_EXIT"
fi

TEMP_API_KEY_HASH=""
if [ "$USE_PROVISIONING_KEY" = true ]; then
    echo "Creating temporary API key with spend limit: $SPEND_LIMIT"
    API_KEY_OUTPUT=$(PYTHONPATH="$(pwd)/src" conda run -n agentomics-env python src/utils/api_keys.py create --name "agentomics_run_$(date +%s)" --limit "$SPEND_LIMIT")
    TEMP_API_KEY=$(echo "$API_KEY_OUTPUT" | cut -d',' -f1)
    TEMP_API_KEY_HASH=$(echo "$API_KEY_OUTPUT" | cut -d',' -f2)
    export OPENROUTER_API_KEY="$TEMP_API_KEY"
fi

if [[ -d /mnt/codex-host ]]; then
    mkdir -p /tmp/codex
    if [[ -f /mnt/codex-host/auth.json ]]; then
        cp -f /mnt/codex-host/auth.json /tmp/codex/auth.json
    fi
    if [[ -f /mnt/codex-host/models_cache.json ]]; then
        cp -f /mnt/codex-host/models_cache.json /tmp/codex/models_cache.json
    fi
    export CODEX_AUTH_FILE=/tmp/codex/auth.json
    export CODEX_MODELS_CACHE_FILE=/tmp/codex/models_cache.json
fi

if [ "$CPU_ONLY" = true ]; then
    export CUDA_VISIBLE_DEVICES=""
fi

if [ -n "$FORK_FROM_RUN" ]; then
    FORK_FROM_RUN_ABS="$(cd "$FORK_FROM_RUN" && pwd)"
    build_setup_fork_args "$FORK_FROM_RUN_ABS" "$WORKSPACE_DIR" "$AGENT_ID" "$FORK_FROM_STEP" "$FORK_FROM_ITERATION"
    PYTHONPATH="$AGENTOMICS_DIR/src" python src/runtime/setup_fork.py "${SETUP_FORK_ARGS[@]}"
fi

RUN_EXIT_CODE=0
if [[ -n "$TIMEOUT_SECS" ]]; then
    need_cmd timeout
    [[ "$TIMEOUT_SECS" =~ ^[0-9]+$ ]] || die "--timeout must be an integer number of seconds (got: $TIMEOUT_SECS)"
    set +e
    timeout "$TIMEOUT_SECS" python src/run_agent_interactive.py "${AGENTOMICS_ARGS[@]}"
    RUN_EXIT_CODE=$?
    set -e
    if [[ "$RUN_EXIT_CODE" -eq 124 ]]; then
        echo "Timed out after $TIMEOUT_SECS seconds"
    elif [[ "$RUN_EXIT_CODE" -ne 0 ]]; then
        warn "Run process exited with code ${RUN_EXIT_CODE}."
    fi
else
    set +e
    python src/run_agent_interactive.py "${AGENTOMICS_ARGS[@]}"
    RUN_EXIT_CODE=$?
    set -e
    if [[ "$RUN_EXIT_CODE" -ne 0 ]]; then
        warn "Run process exited with code ${RUN_EXIT_CODE}."
    fi
fi

if [[ "$LIST_MODE" = true ]]; then
    exit "$RUN_EXIT_CODE"
fi

RUN_SUCCEEDED=true
if [[ ! -f "$WORKSPACE_DIR/best_iteration_snapshot/runtime_info/iteration_metadata.json" ]]; then
    RUN_SUCCEEDED=false
fi

PYTHONPATH="$AGENTOMICS_DIR/src" python -m runtime.iteration_reports --agent-dir "$WORKSPACE_DIR" \
    || warn "Failed to generate iteration reports"
PYTHONPATH="$AGENTOMICS_DIR/src" python src/runtime/generate_final_reports.py \
    --agent-dir "$WORKSPACE_DIR" --prepared-datasets "$PREPARED_DATASETS_DIR" \
    || warn "Failed to generate PDF reports"

write_outputs_readme "$WORKSPACE_DIR" "$AGENT_ID"

if [[ "$RUN_SUCCEEDED" = true ]]; then
    echo -e "${GREEN}Run finished. Files can be found in $WORKSPACE_DIR${NOCOLOR}"
    echo -e "${GREEN}To run inference on new data, use ./scripts/inference.sh --agent-dir $WORKSPACE_DIR --input <path_to_input_csv> --output <path_to_output_csv>${NOCOLOR}"
else
    warn "Agent didn't produce a valid best iteration snapshot. Run artifacts are in $WORKSPACE_DIR."
fi

if [ "$USE_PROVISIONING_KEY" = true ]; then
    CONFIG_PATH="$WORKSPACE_DIR/run/shared/config.json"
    if [[ -f "$CONFIG_PATH" ]]; then
        echo "Logging costs and cleaning up temporary API key"
        PYTHONPATH="$AGENTOMICS_DIR/src" python src/utils/api_keys.py cleanup-and-log \
            --config-path "$CONFIG_PATH" --api-key-hash "$TEMP_API_KEY_HASH" \
            || warn "Failed to clean up temporary API key (hash: $TEMP_API_KEY_HASH)"
    else
        warn "Config not found at $CONFIG_PATH; cannot log costs/clean up key (hash: $TEMP_API_KEY_HASH)"
    fi
fi

if [[ -n "${HOST_UID:-}" ]]; then
    chown -R "$HOST_UID:${HOST_GID:-$HOST_UID}" "$WORKSPACE_DIR"
fi

if [[ "$RUN_EXIT_CODE" -ne 0 ]]; then
    exit "$RUN_EXIT_CODE"
fi
if [[ "$RUN_SUCCEEDED" = false ]]; then
    exit 1
fi
