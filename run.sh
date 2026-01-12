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

write_outputs_readme() {
    local agent_id="$1"
    local out_dir="outputs/${agent_id}"
    local best_iter="UNKNOWN"

    if [[ -f "${out_dir}/best_run_files/iteration_number.txt" ]]; then
        best_iter="$(cat "${out_dir}/best_run_files/iteration_number.txt" | tr -d '[:space:]')"
        [[ -z "$best_iter" ]] && best_iter="UNKNOWN"
    fi
    cat > "${out_dir}/README.md" << EOF

This directory contains the full results of one run (**AGENT_ID: ${agent_id}**).
It includes: (1) the **best model/run chosen across iterations**, (2) per-iteration run artifacts,
(3) reports, and (4) logs/extras.

---

**Best iteration selected:** **${best_iter}**

**Note:** Iterations are **0-indexed** (first iteration is `0`).

The best iteration number is stored in:
- \`best_run_files/iteration_number.txt\`

What to use:
- **Best model + code:** \`best_run_files/\`
- **Best report:** \`reports/run_report_iter_${best_iter}.txt\`

---
\`\`\`
outputs/${agent_id}/
├── best_run_files/                 # Best iteration only (selected automatically)
│   ├── train.py                    # Training script used to produce the best model
│   ├── inference.py                # Inference script for predictions on new data
│   ├── training_artifacts/         # Serialized artifacts (e.g. \`model.joblib\`)
│   │   └── model.joblib
│   │   └── ...
│   ├── validation_metrics.txt
│   ├── train_metrics.txt
│   ├── eval_predictions_train.csv  # Predictions on training set
│   ├── eval_predictions_validation.csv
│   ├── structured_outputs.txt
│   ├── config.json
│   ├── conda_environment.yml
│   └── iteration_number.txt        # Chosen best iteration index
│
├── run_files/                      # All iterations
│   ├── train.csv                   # Full training split
│   ├── validation.csv              # Full validation split
│   ├── iteration_0/                # Snapshot of iteration 0
│   ├── iteration_1/                # Snapshot of iteration 1
│   └── ...                         # Additional iterations if present
│
├── reports/                        # Human-readable reports per iteration
│   ├── run_report_iter_0.md
│   ├── run_report_iter_1.md
│   └── ...
│
├── extras/                         # Logs and debugging information
│   ├── run_logs/
│   └── test_logs/
│
└── README.md                       # This file
\`\`\`
---

## Creating structured PDF reports

Run a helper for plot visualization of the results

\`\`\`bash
./generate_reports.sh <output_folder_name>
\`\`\`

## Running inference on new data

An inference helper command at the end, e.g.:

\`\`\`bash
./inference.sh --agent-dir outputs/${agent_id} --input <path_to_input_csv> --output <path_to_output_csv>
\`\`\`

Inference relies on:
- \`best_run_files/inference.py\`
- \`best_run_files/training_artifacts/\`

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
    write_outputs_readme "${AGENT_ID}"
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

    echo "Building the run image"
    docker build -t agentomics_img -f Dockerfile .
    echo "Build done"
    AGENT_ID=$(docker run --rm -u $(id -u):$(id -g) -v "$(pwd)":/repository:ro --entrypoint \
               /opt/conda/envs/agentomics-env/bin/python agentomics_img /repository/src/utils/create_user.py)

    echo "Building the data preparation image"
    docker build -t agentomics_prepare_img -f Dockerfile.prepare .
    echo "Build done"
    docker run \
        -u $(id -u):$(id -g) \
        --rm \
        -it \
        -e PYTHONWARNINGS=ignore \
        --name agentomics_prepare_cont_${AGENT_ID} \
        -v "$(pwd)":/repository \
        agentomics_prepare_img

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
            --rm \
            -it \
            --name agentomics_cont_${AGENT_ID} \
            ${ENV_FILE_ARGS[@]+"${ENV_FILE_ARGS[@]}"} \
            -e AGENT_ID=${AGENT_ID} \
            -e PYTHONWARNINGS=ignore \
            ${GPU_FLAGS[@]+"${GPU_FLAGS[@]}"} \
            ${OLLAMA_FLAGS[@]+"${OLLAMA_FLAGS[@]}"} \
            ${DOCKER_API_KEY_ENV_VARS[@]+"${DOCKER_API_KEY_ENV_VARS[@]}"} \
            -v "$(pwd)/src":/repository/src:ro \
            -v "$(pwd)/prepared_datasets":/repository/prepared_datasets:ro \
            -v temp_agentomics_volume_${AGENT_ID}:/workspace \
            agentomics_img ${AGENTOMICS_ARGS+"${AGENTOMICS_ARGS[@]}"}

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
            ${GPU_FLAGS[@]+"${GPU_FLAGS[@]}"} \
            -v "$(pwd)/src":/repository/src:ro \
            -v "$(pwd)/prepared_datasets":/repository/prepared_datasets:ro \
            -v "$(pwd)/prepared_test_sets":/repository/prepared_test_sets:ro \
            -v temp_agentomics_volume_${AGENT_ID}:/workspace \
            --entrypoint /opt/conda/envs/agentomics-env/bin/python \
            agentomics_img src/run_logging/evaluate_log_test.py

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
        write_outputs_readme "${AGENT_ID}"
        echo -e "${GREEN}Run finished. Report and files can be found in outputs/${AGENT_ID}${NOCOLOR}"
        echo -e "${GREEN}To run inference on new data, use ./inference.sh --agent-dir outputs/${AGENT_ID} --input <path_to_input_csv> --output <path_to_output_csv>${NOCOLOR}"

      fi
  fi
