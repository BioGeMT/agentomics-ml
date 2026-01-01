#!/usr/bin/env bash

# Ensure we are running under bash even if invoked via sh/zsh
if [ -z "${BASH_VERSION:-}" ]; then
  exec /usr/bin/env bash "$0" "$@"
fi

set -euo pipefail

# ABLATION STUDY CONFIGURATION
MODELS=(
    "gpt-oss:20b"
    "qwen3-coder:30b"
    # "claude-3-5-sonnet"
)

DATASETS=(
    "AGO2_CLASH_Hejret"
)

# LLM Provider (openrouter, anthropic, openai, ollama, google)
PROVIDER="ollama"

# Validation metric
VAL_METRIC="ACC"

# Number of iterations per run
ITERATIONS=5

# Number of repetitions for each ablation setting
REPETITIONS=1

# User prompt (optional)
USER_PROMPT="Create the best possible machine learning model that will generalize to new unseen data."

# Additional W&B tags (optional)
TAGS=(
    "ablation_study_test_friday"
    # "experiment_v1"
)

# Runtime flags
CPU_ONLY=false
OLLAMA=true
USE_PROVISIONING_KEY=false
SPEND_LIMIT=10

# ============================================
# Ablation configurations
# ============================================

ABLATION_CONFIGS=(
    "no_final_outcome:final_outcome"
    "baseline:"
    "no_data_exploration:data_exploration"
    "no_data_split:data_split"
    "no_data_representation:data_representation"
    "no_model_architecture:model_architecture"
    "no_model_training:model_training"
)

# ============================================
# Print configuration summary
# ============================================

echo "============================================"
echo "ABLATION STUDY CONFIGURATION"
echo "============================================"
echo "Models: ${MODELS[*]}"
echo "Datasets: ${DATASETS[*]}"
echo "Provider: $PROVIDER"
echo "Val Metric: $VAL_METRIC"
echo "Iterations per run: $ITERATIONS"
echo "Repetitions: $REPETITIONS"
echo "Tags: ${TAGS[*]}"
echo ""
TOTAL_RUNS=$((${#MODELS[@]} * ${#DATASETS[@]} * ${#ABLATION_CONFIGS[@]} * $REPETITIONS))
echo "Total runs: $TOTAL_RUNS (${#MODELS[@]} models × ${#DATASETS[@]} datasets × ${#ABLATION_CONFIGS[@]} ablations × $REPETITIONS repetitions)"
echo "============================================"
echo ""

# Build Docker images once
echo "Building the run image"
docker build --progress=quiet -t agentomics_img -f Dockerfile .
echo "Build done"

echo "Building the data preparation image"
docker build -t agentomics_prepare_img -f Dockerfile.prepare .
echo "Build done"

PROVIDERS_CONFIG_FILE="src/utils/providers/configured_providers.yaml"
API_KEY_NAMES=$(grep -E 'apikey:' "$PROVIDERS_CONFIG_FILE" | grep -o '\${[^}]*}' | tr -d '${}' | sort -u)
DOCKER_API_KEY_ENV_VARS_BASE=()
for KEY_NAME in $API_KEY_NAMES; do
    if [ -n "${!KEY_NAME:-}" ]; then
        DOCKER_API_KEY_ENV_VARS_BASE+=(-e "$KEY_NAME=${!KEY_NAME}")
        echo "Adding API key env var to docker: $KEY_NAME"
    fi
done

GPU_FLAGS=()
if [ "$CPU_ONLY" = false ]; then
    GPU_FLAGS+=(--gpus all)
    GPU_FLAGS+=(--env NVIDIA_VISIBLE_DEVICES=all)
fi

OLLAMA_FLAGS=()
if [ "$OLLAMA" = true ]; then
    OLLAMA_FLAGS+=(--add-host=host.docker.internal:host-gateway)
fi

# Run ablation loops
for model in "${MODELS[@]}"; do
    for dataset in "${DATASETS[@]}"; do
        for ablation_config in "${ABLATION_CONFIGS[@]}"; do
            ABLATION_NAME="${ablation_config%%:*}"
            STEPS_TO_SKIP="${ablation_config#*:}"

            for repetition in $(seq 1 $REPETITIONS); do
                echo "Model: $model"
                echo "Dataset: $dataset"
                echo "Ablation: $ABLATION_NAME"
                echo "Repetition: $repetition / $REPETITIONS"
                echo "========================================"
                echo ""

                # Create unique agent ID for this run
                AGENT_ID=$(docker run --rm -u $(id -u):$(id -g) -v "$(pwd)":/repository:ro --entrypoint \
                           /opt/conda/envs/agentomics-env/bin/python agentomics_img /repository/src/utils/create_user.py)

                # Run data preparation
                docker run \
                    -u $(id -u):$(id -g) \
                    --rm \
                    -it \
                    --name agentomics_prepare_cont_${AGENT_ID} \
                    -v "$(pwd)":/repository \
                    agentomics_prepare_img

                # Create volume for this run
                docker volume create temp_agentomics_volume_${AGENT_ID}

                # Create temporary API key for cost tracking if enabled
                TEMP_API_KEY_HASH=""
                if [ "$USE_PROVISIONING_KEY" = true ]; then
                    if ! conda env list | grep -q "^agentomics-env "; then
                        echo "Creating agentomics-env conda environment"
                        conda env create -f environment.yaml -q
                    fi
                    echo "Creating temporary API key with spend limit: $SPEND_LIMIT"
                    API_KEY_OUTPUT=$(PYTHONPATH="$(pwd)/src" conda run -n agentomics-env python src/utils/api_keys_utils.py create --name "ablation_${ABLATION_NAME}_${model}_${dataset}_rep${repetition}_$(date +%s)" --limit "$SPEND_LIMIT")
                    TEMP_API_KEY=$(echo "$API_KEY_OUTPUT" | cut -d',' -f1)
                    TEMP_API_KEY_HASH=$(echo "$API_KEY_OUTPUT" | cut -d',' -f2)
                    export OPENROUTER_API_KEY="$TEMP_API_KEY"
                fi

                # Build arguments array
                AGENTOMICS_ARGS=(
                    --model "$model"
                    --dataset-name "$dataset"
                    --val-metric "$VAL_METRIC"
                    --iterations "$ITERATIONS"
                    --user-prompt "$USER_PROMPT"
                    --provider "$PROVIDER"
                )

                # Add tags
                for tag in "${TAGS[@]}"; do
                    AGENTOMICS_ARGS+=(--tags "$tag")
                done
                AGENTOMICS_ARGS+=(--tags "ablation:${ABLATION_NAME}")

                # Add steps to skip if not baseline
                if [ -n "$STEPS_TO_SKIP" ]; then
                    AGENTOMICS_ARGS+=(--steps-to-skip "$STEPS_TO_SKIP")
                fi

                # Prepare API key environment variables for this run
                DOCKER_API_KEY_ENV_VARS=("${DOCKER_API_KEY_ENV_VARS_BASE[@]}")
                if [ "$USE_PROVISIONING_KEY" = true ]; then
                    DOCKER_API_KEY_ENV_VARS+=(-e "OPENROUTER_API_KEY=${OPENROUTER_API_KEY}")
                fi

                echo "Debug: Full command arguments:"
                echo "${AGENTOMICS_ARGS[@]}"
                echo ""
                echo "Starting experiment..."
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
                    --entrypoint /opt/conda/envs/agentomics-env/bin/python \
                    agentomics_img /repository/src/run_agent.py "${AGENTOMICS_ARGS[@]}"

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
                    agentomics_img src/run_logging/evaluate_log_test.py || echo "Test evaluation failed or skipped"

                mkdir -p outputs/ablation_results/${AGENT_ID}/best_run_files outputs/ablation_results/${AGENT_ID}/reports outputs/ablation_results/${AGENT_ID}/agent_logs outputs/ablation_results/${AGENT_ID}/extras

                # Copy snapshot files if they exist
                docker run --rm -u $(id -u):$(id -g) \
                    -v temp_agentomics_volume_${AGENT_ID}:/source \
                    -v $(pwd)/outputs/ablation_results/${AGENT_ID}:/dest \
                    busybox sh -c 'if [ -d /source/snapshots/${AGENT_ID} ]; then cp -r /source/snapshots/${AGENT_ID}/. /dest/best_run_files/; else echo "No snapshot to copy"; fi' \
                    || echo "Snapshot copy failed"

                # Copy reports from all iterations (may be partial if some iterations failed)
                docker run --rm -u $(id -u):$(id -g) \
                    -v temp_agentomics_volume_${AGENT_ID}:/source \
                    -v $(pwd)/outputs/ablation_results/${AGENT_ID}:/dest \
                    busybox sh -c 'if [ -d /source/reports/${AGENT_ID} ]; then cp -r /source/reports/${AGENT_ID}/. /dest/reports/; else echo "No reports to copy"; fi' \
                    || echo "Reports copy failed"

                # Copy agent logs from all iterations (full conversation history)
                docker run --rm -u $(id -u):$(id -g) \
                    -v temp_agentomics_volume_${AGENT_ID}:/source \
                    -v $(pwd)/outputs/ablation_results/${AGENT_ID}:/dest \
                    busybox sh -c 'if [ -d /source/runs/${AGENT_ID}/agent_logs ]; then cp -r /source/runs/${AGENT_ID}/agent_logs/. /dest/agent_logs/; else echo "No agent logs to copy"; fi' \
                    || echo "Agent logs copy failed"

                # Copy extras directory (contains config.json for cost logging)
                docker run --rm -u $(id -u):$(id -g) \
                    -v temp_agentomics_volume_${AGENT_ID}:/source \
                    -v $(pwd)/outputs/ablation_results/${AGENT_ID}:/dest \
                    busybox sh -c 'if [ -d /source/extras ]; then cp -r /source/extras/. /dest/extras/; else echo "No extras to copy"; fi' \
                    || echo "Extras copy failed"

                # Log costs and cleanup temporary API key if provisioning is enabled
                if [ "$USE_PROVISIONING_KEY" = true ]; then
                    echo "Logging costs and cleaning up temporary API key"
                    CONFIG_PATH="outputs/ablation_results/${AGENT_ID}/extras/config.json"
                    PYTHONPATH="$(pwd)/src" conda run -n agentomics-env python src/utils/api_keys_utils.py cleanup-and-log --config-path "$CONFIG_PATH" --api-key-hash "$TEMP_API_KEY_HASH" || echo "Cost logging failed"
                fi

                docker volume rm temp_agentomics_volume_${AGENT_ID}
            done
        done
    done
done

GREEN='\033[0;32m'
NOCOLOR='\033[0m'
echo ""
echo "========================================"
echo -e "${GREEN}ABLATION STUDY COMPLETE${NOCOLOR}"
echo "========================================"
echo "Results directory: outputs/ablation_results/"
echo "========================================"