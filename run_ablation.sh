#!/usr/bin/env bash

# Ensure we are running under bash even if invoked via sh/zsh
if [ -z "${BASH_VERSION:-}" ]; then
  exec /usr/bin/env bash "$0" "$@"
fi

set -euo pipefail

# ============================================
# ABLATION STUDY CONFIGURATION
# Edit these parameters for your experiment
# ============================================

# Models to test (space-separated)
MODELS=(
    "gpt-4o-mini"
    # "claude-3-5-sonnet"
)

# Datasets to test (space-separated)
DATASETS=(
    "AGO2_CLASH_Hejret"
    # "diabetes"
    # "breast_cancer"
)

# LLM Provider (openrouter, anthropic, openai, ollama)
PROVIDER="openrouter"

# Validation metric
VAL_METRIC="ACC"

# Number of iterations per run
ITERATIONS=5

REPETITIONS=5  # Number of repetitions for each ablation setting

# Timeout per experiment in seconds
TIMEOUT=86400  # 24 hours

# User prompt (optional)
USER_PROMPT="Create the best possible machine learning model that will generalize to new unseen data."

# Additional W&B tags (optional)
TAGS=(
    "ablation_study"
    # "experiment_v1"
)

# ============================================
# Build argument array for Python script
# ============================================

ABLATION_ARGS=()

# Add models
for model in "${MODELS[@]}"; do
    ABLATION_ARGS+=(--models "$model")
done

# Add datasets
for dataset in "${DATASETS[@]}"; do
    ABLATION_ARGS+=(--datasets "$dataset")
done

# Add other parameters
ABLATION_ARGS+=(--provider "$PROVIDER")
ABLATION_ARGS+=(--val-metric "$VAL_METRIC")
ABLATION_ARGS+=(--iterations "$ITERATIONS")
ABLATION_ARGS+=(--user-prompt "$USER_PROMPT")
ABLATION_ARGS+=(--repetitions "$REPETITIONS")
ABLATION_ARGS+=(--timeout "$TIMEOUT")

# Add tags
for tag in "${TAGS[@]}"; do
    ABLATION_ARGS+=(--tags "$tag")
done

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
echo "Iterations: $ITERATIONS"
echo "Tags: ${TAGS[*]}"
echo ""
echo "Total runs: $((${#MODELS[@]} * ${#DATASETS[@]} * 7 * $REPETITIONS)) (${#MODELS[@]} models × ${#DATASETS[@]} datasets × 7 ablations x $REPETITIONS repetitions)"
echo "============================================"
echo ""

# ============================================
# Build Docker images and run ablation study
# ============================================

docker build -t agentomics_prepare_img -f Dockerfile.prepare .
docker run \
    -u $(id -u):$(id -g) \
    --rm \
    -it \
    --name agentomics_prepare_cont \
    -v "$(pwd)":/repository \
    agentomics_prepare_img

docker build -t agentomics_img .
docker volume create temp_agentomics_ablation_volume

docker run \
    -it \
    --rm \
    --name agentomics_ablation_cont \
    --gpus all \
    --env NVIDIA_VISIBLE_DEVICES=all \
    -v "$(pwd)/src":/repository/src:ro \
    -v "$(pwd)/prepared_datasets":/repository/prepared_datasets:ro \
    -v "$(pwd)/.env":/repository/.env:ro \
    -v temp_agentomics_ablation_volume:/workspace \
    --entrypoint /opt/conda/envs/agentomics-env/bin/python \
    agentomics_img /repository/src/run_ablation.py "${ABLATION_ARGS[@]}"

mkdir -p outputs/ablation_results
docker run --rm -u $(id -u):$(id -g) \
    -v temp_agentomics_ablation_volume:/source \
    -v $(pwd)/outputs:/dest \
    busybox cp -r /source/. /dest/ablation_results/

docker volume rm temp_agentomics_ablation_volume
