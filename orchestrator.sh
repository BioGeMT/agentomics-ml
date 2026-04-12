#!/usr/bin/env bash

./download_example_datasets.sh

REPOS_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)" 

SPEND_LIMIT=100
MODELS=("openai/gpt-5.1-codex-max")
ITERATIONS=100 # Set to a large number because timeout will take precedence anyways
TIME_BUDGET_S=$(( 8 * 60 * 60 )) # 8 hours
SPLIT_TIME_BUDGET_S=$(( 4 * 60 * 60 )) # allowing to re-split for 4 hours
# BASELINE_ITERS currently not parametrizable, hardcoded to 4
SPLIT_ALLOWED_ITERS=0 #SPLIT_TIME_BUDGET gets precedence over this
TAGS=("agentomics_reproduce_v1")
REPETITIONS=3
USER_PROMPT="Create a machine learning model that will generalize to new unseen data."

GENOMIC_DATASETS=(
    "AGO2_CLASH_Hejret2023"
    "human_enhancers_cohn"
    "human_enhancers_ensembl"
    "human_ocr_ensembl"
    "drosophila_enhancers_stark"
)
GENOMIC_DATASETS_VAL_METRICS=(
    "AGO2_CLASH_Hejret2023:AUPRC"
    "human_enhancers_cohn:ACC"
    "human_enhancers_ensembl:ACC"
    "human_ocr_ensembl:ACC"
    "drosophila_enhancers_stark:ACC"
)

declare -A metric_map
  for config in "${GENOMIC_DATASETS_VAL_METRICS[@]}"; do
      dataset="${config%%:*}"
      metric="${config##*:}"
      metric_map["$dataset"]="$metric"
  done

./download_example_datasets.sh

for repetition in $(seq 1 $REPETITIONS); do
    for dataset in "${GENOMIC_DATASETS[@]}"; do
        for model in "${MODELS[@]}"; do
            ./run.sh \
                --model "$model" \
                --dataset "$dataset" \
                --iterations "$ITERATIONS" \
                --use-provisioning-key \
                --spend-limit "$SPEND_LIMIT" \
                --user-prompt "$USER_PROMPT" \
                --split-allowed-iterations "$SPLIT_ALLOWED_ITERS" \
                --val-metric "${metric_map[$dataset]}" \
                --tags "${TAGS[@]}" \
                --timeout "$TIME_BUDGET_S" \
                --split-timeout "$SPLIT_TIME_BUDGET_S" \
                --foundation-model-type all
        done
    done
done
echo 'Orchestrator done'
