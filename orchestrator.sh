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
PULL_BRANCH="ismb_submission" #Branch to pull for biomlbench runs
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
BIOMLBENCH_DATASETS=(
  "polarishub/polaris-pkis2-egfr-wt-c-1"
  "polarishub/polaris-adme-fang-hclint-1"
  "polarishub/polaris-adme-fang-hppb-1"
  "polarishub/polaris-adme-fang-solu-1"
  "polarishub/tdcommons-cyp2d6-substrate-carbonmangels"
  "polarishub/tdcommons-lipophilicity-astrazeneca"
  "polarishub/tdcommons-herg"
  "polarishub/tdcommons-bbb-martins"
  "polarishub/tdcommons-caco2-wang"
  "proteingym-dms/SPIKE_SARS2_Starr_2020_binding"
  "proteingym-dms/SPA_STAAU_Tsuboyama_2023_1LP1"
  "proteingym-dms/PSAE_PICP2_Tsuboyama_2023_1PSE_indels"
  "proteingym-dms/CBX4_HUMAN_Tsuboyama_2023_2K28"
  "proteingym-dms/Q8EG35_SHEON_Campbell_2022_indels"
  "proteingym-dms/CSN4_MOUSE_Tsuboyama_2023_1UFM_indels"
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

for repetition in $(seq 1 $REPETITIONS); do
    for dataset in "${BIOMLBENCH_DATASETS[@]}"; do
        for model in "${MODELS[@]}"; do
            ./biomlbench/run_benchmarks.sh \
                --repos-dir "$REPOS_DIR" \
                --spend-limit "$SPEND_LIMIT" \
                --dset "$dataset" \
                --iterations "$ITERATIONS" \
                --split-allowed-iterations "$SPLIT_ALLOWED_ITERS" \
                --foundation-model-type all \
                --model "$model" \
                --user-prompt "$USER_PROMPT" \
                --pull-branch "$PULL_BRANCH" \
                --timeout "$TIME_BUDGET_S" \
                --split-timeout "$SPLIT_TIME_BUDGET_S" \
                --tags "${TAGS[@]}"
        done
    done
done

./competitors/setup.sh
for repetition in $(seq 1 $REPETITIONS); do
    conda run -n biomlbench-agents python competitors/run_competitors.py --agents zeroshot
done
echo 'Orchestrator done'
