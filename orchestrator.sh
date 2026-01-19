#!/usr/bin/env bash

REPOS_DIR="/home/$USER/repos" #this needs to be configured to the agentomics repository parent directory (biomlbench will be pulled as a sibling to the agentomics repo)

RUN_COMPETITORS_ZEROSHOT=true

SPEND_LIMIT=100
MODELS=("openai/gpt-5.1-codex")
ITERATIONS=1000000 # Set very high so timeout is the limiting factor
TIME_BUDGET_S=$(( 8 * 60 * 60 )) # 8 hours, biomlbench datasets are set to 8h automatically and will not react to this
SPLIT_ALLOWED_ITERS=4
PULL_BRANCH="run_experiments" #Branch to pull for biomlbench runs
TAGS=("experiment_orchestrator" "test_run")
REPETITIONS=3
USER_PROMPT="Create a machine learning model that will generalize to new unseen data."

GENOMIC_DATASETS=(
    "AGO2_CLASH_Hejret"
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
    "AGO2_CLASH_Hejret:AUPRC"
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
                --timeout "$TIME_BUDGET_S"
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
                --model "$model" \
                --user-prompt "$USER_PROMPT" \
                --pull-branch "$PULL_BRANCH" \
                --tags "${TAGS[@]}"
        done
    done
done

if [[ "$RUN_COMPETITORS_ZEROSHOT" == "true" ]]; then
    for repetition in $(seq 1 $REPETITIONS); do
        conda run -n biomlbench-agents python competitors/run_competitors.py --agents zeroshot
    done
fi
echo 'Orchestrator done'
