MODELS=("openai/gpt-5" "openai/gpt-5-codex" "anthropic/claude-sonnet-4.5" "anthropic/claude-haiku-4.5")
DATASETS=("AGO2_CLASH_Hejret")
ITERATIONS=5
TIME_BUDGET_H=8
SPLIT_ALLOWED_ITERS=3
TAGS=("experiment_orchestrator" "test_run")
REPETITIONS=1

DATASETS_VAL_METRICS=(
    "AGO2_CLASH_Hejret:AUPRC"
    "human_enhancers_cohn:ACC"
    "human_enhancers_ensembl:ACC"
    "human_ocr_ensembl:ACC"
    "drosophila_enhancers_stark:ACC"
)

declare -A metric_map
  for config in "${DATASETS_VAL_METRICS[@]}"; do
      dataset="${config%%:*}"
      metric="${config##*:}"
      metric_map["$dataset"]="$metric"
  done

for repetition in $(seq 1 $REPETITIONS); do
    for dataset in "${DATASETS[@]}"; do
        for model in "${MODELS[@]}"; do
            ./run.sh \
                --model "$model" \
                --dataset "$dataset" \
                --iterations "$ITERATIONS" \
                --split-allowed-iterations "$SPLIT_ALLOWED_ITERS" \
                --val-metric "${metric_map[$dataset]}" \
                --tags "${TAGS[@]}"
                #----time-budget-h "$TIME_LIMIT" \
        done
    done
done