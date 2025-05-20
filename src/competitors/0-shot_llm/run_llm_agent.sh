DATASETS=("human_nontata_promoters" "human_enhancers_cohn" "drosophila_enhancers_stark" "human_enhancers_ensembl"  "AGO2_CLASH_Hejret2023" "human_ocr_ensembl")
MODELS=("anthropic/claude-3.7-sonnet" "openai/gpt-4.1-2025-04-14" "openai/o4-mini" "qwen/qwen3-32b" "deepseek/deepseek-r1" "deepseek/deepseek-chat" "meta-llama/llama-4-maverick")

TEMP=1.0 #Openrouter default
TAGS=("testing")
RUNS=5
TIME_BUDGET_IN_HOURS=24

for DATASET in "${DATASETS[@]}"
do
  echo "Current dataset: $DATASET"
  
  for MODEL in "${MODELS[@]}"
  do
    echo "Running with model: $MODEL"
    
    for ((i=1; i<=$RUNS; i++))
    do
      python 0-shot_llm_run.py \
        --dataset "$DATASET" \
        --model "$MODEL" \
        --temp "$TEMP" \
        --tags "${TAGS[@]}" \
        --timeout $TIME_BUDGET_IN_HOURS

      # Capture exit status of the Python script
      STATUS=$?
    done
  done
done