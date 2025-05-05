# "human_ensembl_regulatory" - SKIPPED FOR NOW
# DATASETS=("human_nontata_promoters" "human_enhancers_cohn" "drosophila_enhancers_stark" "human_enhancers_ensembl"  "AGO2_CLASH_Hejret2023" "human_ocr_ensembl")
DATASETS=("human_nontata_promoters")

# claude 3.7 - 3$/M
# gpt 4.1 - 2$/M 
# gpt o4-mini  - 1.10$/M
# gemini 2.5 pro preview - 1.25$/M
# qwen/qwen3-32b - 0.1$/M
# deepseek/deepseek-r1 - 0.5$/M
# deepseek/deepseek-chat (v3) - 0.38/M

# MODELS=("anthropic/claude-3.7-sonnet" "openai/gpt-4.1-2025-04-14" "openai/o4-mini" "google/gemini-2.5-pro-preview-03-25" "qwen/qwen3-32b" "deepseek/deepseek-r1" "deepseek/deepseek-chat")
MODELS=("openai/gpt-4.1-2025-04-14")

TEMP=1.0 #Openrouter default
MAX_TOKENS=10000
TAGS=("testing" "oneshot")
RUNS=1

for DATASET in "${DATASETS[@]}"
do
  echo "Current dataset: $DATASET"
  
  for MODEL in "${MODELS[@]}"
  do
    echo "Running with model: $MODEL"
    
    for ((i=1; i<=$RUNS; i++))
    do
      python 1-shot_llm_run.py \
        --dataset "$DATASET" \
        --model "$MODEL" \
        --temp "$TEMP" \
        --max-tokens "$MAX_TOKENS" \
        --tags "${TAGS[@]}" 

      # Capture exit status of the Python script
      STATUS=$?
    done
  done
done