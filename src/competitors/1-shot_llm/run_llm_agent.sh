RUNS=1
# DATASETS=("human_nontata_promoters" "human_enhancers_cohn" "drosophila_enhancers_stark" "human_enhancers_ensembl" "human_ensembl_regulatory" "AGO2_CLASH_Hejret2023" "human_ocr_ensembl")
DATASETS=("human_nontata_promoters")
# claude 3.7 - 3$/M
# gpt 4.1 - 2$/M 
# gemini 2.5 pro preview - 1.25$/M
# MODELS=("anthropic/claude-3.7-sonnet" "openai/gpt-4.1-2025-04-14" "google/gemini-2.5-pro-preview-03-25")
MODELS=("openai/gpt-4.1-2025-04-14")
TEMP=1.0 #Openrouter default
MAX_TOKENS=10000
TAGS=("testing" "oneshot")

for DATASET in "${DATASETS[@]}"
do
  echo "Current dataset: $DATASET"
  
  for MODEL in "${MODELS[@]}"
  do
    echo "Running with model: $MODEL"
    
    for ((i=1; i<=$RUNS; i++))
    do
      python 1-shot_llm_agent_gpt4o-claude.py \
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