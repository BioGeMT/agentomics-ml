RUNS=1
DATASET="human_nontata_promoters"
PROVIDER="openai"
MODEL="gpt-4o-2024-08-06"
TEMP=0.7
MAX_TOKENS=4000
TAGS=("testing" "oneshot")

for ((i=1; i<=$RUNS; i++))
do
  python 1-shot_llm_agent_gpt4o-claude.py \
    --dataset "$DATASET" \
    --provider "$PROVIDER" \
    --model "$MODEL" \
    --temp "$TEMP" \
    --max-tokens "$MAX_TOKENS" \
    --tags "${TAGS[@]}" 

  # Capture exit status of the Python script
  STATUS=$?
done