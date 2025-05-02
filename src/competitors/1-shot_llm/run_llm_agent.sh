#!/bin/bash
DATASET="human_non_tata_promoters"
PROVIDER="openai"
MODEL="gpt-4o-2024-08-06"
TEMP=0.7
MAX_TOKENS=4000
RUNS=1

for ((i=1; i<=$RUNS; i++))
do
  echo "=========================================="
  echo "STARTING RUN $i of $RUNS"
  echo "=========================================="

  python 1-shot_llm_agent_gpt4o-claude.py \
    --dataset "$DATASET" \
    --provider "$PROVIDER" \
    --model "$MODEL" \
    --temp "$TEMP" \
    --max-tokens "$MAX_TOKENS" \

  # Capture exit status of the Python script
  STATUS=$?

  echo "=========================================="
  echo "COMPLETED RUN $i of $RUNS (Status: $STATUS)"
  echo "=========================================="

done