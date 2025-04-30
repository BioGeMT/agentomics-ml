#!/bin/bash

# Default parameters
DATASET="human_non_tata_promoters"
PROVIDER="openai"
MODEL="gpt-4o-2024-08-06"
TEMP=0.7
MAX_TOKENS=4000
BASE_PATH="repository"
OUTPUT_DIR="/workspace/runs"
TRAIN_CSV="/repository/datasets/human_non_tata_promoters/human_nontata_promoters_train.csv"
TEST_CSV="/repository/datasets/human_non_tata_promoters/human_nontata_promoters_test.no_label.csv"
RUNS=5

echo "Starting $RUNS sequential runs..."

for ((i=1; i<=$RUNS; i++))
do
  echo ""
  echo "=========================================="
  echo "STARTING RUN $i of $RUNS"
  echo "=========================================="
  echo ""

  # Execute the Python script with all parameters
  python 1-shot_llm_agent_gpt4o-claude.py \
    --dataset "$DATASET" \
    --provider "$PROVIDER" \
    --model "$MODEL" \
    --temp "$TEMP" \
    --max-tokens "$MAX_TOKENS" \
    --base-path "$BASE_PATH" \
    --output-dir "$OUTPUT_DIR" \
    --train-csv "$TRAIN_CSV" \
    --test-csv "$TEST_CSV"

  # Capture exit status of the Python script
  STATUS=$?

  echo ""
  echo "=========================================="
  echo "COMPLETED RUN $i of $RUNS (Status: $STATUS)"
  echo "=========================================="
  echo ""

done

echo "All $RUNS runs completed."