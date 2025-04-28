#!/bin/bash

# Default parameters
DATASET="human_non_tata_promoters"
PROVIDER="openai"
MODEL="gpt-4o-2024-08-06"
TEMP=0.7
MAX_TOKENS=4000
BASE_PATH="/home/user/Documents/Agentomics-ML"
OUTPUT_DIR="/home/user/Documents/Agentomics-ML/outputs"
RUN_NAME="ed_run"
MAX_ATTEMPTS=5
TRAIN_CSV="/home/user/Documents/Agentomics-ML/datasets/human_non_tata_promoters/human_nontata_promoters_train.csv"
TEST_CSV="/home/user/Documents/Agentomics-ML/datasets/human_non_tata_promoters/human_nontata_promoters_test.no_label.csv"

# Allow overriding defaults through command-line arguments
while [[ $# -gt 0 ]]; do
  case "$1" in
    --dataset)
      DATASET="$2"
      shift 2
      ;;
    --provider)
      PROVIDER="$2"
      shift 2
      ;;
    --model)
      MODEL="$2"
      shift 2
      ;;
    --temp)
      TEMP="$2"
      shift 2
      ;;
    --max-tokens)
      MAX_TOKENS="$2"
      shift 2
      ;;
    --base-path)
      BASE_PATH="$2"
      shift 2
      ;;
    --output-dir)
      OUTPUT_DIR="$2"
      shift 2
      ;;
    --run-name)
      RUN_NAME="$2"
      shift 2
      ;;
    --max-attempts)
      MAX_ATTEMPTS="$2"
      shift 2
      ;;
    --train-csv)
      TRAIN_CSV="$2"
      shift 2
      ;;
    --test-csv)
      TEST_CSV="$2"
      shift 2
      ;;
    *)
      echo "Unknown option: $1"
      exit 1
      ;;
  esac
done

# Execute the Python script with all parameters
python 1-shot_llm_agent_gpt4o-claude.py \
  --dataset "$DATASET" \
  --provider "$PROVIDER" \
  --model "$MODEL" \
  --temp "$TEMP" \
  --max-tokens "$MAX_TOKENS" \
  --base-path "$BASE_PATH" \
  --output-dir "$OUTPUT_DIR" \
  --run-name "$RUN_NAME" \
  --max-attempts "$MAX_ATTEMPTS" \
  --train-csv "$TRAIN_CSV" \
  --test-csv "$TEST_CSV"