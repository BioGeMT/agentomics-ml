#!/usr/bin/env bash
set -euo pipefail
DATASETS=("human_nontata_promoters" "human_enhancers_cohn" "drosophila_enhancers_stark" "human_enhancers_ensembl"  "AGO2_CLASH_Hejret2023" "human_ocr_ensembl")
MODELS=("openai/gpt-4.1-2025-04-14")
TAGS=("testing")
RUNS=5

ENV_NAME="aideml-env"     # pre-created conda environment

# Load .env
set -a
source /repository/.env
set +a

export OPENAI_BASE_URL='https://openrouter.ai/api/v1'
export OPENAI_API_KEY="$OPENROUTER_API_KEY"

for DATASET in "${DATASETS[@]}"; do
  echo "Dataset: $DATASET"

  for MODEL in "${MODELS[@]}"; do
    echo "  Model: $MODEL"

    for ((i = 1; i <= RUNS; i++)); do
      AGENT_ID=$(python /repository/src/utils/create_user.py)
      AGENT_DIR="/workspace/runs/$AGENT_ID"
      echo "    Run $i  –  AGENT_ID: $AGENT_ID"

      # Create necessary directories
      mkdir -p "$AGENT_DIR"/AIDE/input

      # Copy files
      cp /repository/src/competitors/aideml/run.py              "$AGENT_DIR"/AIDE/
      cp /repository/src/competitors/aideml/aide_prompt.txt     "$AGENT_DIR"/AIDE/
      cp /repository/datasets/"$DATASET"/train.csv              "$AGENT_DIR"/AIDE/input/
      cp /repository/datasets/"$DATASET"/dataset_description.md "$AGENT_DIR"/AIDE/input/

      chmod -R 777 "$AGENT_DIR"/AIDE

      cd "$AGENT_DIR"/AIDE 

      # Run Python script in env
      bash -c "
        export OPENAI_API_KEY=\"$OPENAI_API_KEY\"
        export OPENAI_BASE_URL=\"$OPENAI_BASE_URL\"
        export WANDB_API_KEY=\"$WANDB_API_KEY\"
        source \"$(conda info --base)/etc/profile.d/conda.sh\"
        conda activate $ENV_NAME
        python \"$AGENT_DIR/AIDE/run.py\" \
          --dataset \"$DATASET\" \
          --model \"$MODEL\" \
          --tags ${TAGS[*]} \
          --run_id \"$AGENT_ID\" \
          --eval \"Use accuracy\"
      "

      echo "    Exit status: $?"
    done
  done
done
