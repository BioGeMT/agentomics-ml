
#!/bin/bash
set -e # Exit immediately if a command exits with a non-zero status.

echo_green() { echo -e "\033[0;32m$1\033[0m"; }
echo_yellow() { echo -e "\033[1;33m$1\033[0m"; }
echo_red() { echo -e "\033[0;31m$1\033[0m"; }

# --- 1. Activate Conda Environment and Set Working Directory ---
echo_green "Step 1: Activating SELA conda environment and setting working directory..."
source /opt/conda/etc/profile.d/conda.sh
conda activate /tmp/sela_env
if [ $? -ne 0 ]; then echo_red "Failed to activate conda environment /tmp/sela_env. Exiting."; exit 1; fi
echo_green "Conda environment /tmp/sela_env activated."

cd /tmp/sela
if [ $? -ne 0 ]; then echo_red "Failed to change directory to /tmp/sela. Exiting."; exit 1; fi
echo_green "Current working directory: $(pwd)" # Should be /tmp/sela

# --- 2. Configuration ---
DATASETS=("human_nontata_promoters")
MODELS=("openai/gpt-4.1-2025-04-14")
TAGS=("sela_final_test_v5" "promoters" "default_config_workaround") # Updated tags
RUNS=1

# --- 3. Export API Keys from .env file ---
echo_green "Step 2: Exporting API keys..."
ENV_FILE_PATH="/repository/Agentomics-ML/.env" # Verify this path
if [ -f "$ENV_FILE_PATH" ]; then
    echo "Sourcing API keys from $ENV_FILE_PATH"
    set -a; source "$ENV_FILE_PATH"; set +a
else echo_yellow "Warning: Environment file $ENV_FILE_PATH not found."; fi

# Check for essential API Key
if [ -z "$OPENROUTER_API_KEY" ]; then echo_red "Error: OPENROUTER_API_KEY is not set."; exit 1; fi
echo_green "OPENROUTER_API_KEY seems to be set."
if [ -z "$WANDB_API_KEY" ]; then echo_yellow "Warning: WANDB_API_KEY is not set."; fi

# --- 4. Main Execution Loop ---
echo_green "Step 3: Starting main execution loop..."
for DATASET in "${DATASETS[@]}"; do
  echo_green "-----------------------------------------------------"
  echo_green "Processing Dataset: $DATASET"
  echo_green "-----------------------------------------------------"

  for MODEL in "${MODELS[@]}"; do
    echo_yellow "  Running with Model: $MODEL"
    if [ -z "$MODEL" ]; then echo_red "Error: MODEL variable empty."; exit 1; fi

    for ((i=1; i<=$RUNS; i++)); do
      echo_yellow "    Run iteration: $i / $RUNS"

      # Generate Agent ID
      CREATE_USER_SCRIPT_PATH="/repository/Agentomics-ML/src/utils/create_user.py"
      AGENT_ID_PREFIX="sela_run"
      if [ -f "$CREATE_USER_SCRIPT_PATH" ]; then AGENT_ID=$(python "$CREATE_USER_SCRIPT_PATH"); if [ -z "$AGENT_ID" ]; then echo_red "create_user.py failed."; AGENT_ID="${AGENT_ID_PREFIX}_${DATASET}_${MODEL//\//_}_$(date +%s%N)_run${i}"; fi
      else echo_yellow "Warn: create_user.py not found."; AGENT_ID="${AGENT_ID_PREFIX}_${DATASET}_${MODEL//\//_}_$(date +%s%N)_run${i}"; fi
      echo "    Generated AGENT_ID (run_id): $AGENT_ID"

      AGENT_RUN_DIR="/workspace/runs/$AGENT_ID"; echo "    Agent run output directory: $AGENT_RUN_DIR"
      mkdir -p /root/.metagpt/ # Ensure default config dir exists (for workaround)

      echo "    Configuring MetaGPT (WORKAROUND: writing to /root/.metagpt/config2.yaml)..."
      python set_config.py --config-path "/root/.metagpt/config2.yaml" --api-type 'openrouter' --model "$MODEL" --base-url 'https://openrouter.ai/api/v1' --api-key "$OPENROUTER_API_KEY"
      if [ $? -ne 0 ]; then echo_red "set_config.py failed."; continue; fi
      echo "    MetaGPT configured (using default path)."

      echo "    Launching SELA run.py script (in $(pwd))..."
      python run.py --dataset "$DATASET" --model "$MODEL" --tags "${TAGS[@]}" --run_id "$AGENT_ID" --rollouts "10"
      EXIT_STATUS=$?
      echo "    Exit status of run.py: $EXIT_STATUS"
      if [ $EXIT_STATUS -ne 0 ]; then echo_red "    run.py FAILED."; else echo_green "    run.py COMPLETED."; fi
      echo_yellow "    ------------------------------------"
    done # End of runs loop
  done # End of models loop
done # End of datasets loop

echo_green "====================================================="
echo_green "run.sh script finished all planned executions."
echo_green "====================================================="

