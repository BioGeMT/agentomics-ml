# DATASETS=("human_nontata_promoters" "human_enhancers_cohn" "drosophila_enhancers_stark" "human_enhancers_ensembl"  "AGO2_CLASH_Hejret2023" "human_ocr_ensembl")
DATASETS=("human_nontata_promoters")

#TODO config needs to be specific for o-series of models https://docs.deepwisdom.ai/main/en/guide/get_started/configuration/llm_api_configuration.html
# MODELS=("anthropic/claude-3.7-sonnet" "openai/gpt-4.1-2025-04-14" "openai/o4-mini" "google/gemini-2.5-pro-preview-03-25" "qwen/qwen3-32b" "deepseek/deepseek-r1" "deepseek/deepseek-chat")
MODELS=("openai/gpt-4.1-2025-04-14")

TAGS=("testing")
RUNS=1

# Export the API keys
set -a
source /repository/.env
set +a

for DATASET in "${DATASETS[@]}"
do
  echo "Current dataset: $DATASET"
  
  for MODEL in "${MODELS[@]}"
  do
    echo "Running with model: $MODEL"
    
    for ((i=1; i<=$RUNS; i++))
    do
      # run the python script create_user.py to get the agent id
      AGENT_ID=$(python /repository/src/utils/create_user.py)
      AGENT_DIR=/workspace/runs/"$AGENT_ID"

      echo "Running with agent ID: $AGENT_ID"
        
      AGENT_ENV="$AGENT_DIR"/"$AGENT_ID"_env

      conda create -p "$AGENT_ENV" python=3.9 -y
      source /opt/conda/etc/profile.d/conda.sh
      conda activate "$AGENT_ENV"
      pip install --upgrade metagpt #use --no-cache-dir in case of problems
      pip install agentops==0.4.9 #Fix metagpt env error

      pip install wandb
      pip install python-dotenv
      pip install pyyaml
      pip install hrid==0.2.4
      pip install pandas
      pip install scikit-learn
      
      mkdir "$AGENT_DIR"/DI
      cd "$AGENT_DIR"/DI
      metagpt --init-config

      cp /repository/src/competitors/DI/run.py "$AGENT_DIR"/DI/run.py
      cp /repository/src/competitors/DI/set_config.py "$AGENT_DIR"/DI/set_config.py

      # Give the run permission to write to the DI directory and env
      chmod -R 777 "$AGENT_DIR"/DI
      chmod -R 777 "$AGENT_ENV"

      python set_config.py --config-path "$AGENT_DIR"/.metagpt/config2.yaml --api-type 'openrouter' --model "$MODEL" --base-url 'https://openrouter.ai/api/v1' --api-key "$OPENROUTER_API_KEY"

      echo "Launching the python run script"
      # Run the main script as the generated user
      sudo -u $AGENT_ID bash -c "source /opt/conda/etc/profile.d/conda.sh && conda activate $AGENT_ENV && python run.py \
        --dataset $DATASET \
        --model $MODEL \
        --tags ${TAGS[@]} \
        --run_id $AGENT_ID"

      # Capture exit status of the Python script
      STATUS=$?
      echo "Exit status: $STATUS"
    done
  done
done