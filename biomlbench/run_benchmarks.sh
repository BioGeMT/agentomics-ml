#!/usr/bin/env bash

while [[ $# -gt 0 ]]; do
  case $1 in
    --repos-dir)
      REPOS_DIR="$2"
      shift 2
      ;;
    --spend-limit)
      SPEND_LIMIT="$2"
      shift 2
      ;;
    --dset)
      DSET="$2"
      shift 2
      ;;
    --split-allowed-iterations)
      SPLIT_ALLOWED_ITERATIONS="$2"
      shift 2
      ;;
    --timeout)
      TIME_LIMIT_SECS="$2"
      shift 2
      ;;
    --split-timeout)
      SPLIT_TIME_LIMIT_SECS="$2"
      shift 2
      ;;
    --foundation-model-type)
      FOUNDATION_MODEL_TYPE="$2"
      shift 2
      ;;
    --exploration-iterations)
      EXPLORATION_ITERATIONS="$2"
      shift 2
      ;;
    --iterations)
      ITERATIONS="$2"
      shift 2
      ;;
    --model)
      MODEL="$2"
      shift 2
      ;;
    --user-prompt)
      USER_PROMPT="$2"
      shift 2
      ;;
    --pull-branch)
      PULL_BRANCH="$2"
      shift 2
      ;;
    --tags)
      shift
      while [[ $# -gt 0 && "$1" != -* ]]; do
        TAGS+=("$1")
        shift
      done
      ;;
    *)
      shift
      ;;
  esac
done

update_config() {
  set -a
  source "$REPOS_DIR/agentomics-ml/.env" 2>/dev/null
  set +a

  local cfg="$REPOS_DIR/agentomics-ml/biomlbench/agentomics-ml/config.yaml"
  local escaped_prompt=$(printf '%s\n' "$USER_PROMPT" | sed -e 's/[\/&]/\\&/g')
  local tags="${TAGS[*]}"

  for var in WANDB_API_KEY WANDB_PROJECT_NAME WANDB_ENTITY SPLIT_ALLOWED_ITERATIONS ITERATIONS EXPLORATION_ITERATIONS MODEL PULL_BRANCH TIME_LIMIT_SECS SPLIT_TIME_LIMIT_SECS FOUNDATION_MODEL_TYPE; do
    local val="${!var}"
    if grep -q "^    $var:" "$cfg"; then
      sed -i "s|^    $var:.*|    $var: $val|" "$cfg"
    else
      sed -i "/env_vars:/a\    $var: $val" "$cfg"
    fi
  done

  if grep -q "^    USER_PROMPT:" "$cfg"; then
    sed -i "s|^    USER_PROMPT:.*|    USER_PROMPT: \"$escaped_prompt\"|" "$cfg"
  else
    sed -i "/env_vars:/a\    USER_PROMPT: \"$escaped_prompt\"" "$cfg"
  fi

  if grep -q "^    TAGS:" "$cfg"; then
    sed -i "s|^    TAGS:.*|    TAGS: $tags|" "$cfg"
  else
    sed -i "/env_vars:/a\    TAGS: $tags" "$cfg"
  fi
}

ensure_agentomics_env() {
  if ! conda env list | grep -q "^agentomics-env "; then
    echo "Creating agentomics-env conda environment"
    conda env create -f environment.yaml -q
  fi
}

update_config
ensure_agentomics_env

# Create API key
API_KEY_OUTPUT=$(cd "$REPOS_DIR/agentomics-ml" && PYTHONPATH="$REPOS_DIR/agentomics-ml/src" conda run -n agentomics-env python src/utils/api_keys_utils.py create --name "agentomics_run_$(date +%s)" --limit "$SPEND_LIMIT")
API_KEY=$(echo "$API_KEY_OUTPUT" | cut -d',' -f1)
API_KEY_HASH=$(echo "$API_KEY_OUTPUT" | cut -d',' -f2)
echo "Created API key with the following spend limit: $SPEND_LIMIT"
# Update config.yaml with the new API key
if grep -q "^    OPENROUTER_API_KEY:" "$REPOS_DIR"/agentomics-ml/biomlbench/agentomics-ml/config.yaml; then
  sed -i "s|^    OPENROUTER_API_KEY:.*|    OPENROUTER_API_KEY: $API_KEY|" "$REPOS_DIR"/agentomics-ml/biomlbench/agentomics-ml/config.yaml
else
  sed -i "/env_vars:/a\    OPENROUTER_API_KEY: $API_KEY" "$REPOS_DIR"/agentomics-ml/biomlbench/agentomics-ml/config.yaml
fi

setup_biomlbench_repo() {
  if [ ! -d "$REPOS_DIR/biomlbench" ] ; then
    #TODO freeze repo version
    git clone https://github.com/science-machine/biomlbench.git "$REPOS_DIR/biomlbench"
  fi
  cd "$REPOS_DIR/biomlbench"
  if ! conda env list | grep -q "^biomlbench_conda_env "; then
    conda create -n biomlbench_conda_env python=3.11 uv -c conda-forge -y
  fi
  source activate biomlbench_conda_env
  uv sync
  source .venv/bin/activate

  # Fix polaris data download (only if not already patched)
  if ! grep -q "_patch_fsspec_for_proxy" "$REPOS_DIR/biomlbench/biomlbench/data_sources/polaris.py"; then
    sed -i "20 r $REPOS_DIR/agentomics-ml/biomlbench/proxyfix.py" "$REPOS_DIR/biomlbench/biomlbench/data_sources/polaris.py"
  fi

  # ./scripts/build_base_env.sh
  ./scripts/pull_prebuilt_images.sh
  biomlbench prepare -t $DSET

  # Update container config: set gpus to 1 and nano_cpus to 48000000000
  sed -i 's/"nano_cpus": 12000000000/"nano_cpus": 48000000000/' "$REPOS_DIR/biomlbench/environment/config/container_configs/default.json"
  sed -i 's/"gpus": 0/"gpus": 1/' "$REPOS_DIR/biomlbench/environment/config/container_configs/default.json"
}

setup_biomlbench_repo

cleanup() {
  rm -rf "$REPOS_DIR"/biomlbench/agents/agentomics-ml
}

# Optional cleanup
cleanup

# Setup files that need to be in the agentomics 'biomlbench agent' folder
setup_support_files() {
  cp -r "$REPOS_DIR"/agentomics-ml/foundation_models "$REPOS_DIR"/biomlbench/agents/agentomics-ml
  cp "$REPOS_DIR"/agentomics-ml/environment.yaml "$REPOS_DIR"/biomlbench/agents/agentomics-ml/environment.yaml
  cp "$REPOS_DIR"/agentomics-ml/environment_agent.yaml "$REPOS_DIR"/biomlbench/agents/agentomics-ml/environment_agent.yaml
  cp "$REPOS_DIR"/agentomics-ml/src/utils/foundation_models_utils.py "$REPOS_DIR"/biomlbench/agents/agentomics-ml/foundation_models_utils.py
  cp "$REPOS_DIR"/agentomics-ml/src/utils/download_foundation_models.py "$REPOS_DIR"/biomlbench/agents/agentomics-ml/download_foundation_models.py
}

cd "$REPOS_DIR"/biomlbench
source .venv/bin/activate
mkdir -p "$REPOS_DIR"/biomlbench/agents/agentomics-ml
cp -r "$REPOS_DIR"/agentomics-ml/biomlbench/agentomics-ml "$REPOS_DIR"/biomlbench/agents/
setup_support_files

# Build and run the agent
./scripts/build_agent.sh agentomics-ml
echo RUNNING AGENT
OUTPUT=$(biomlbench run-agent --agent agentomics-ml --task-id "$DSET" 2>&1 | tee /dev/tty)

RESULTS_DIR=$(echo "$OUTPUT" | grep -oP "Results saved to: \K.*" | head -1)
submission_path=$(jq -r '."submission_path"' "$RESULTS_DIR/submission.jsonl")
task_id=$(jq -r '."task_id"' "$RESULTS_DIR/submission.jsonl")
code_path=$(jq -r '."code_path"' "$RESULTS_DIR/submission.jsonl")

GRADE=$(biomlbench grade-sample "$submission_path" "$task_id" 2>&1 | tee /dev/tty)
GRADE_JSON=$(echo "$GRADE" | perl -0777 -nle 'print $1 if /({.*?})/s')

deactivate

if ! conda env list | grep -q "^agentomics-env "; then
  conda env create -f "$REPOS_DIR/agentomics-ml/environment.yaml"
fi

PROJECT_ROOT="$REPOS_DIR/agentomics-ml/src"
export PYTHONPATH="$PROJECT_ROOT:$PYTHONPATH"
cd "$REPOS_DIR"/agentomics-ml
conda run -n agentomics-env python src/run_logging/biomlbench_test_eval.py --results-dir=$RESULTS_DIR --grade-json "$GRADE_JSON"

# Get config path and log API usage, then delete key
CONFIG_PATH="$code_path/extras/config.json"
cd "$REPOS_DIR/agentomics-ml" && PYTHONPATH="$REPOS_DIR/agentomics-ml/src" conda run -n agentomics-env python src/utils/api_keys_utils.py cleanup-and-log --config-path "$CONFIG_PATH" --api-key-hash "$API_KEY_HASH"

# Optional removal of conda (uses a lot of storage)
# cd "$REPOS_DIR/biomlbench"
# find . -type d -name ".conda" -exec rm -rf {} +

./compute_stealth_test.sh --exp-folder "$code_path" --agentomics-dir "$REPOS_DIR/agentomics-ml"

echo DONE
# Optional cleanup
cleanup