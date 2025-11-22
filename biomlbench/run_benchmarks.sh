#!/usr/bin/env bash

# Needs config in biomlbench/agentomics-ml (with wandb keys) and normal .env in agenotmics (with wandb and openrouter provisioning) + needs agentomics env created already
SPEND_LIMIT=10
REPOS_DIR="/home/$USER/repos" #TODO needs to be configured
DSET=proteingym-dms/SPIKE_SARS2_Starr_2020_binding

# DSET=proteingym-dms/SPA_STAAU_Tsuboyama_2023_1LP1
# DSET=polarishub/tdcommons-caco2-wang
#TODO generalize copying, repos

# Drug discovery (polarishub/)
# polaris-pkis2-egfr-wt-c-1 **pr_auc** CLF (targetcol CLASS_EGFR) README-OK
# polaris-adme-fang-hclint-1 **pearsonr** REG README-OK
# polaris-adme-fang-hppb-1 **pearsonr** REG README-OK
# polaris-adme-fang-solu-1 **pearsonr** REG README-OK
# tdcommons-cyp2d6-substrate-carbonmangels **pr_auc** CLF README-OK
# tdcommons-lipophilicity-astrazeneca mean_absolute_error REG README-OK
# tdcommons-herg roc_auc CLF README-OK
# tdcommons-bbb-martins roc_auc CLF README-OK
# tdcommons-caco2-wang mean_absolute_error REG README-OK

# Protein engineering (proteingym-dms/)
# SPIKE_SARS2_Starr_2020_binding REG
# SPA_STAAU_Tsuboyama_2023
# PSAE_PICP2_indels
# CBX4_HUMAN_multi-sub
# Q8EG35_SHEON_indels
# CSN4_MOUSE_indels

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

  # Update container config: set gpus to 1 and nano_cpus to 32000000000
  sed -i 's/"nano_cpus": 12000000000/"nano_cpus": 32000000000/' "$REPOS_DIR/biomlbench/environment/config/container_configs/default.json"
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
CONFIG_PATH=$(find "$RESULTS_DIR" -name "config.json" -type f 2>/dev/null | head -1)
cd "$REPOS_DIR/agentomics-ml" && PYTHONPATH="$REPOS_DIR/agentomics-ml/src" conda run -n agentomics-env python src/utils/api_keys_utils.py cleanup-and-log --config-path "$CONFIG_PATH" --api-key-hash "$API_KEY_HASH"

# Optional removal of conda (uses a lot of storage)
cd "$REPOS_DIR/biomlbench"
find . -type d -name ".conda" -exec rm -rf {} +
echo DONE
# Optional cleanup
cleanup