#!/usr/bin/env bash
REPOS_DIR="/home/$USER/repos"
DSET=polarishub/tdcommons-caco2-wang

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
  rm -rf "$REPOS_DIR"/agentomics-ml/biomlbench/agentomics-ml/foundation_models
  rm "$REPOS_DIR"/agentomics-ml/biomlbench/agentomics-ml/environment.yaml
  rm "$REPOS_DIR"/agentomics-ml/biomlbench/agentomics-ml/environment_agent.yaml
  rm "$REPOS_DIR"/agentomics-ml/biomlbench/agentomics-ml/foundation_models_utils.py
  rm "$REPOS_DIR"/agentomics-ml/biomlbench/agentomics-ml/download_foundation_models.py
}

# Optional cleanup
cleanup

# Setup files that need to be in the agentomics 'biomlbench agent' folder
cp -r "$REPOS_DIR"/agentomics-ml/foundation_models "$REPOS_DIR"/agentomics-ml/biomlbench/agentomics-ml/
cp "$REPOS_DIR"/agentomics-ml/environment.yaml "$REPOS_DIR"/agentomics-ml/biomlbench/agentomics-ml/environment.yaml
cp "$REPOS_DIR"/agentomics-ml/environment_agent.yaml "$REPOS_DIR"/agentomics-ml/biomlbench/agentomics-ml/environment_agent.yaml
cp "$REPOS_DIR"/agentomics-ml/src/utils/foundation_models_utils.py "$REPOS_DIR"/agentomics-ml/biomlbench/agentomics-ml/foundation_models_utils.py
cp "$REPOS_DIR"/agentomics-ml/src/utils/download_foundation_models.py "$REPOS_DIR"/agentomics-ml/biomlbench/agentomics-ml/download_foundation_models.py

cd "$REPOS_DIR"/biomlbench
source .venv/bin/activate
mkdir -p "$REPOS_DIR"/biomlbench/agents/agentomics-ml
cp -r "$REPOS_DIR"/agentomics-ml/biomlbench/agentomics-ml "$REPOS_DIR"/biomlbench/agents/

# Build and run the agent
./scripts/build_agent.sh agentomics-ml
echo RUNNING AGENT
OUTPUT=$(biomlbench run-agent --agent agentomics-ml --task-id $DSET 2>&1 | tee /dev/tty)

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

echo DONE

# Optional cleanup
cleanup