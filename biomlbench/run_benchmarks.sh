#!/usr/bin/env bash
REPOS_DIR=/home/vmart01/repos

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

#TODO pull the fresh repo and make changes to gpu config etc..
# Copy the agentomics 'biomlbench agent' folder into the right place in biomlbench repo
cd "$REPOS_DIR"/biomlbench
source .venv/bin/activate
mkdir -p "$REPOS_DIR"/biomlbench/agents/agentomics-ml
cp -r "$REPOS_DIR"/agentomics-ml/biomlbench/agentomics-ml "$REPOS_DIR"/biomlbench/agents/
#TODO wtf cd -r "$REPOS_DIR"/agentomics-ml/foundation_models "$REPOS_DIR"/biomlbench/agents/agentomics-ml

# Build and run the agent 
./scripts/build_agent.sh agentomics-ml
echo RUNNING AGENT
biomlbench run-agent --agent agentomics-ml --task-id polarishub/tdcommons-caco2-wang
echo DONE

# Optional cleanup
cleanup