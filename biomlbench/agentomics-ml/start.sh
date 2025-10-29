#!/bin/bash
set -euo pipefail
git clone https://github.com/BioGeMT/agentomics-ml.git /home/agentomics-ml

cp -r /home/agentomics-ml/src /home/agent/src

format_time() {
  local time_in_sec=$1
  local hours=$((time_in_sec / 3600))
  local minutes=$(((time_in_sec % 3600) / 60))
  local seconds=$((time_in_sec % 60))
  echo "${hours}hrs ${minutes}mins ${seconds}secs"
}
export TIME_LIMIT=$(format_time $TIME_LIMIT_SECS)

echo -e "\033[0;31mStarting Agentomics-ML...\033[0m"
timeout $TIME_LIMIT_SECS /opt/conda/envs/agentomics-env/bin/python /home/agent/src/run_agent_biomlbench.py \
    --model $MODEL \
    --val-metric $VAL_METRIC \
    --iterations $ITERATIONS \
    --target-col $TARGET_COL \
    --task-type $TASK_TYPE \
    --user-prompt "$USER_PROMPT" \
    --split-allowed-iterations $SPLIT_ALLOWED_ITERATIONS

if [ $? -eq 124 ]; then
  echo "Timed out after $TIME_LIMIT"
fi