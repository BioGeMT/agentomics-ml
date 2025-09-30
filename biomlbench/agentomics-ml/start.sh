#!/bin/bash
set -euo pipefail
#TODO update to main branch before merging
git clone -b biomlbench-implementation https://github.com/BioGeMT/agentomics-ml.git /home/agentomics-ml

cp -r /home/agentomics-ml/src /home/agent/src

echo -e "\033[0;31mStarting Agentomics-ML...\033[0m"
/opt/conda/envs/agentomics-env/bin/python /home/agent/src/run_agent_biomlbench.py \
    --model openai/gpt-5-nano \
    --val-metric MSE \
    --iterations 1 \
    --target-col Y \
    --task-type regression