Guide to reproduce Agentomics ISMB 2026 results:

1) use machine with a GPU and installed [conda](https://docs.conda.io/projects/conda/en/latest/user-guide/install/index.html)
2) create an [Openrouter](https://openrouter.ai/) provisioning key + [Weights & Biases](https://wandb.ai/) keys to log results
3) install and launch [Docker](https://www.docker.com/get-started/)
4) Run these bash commands, replacing the API key variables with your keys

```bash
mkdir AgentomicsISMB
cd AgentomicsISMB
git clone https://github.com/BioGeMT/agentomics-ml.git -b ismb_submission
cd agentomics-ml
cat > .env <<'EOF'
PROVISIONING_OPENROUTER_API_KEY=your_provisioning_openrouter_key
WANDB_API_KEY=your_wandb_key
WANDB_PROJECT_NAME=your_wandb_project_name
WANDB_ENTITY=your_wandb_entity
EOF
./orchestrator.sh
```
