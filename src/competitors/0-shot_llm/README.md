## Setup to run 0-shots

1. Create a conda environment using the provided YAML file:

```bash
conda env create -f environment.yaml
```

2. Activate the environment:

```bash
conda activate oneshot-env
```

3. Create a `.env` file in the repository root 

- Add a basic openrouter key:
```
OPENROUTER_API_KEY=your_openrouter_api_key
```
- Add your `WANDB_API_KEY` and add your entity to `src/run_logging/wandb.py` file


4. Execute the workflow:
```bash
bash run_llm_agent.sh
```


## File Structure

- `1-shot_llm_run.py` - Main Python script that handles LLM code generation and execution
- `run_llm_agent.sh` - Shell script that orchestrates the workflow (customizable)
- `environment.yaml` - Conda environment specification

## Outputs
Resulting files are located in `/workflow`