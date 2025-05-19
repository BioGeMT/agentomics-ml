## Setup and Configuration

### Environment Setup

1. Create the conda environment using the provided YAML file:

```bash
conda env create -f environment.yaml
```

2. Activate the environment:

```bash
conda activate oneshot-env
```

### Configuration

All configurable aspects are listed below:

1. **Environment Variables**:
   - Create a `.env` file in the repository root with:
     ```
     OPENROUTER_API_KEY=your_openrouter_api_key
     ```

2. **Shell Script Variables** (`run_llm_agent.sh`):
   - `DATASETS`: List of datasets to process (e.g., "human_nontata_promoters", "human_enhancers_cohn")
   - `MODELS`: List of models to use via OpenRouter (e.g., "anthropic/claude-3.7-sonnet", "openai/gpt-4.1-2025-04-14")
   - `TEMP`: Temperature setting for LLM generation
   - `TAGS`: Tags for experiment tracking
   - `RUNS`: Number of runs to execute for each model-dataset combination
   - `TIME_BUDGET_IN_HOURS`: Timeout in hours for each run

3. **Code Generation Prompt**:
   - The prompt template in `1-shot_llm_run.py` can be modified to customize instructions for the LLM
   - You can adjust requirements and specific instructions for how the model should be trained and used
        ```python
            }
            prompt = f"""
                Create the best possible classifier that will generalize to new unseen data.
                ..."""
     ```
     
## File Structure

- `1-shot_llm_run.py` - Main Python script that handles LLM code generation and execution
- `run_llm_agent.sh` - Shell script that orchestrates the workflow (customizable)
- `environment.yaml` - Conda environment specification

## Running the Agent

1. Activate the environment if not already activated:
   ```bash
   conda activate oneshot-env
   ```

2. Execute with a single command:
   ```bash
   bash run_llm_agent.sh
   ```

