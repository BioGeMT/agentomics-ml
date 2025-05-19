## Setup and Configuration

### Environment Setup

1. Create the conda environment using the provided YAML file:

```bash
conda env create -f enviroment.yaml
```

2. Activate the environment:

```bash
conda activate DI-env
```

### Configuration

All configurable aspects are listed below:

1. **Environment Variables**:
   - Create a `.env` file in the repository root with:
     ```
     OPENROUTER_API_KEY=your_openrouter_api_key
     
     # If using a proxy, also add:
     HTTP_PROXY=http://your-proxy:port
     HTTPS_PROXY=http://your-proxy:port
     http_proxy=http://your-proxy:port
     https_proxy=http://your-proxy:port
     ```

2. **Shell Script Variables** (`run.sh`):
   - `DATASETS`: List of datasets to process
   - `MODELS`: List of models to use (e.g., "openai/gpt-4.1-2025-04-14")
   - `RUNS`: Number of runs to execute
   - `PER_RUN_CREDIT_BUDGET`: Credit budget for each run
   - `TIME_BUDGET_IN_HOURS`: Timeout in hours for each run

3. **Agent Prompt** (`run.py`):
   - The prompt template in `run.py` can be modified to customize instructions for the Data Interpreter
   - You can adjust requirements, file paths, and specific instructions for how the model should be trained and used
   - The prompt section is located in the `main()` function and starts with:
     ```python
     prompt = f"""
         Create the best possible classifier that will generalize to new unseen data.
         ...
     """
     ```

## File Structure

- `run.py` - Python script that executes the Data Interpreter agent
- `run.sh` - Shell script that sets up and orchestrates the workflow (customizable)
- `set_config.py` - Helper script to set the DI configuration
- Repository datasets will be automatically accessed as specified in the configuration

## Running the Agent

1. Activate the environment if not already activated:
   ```bash
   conda activate DI-env
   ```

2. Execute with a single command:
   ```bash
   bash run.sh
   ```
