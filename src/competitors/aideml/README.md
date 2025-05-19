## Setup and Configuration

### Environment Setup

1. Create the conda environment using the provided YAML file:

```bash
conda env create -f environment.yaml
```

2. Activate the environment:

```bash
conda activate aideml-env
```

### Configuration

All configurable aspects are listed below:

1. **Environment Variables**:
   - Create a `.env` file in the repository root with:
     ```
     OPENROUTER_API_KEY=your_openrouter_api_key
     ```

2. **Shell Script Variables** (`run.sh`):
   - `DATASETS`: List of datasets to process
   - `MODELS`: List of models to use (e.g., "openai/gpt-4.1-2025-04-14")
   - `RUNS`: Number of runs to execute
   - Repository datasets will be automatically accessed as specified in the configuration


3. **Agent Instructions** (`aide_prompt.txt`):
   - Modify this file to customize the instructions given to the AIDE agent
   - Control the type of classifier the agent will build
   - Specify requirements, constraints, and objectives

## File Structure

- `aide_prompt.txt` - Contains instructions for the AIDE agent (customizable)
- `run.py` - Python script that executes the AIDE agent
- `run.sh` - Shell script that orchestrates the workflow (customizable)

## Running the Agent

1. Activate the environment if not already activated:
   ```bash
   conda activate aideml-env
   ```

2. Execute with a single command:
   ```bash
   bash run.sh
   ```
