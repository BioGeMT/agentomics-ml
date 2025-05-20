## Setup to run AIDE

1. Create the conda environment using the provided YAML file:

```bash
conda env create -f environment.yaml
```

2. Activate the environment:

```bash
conda activate aideml-env
```

3. Create a `.env` file in the repository root 

- Add a basic openrouter key:
```
OPENROUTER_API_KEY=your_openrouter_api_key
```
- Add your `WANDB_API_KEY` and add your entity to `src/run_logging/wandb.py` file



4. Execute the workflow:
   ```bash
   bash run.sh
   ```

## File Structure

- `aide_prompt.txt` - Contains instructions for the AIDE agent
- `run.py` - Python script that executes the AIDE agent
- `run.sh` - Shell script that orchestrates the workflow

## Outputs
Resulting files are located in `/workflow`