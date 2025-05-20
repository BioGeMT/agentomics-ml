## Setup to run Data Interpreter

1. Create the conda environment using the provided YAML file:

```bash
conda env create -f environment.yaml
```

2. Activate the environment:

```bash
conda activate DI-env
```


3. Create a `.env` file in the repository root 

- Add a basic openrouter key:
```
OPENROUTER_API_KEY=your_openrouter_api_key
```
- Add your `WANDB_API_KEY` and add your entity to `src/run_logging/wandb.py` file

- If using a proxy, also add:
```
HTTP_PROXY=http://your-proxy:port
HTTPS_PROXY=http://your-proxy:port
http_proxy=http://your-proxy:port
https_proxy=http://your-proxy:port
```

4. Execute the workflow:
```bash
bash run.sh
```


## File Structure

- `run.py` - Python script that executes the Data Interpreter agent
- `run.sh` - Shell script that sets up and orchestrates the workflow (customizable)
- `set_config.py` - Helper script to set the DI configuration
- Repository datasets will be automatically accessed as specified in the configuration

## Outputs
Resulting files are located in `/workflow`