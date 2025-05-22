# Agentomics-ML

Agentomics-ML is an autonomous agentic system for development of machine learning models for omics data.

Given a classification dataset, the system automatically generates:
- A trained model file, loaded by the inference script.
- An inference script, allowing immediate predictions on new data.

Agents run inside an isolated docker environment for security.

## Environment setup (Datasets + Docker + conda environment)

- Download datasets locally so they can be mounted into the container
```
conda create -n agentomics_temp_env --yes python && conda activate agentomics_temp_env && pip install genomic-benchmarks miRBench && python setup.py && conda deactivate && conda remove -n agentomics_temp_env --all --yes
```


- Make sure you have Docker installed
```
docker --version
```

- Create volume
```
docker volume create agents_volume
```

- Build docker image
```
docker build -t agents_img .
```

- Run the container
```
docker run -d \
    --name agents_cont \
    -v $(pwd):/repository:ro \
    -v agents_volume:/workspace/runs \
    agents_img
```

- Attach console to the container
```
docker exec -it agents_cont bash
```

Create a `.env` file containing your API key. You need to add a provisioning openrouter API key. Each run will create its own API key with a credit limit and delete it after the run is finished.

Example content: `PROVISIONING_OPENROUTER_API_KEY=my-api-key-1234`

### Optional - Wandb logging
If you want to visualize runs on WANDB (agent trace, metrics, ...) add your `WANDB_API_KEY` to the `.env` file and add your entity to `src/run_logging/wandb.py` file.


## Reproduce Agentomics runs
- Activate the default conda environment
```
source activate agentomics-env
```

- Run the full training script
`python src/agentomics_ml.py`

- Best models, files, and metrics will be saved in the `/snapshots` folder
- Train and validation set metric are reported with prefix (e.g. train/ACC)
- Test set metrics are reported without any prefix (e.g. ACC)

## Reproduce zeroshot/DI/AIDE runs

- Go into `src/competitors` and follow the corresponding README.md 

## Prompts
Can be found in `src/prompts/ALL_PROMPTS.md`

## Results and analysis
All single runs results can be found in `src/eval/FINAL_TABLES` 
- Agentomics runs: `Agentomics_runs.csv`
- AIDE runs: `AIDE_runs.csv`
- Data Interpreter runs: `DI_runs.csv`
- zero shot runs: `zeroshot_runs.csv`

All aggregated files and statistics can also be found in the `src/eval/FINAL_TABLES` folder.
The main results table (best out of 5 runs metrics) is the `max_metrics_and_sota_df.csv`

| Method                              | AGO2  | DE    | HEC   | HEE   | NTP   | OCRE  |
|-------------------------------------|-------|-------|-------|-------|-------|--------|
| Agentomics-ML: gpt-4.1 (no feedback)| 0.724 | 0.716 | 0.716 | 0.864 | 0.897 | 0.786  |
| Agentomics-ML: gpt4_1               | **0.778** | **0.736** | **0.743** | **0.885** | **0.925** | **0.816** |
| AIDE: gpt-4.1                        | N/A   | N/A   | N/A   | 0.920 | 0.758 | N/A    |
| DI: claude-3.7-sonnet               | N/A   | N/A   | 0.724 | N/A   | 0.839 | N/A    |
| DI: deepseek-chat                   | N/A   | N/A   | N/A   | 0.871 | N/A   | N/A    |
| DI: deepseek-r1                     | N/A   | N/A   | N/A   | N/A   | N/A   | N/A    |
| DI: llama-4-maverick                | N/A   | 0.500 | N/A   | 0.738 | 0.874 | N/A    |
| DI: gpt4.1                          | N/A   | N/A   | 0.500 | 0.752 | 0.873 | N/A    |
| DI: qwen3-32b                       | N/A   | N/A   | N/A   | N/A   | N/A   | N/A    |
| zero_shot: claude-3.7-sonnet        | N/A   | 0.650 | 0.728 | 0.864 | 0.901 | N/A    |
| zero_shot: deepseek-chat            | N/A   | 0.680 | 0.681 | 0.811 | 0.897 | 0.666  |
| zero_shot: deepseek-r1              | N/A   | 0.700 | 0.621 | 0.568 | 0.874 | 0.458  |
| zero_shot: llama-4-maverick         | N/A   | 0.661 | N/A   | N/A   | N/A   | 0.651  |
| zero_shot: gpt-4.1                  | N/A   | 0.703 | N/A   | N/A   | N/A   | N/A    |
| zero_shot: o4-mini                  | N/A   | 0.708 | 0.728 | 0.848 | 0.885 | 0.506  |
| zero_shot: qwen3-32b                | N/A   | N/A   | N/A   | 0.850 | N/A   | 0.662  |
| **Human SOTA**                      | **0.860** | **0.586** | **0.747** | **0.933** | **0.974** | **0.825** |


To re-run the analysis and re-generate results tables:
(Run all these command from outside of docker)
`cd` into `src/eval` 
```
conda env create -f environment.yaml
conda activate analysis-env
python analysis.py
```

## Proxy settings

If you are using a proxy, Docker will not automatically detect it and therefore every installation command will fail.

Create the systemd service directory if it doesn't exist and create or edit the proxy configuration file:
```
sudo mkdir -p /etc/systemd/system/docker.service.d
sudo nano /etc/systemd/system/docker.service.d/http-proxy.conf
```

Add the following lines:
```
[Service]
Environment="HTTP_PROXY=http://your-proxy:port"
Environment="HTTPS_PROXY=https://your-proxy:port"
Environment="NO_PROXY=localhost,127.0.0.1"
```

Reload the systemd configuration and restart Docker
```
sudo systemctl daemon-reload
sudo systemctl restart docker
```

Make sure to have at least one of the following environment variables with the proxy address:

* http_proxy
* https_proxy
* HTTP_PROXY 
* HTTPS_PROXY 

You can run the following commands to check the value of these variables and check if they have been defined:

```
env | grep -i "http_proxy"

env | grep -i "https_proxy"
```

Build the Docker image passing the proxy build arguments:
```
docker build \
  --build-arg HTTP_PROXY=$HTTP_PROXY \
  --build-arg HTTPS_PROXY=$HTTPS_PROXY \
  --build-arg http_proxy=$http_proxy \
  --build-arg https_proxy=$https_proxy \
  -t agents_img .
```

Run the container passing the proxy arguments:
```
docker run -d \
    --name agents_cont \
    -v "$(pwd)":/repository:ro \
    -v agents_volume:/workspace/runs \
    --env HTTP_PROXY=$HTTP_PROXY \
    --env HTTPS_PROXY=$HTTPS_PROXY \
    --env http_proxy=$http_proxy \
    --env https_proxy=$https_proxy \
    agents_img
```

## GPU settings

If you need to use GPU acceleration with your container, you'll need to configure Docker to access your NVIDIA GPUs.

1. Install the NVIDIA Container Toolkit:
   ```
   # Follow the installation guide at:
   # https://docs.nvidia.com/datacenter/cloud-native/container-toolkit/latest/install-guide.html
   ```

2. Build the Docker image as per above instructions (add proxy arguments if needed)

3. Run the container with GPU access:
   ```
   docker run -d \
       --name agents_cont \
       -v $(pwd):/repository:ro \
       -v agents_volume:/workspace/runs \
       --gpus all \
       --env NVIDIA_VISIBLE_DEVICES=all \
       agents_img
   ```

   Note: If using a proxy, add the appropriate environment variables as shown in the proxy section.