# Try our [COLAB DEMO](https://colab.research.google.com/drive/1tCJtTrimw9OviErtKi7FRo5Nx09u7vhv?usp=sharing)
# Agentomics-ML

Agentomics-ML is an autonomous agentic system for development of machine learning models for omics data.

Given a dataset, the system automatically generates a trained model and an inference script, allowing immediate predictions on new data.

Learn more in our [arXiv pre-print](https://arxiv.org/abs/2506.05542) 
## Download
```
git clone https://github.com/BioGeMT/Agentomics-ML.git
cd Agentomics-ML
```

## Environment setup (Docker + conda environment)
NOTE: We provide a docker container for running Agentomics-ML securely. If your machine doesn't allow you to create a new docker container, it's possible to run in an already existing docker container (e.g. colab). If you do, we advise to run in a disposable container and setup proper permissions (e.g. only mounting files read-only, not storing valuable files) since agents will be able to run arbitrary bash commands in this container (for example rm -rf /*).
If you run in your own docker container, create environment by running `conda env create -f environment.yaml`, activate it, and proceed to the <b>Prepare your API keys</b> step.

### Make sure you have Docker installed
```
docker --version
```
  
<!-- - Install plugin for rescricting the volume size
```
docker plugin install ashald/docker-volume-loopback
``` -->

### Create volume (this will store all agent-generated files)
```
docker volume create agentomics_volume
```

<!-- - Create volume with maximum storage size
```
docker volume create -d ashald/docker-volume-loopback:latest -o size=50G agents_volume
``` -->

### Build docker image
```
docker build -t agentomics_img .
```

### Run the container
We recommend running the container with access to your GPUs, scroll down to the `GPU settings` section to see how to install the Nvidia container toolkit if you haven't yet.
```
docker run -d \
    --name agentomics_cont \
    -v $(pwd):/repository:ro \
    -v agentomics_volume:/workspace/runs \
    --gpus all \ #add only if you use GPUs
    --env NVIDIA_VISIBLE_DEVICES=all \ #add only if you use GPUs
    agentomics_img
```

This mounts the repository directory in read-only mode for security

### Attach console to the container
```
docker exec -it agentomics_cont bash
```
### Activate the default conda environment
```
source activate agentomics-env
```

## Prepare your API keys
Create a `.env` file in the Agentomics-ML folder containing your API keys. Example content: `OPENROUTER_API_KEY=my-api-key-1234`

You will need to add an `OPENROUTER_API_KEY` (https://openrouter.ai)

Optionally, to enable logging your runs you can add a `WANDB_API_KEY` (https://wandb.ai)

## Prepare your dataset
### Add your files
Create a folder for your dataset in the `datasets` folder and add your files. Follow the `datasets/sample_dataset` structure. For easiest use follow these rules:
- Name your files exactly `train.csv`, `test.csv` and `dataset_description.md` 
- In your csv files, provide a column called `target` that will contain labels. For classification tasks, these can be both strings and integers. For regression tasks, these should be numeric values.

Possible customizations:
<!-- - providing `test.csv` is optional. Without it, test-set metrics will not be provided to the user at the end of the run. TODO implement -->
- providing `dataset_description.md` is optional. Without it, the agent will be slightly limited but functional.


### Preprocess your files 
To generate necessary metadata and files, run this command. Replace `sample_dataset` with the name of your dataset folder. 
Specify the task type (possible values: `classification` and `regression`).
```
python src/utils/prepare_dataset.py --dataset-dir datasets/sample_dataset --task-type classification
```


This will create `prepared_datasets` folder as a sibling folder to Agentomics-ML.

If you're running in your own Docker container with permission restrictions, you can customize the path of this folder by passing `--output-dir <your/path/prepared_datasets>`

If your label column has a different name than `class`, or you want to specify your own label mapping for binary datasets, run `python src/utils/prepare_dataset.py --help` for more info. 


## Run the agent
Run this command, replace `sample_dataset` with the name of your dataset folder. This can take up to few hours.
```
python src/run_agent.py --dataset-name sample_dataset
```

This will output all files and metrics from the run into the `workspace` directory, which is created as a sibling directory to the Agemtomics-ML folder. See the `workspace/runs/snapshots` directory for best-run files.

If you're using your own docker container and don't have sudo permissions, pass the `--no-root-privileges` flag to the `run_agent.py` script to enable the agent to run. This will cause the agent to have access to files outside of its own workspace (like other agent runs).

If you've changed the `--output-dir` in the `Preprocess your files`  step, you need to pass the same path to the `--prepared-datasets-dir` argument to this script. Run `--help` to see more customization options



If you want to modify any agent-specific details like LLM teperature or timeouts, change them in the `src/utils/config.py` file

If you've initialized a wandb api key, you can also see the whole agent trace and metric progression in your wandb project. You can also track the agent progress in your console and by inspecting files in its `workspace` directory.




# Extras


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

## Developer guidelines and tips

Whatever packages you add, add them to the environment.yaml file

If they are conda packages, you can just re-run
`conda env export --from-history > environment.yaml`

If they are pip packages, you have to manually add them.

You can update your existing environment by running this command
`conda env update --file environment.yaml --prune`
