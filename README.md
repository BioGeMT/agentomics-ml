# Agentomics-ML

Agentomics-ML is an autonomous agentic system for development of machine learning models for omics data.

Given a classification dataset, the system automatically generates:
- A trained model file, loaded by the inference script.
- An inference script, allowing immediate predictions on new data.

Agents run inside an isolated docker environment for security.

![example](https://s6.gifyu.com/images/bz7xh.gif)


## Environment setup (Datasets + Docker + conda environment)

- Download datasets locally so they can be mounted into the container
```
conda create -n agentomics_temp_env --yes python && conda activate agentomics_temp_env && pip install genomic-benchmarks miRBench && python setup.py && conda deactivate && conda remove -n agentomics_temp_env --all --yes
```


- Make sure you have Docker installed
```
docker --version
```
  
- Install plugin for rescricting the volume size
```
docker plugin install ashald/docker-volume-loopback
```

- Create volume with maximum storage size
```
docker volume create -d ashald/docker-volume-loopback:latest -o size=50G agents_volume
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
python tests/tests.py -v
```
- Activate the default conda environment
```
source activate agentomics-env
```

Create a `.env` file containing your API keys. Example content: `OPENAI_API_KEY=my-api-key-1234`

Run `python src/playground.py` to check everything works

Go to `https://wandb.ai/ceitec-ai/BioAgents` to see agent logs.

## Guidelines and tips

Whatever packages you add, add them to the environment.yaml file

If they are conda packages, you can just re-run
`conda env export --from-history > environment.yaml`

If they are pip packages, you have to manually add them.

You can update your existing environment by running this command
`conda env update --file environment.yaml --prune`

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