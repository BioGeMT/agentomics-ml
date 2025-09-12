# Try out [Google Colab Demo](https://colab.research.google.com/drive/1tCJtTrimw9OviErtKi7FRo5Nx09u7vhv?usp=sharing)

# Agentomics-ML

🤖 **Autonomous AI agent for supervised machine learning model development on omics data**

Given a raw dataset, Agentomics-ML autonomously generates
- A trained model, ready to run inference on new data
- A report summarizing the model development process and evaluation metrics

<!-- **⏱️ Typical timing:** 30-120 minutes depending on dataset size and complexity -->

Agentomics-ML works like a ML engineer
- Explores data before designing a model
- Conciders domain information
- Chooses proper data representation
- Designs and trains models, including custom neural networks
- Works iteratively, reacting to issues like overfitting and underfitting based on validation metrics
- Produces working scripts, including their conda environments

Currently Agentomics-ML supports
- Any LLM, including local models
- Any classification or regression dataset in a csv format
- Secure runs using docker containers and volumes, constraining the agent to read-only access to the Agentomics-ML folder and code execution only inside a docker container


📖 [Preprint](https://arxiv.org/abs/2506.05542) | 🚀 [Quick Start](#quick-start) | [Website](https://agentomicsml.com/)
## Download

```
git clone https://github.com/BioGeMT/Agentomics-ML.git
cd Agentomics-ML
```

**Prerequisites:**

- **Docker mode (recommended)**: [Docker](https://docs.docker.com/get-docker/) must be installed.
- **Local mode**: Conda must be installed.

## Quick Start

```bash
# 1. Set your API key (get from https://openrouter.ai)
export OPENROUTER_API_KEY="your-key-here"
# OR create a .env file (see .env.example) 

# 2. Run the agent and select one of the sample datasets
./run.sh
```

After the run is finished, the `outputs` folder contains 
- Generated files (training script, inference script, model files, ...)
- Final report (Summary of the model, train/valid/test metrics, ...)

## Run on your own dataset
Create a folder inside `Agentomics-ML/datasets` and drop your files there

- add `train.csv` - Contains your training data. This will be used by the agent for training and validation
- (OPTIONAL) add `test.csv` - Contains your testing data. This will be hidden from the agent, and used to add test set metrics to the final report.
- (OPTIONAL) add `dataset_description.md` - Data description and domain information for the agent. See the sample datasets for examples.

The csv files must contain a column for the classification or regression labels named exactly either `class` or `target`. 

See the `datasets` folder for examples

## Predictions
When getting predictions on new data, make sure the data file has the same column names as your training data. There's no need to provide the class/target column, as this will be predicted.

The output will be a csv file containing a single columns called 'predictions' in the same order as your data.
```
cd outputs/best_run_files/<run_name>
conda activate .conda/envs/<run_name> 
python inference.py --input <path_to_inference_data_csv> --output <path_to_output_csv>
```


# Advanced run parameters
## Explicit parameters
Running `./run.sh` with no parameters will prompt you to select them interactively.

You can also supply them directly to skip the interactive selection
```
.run/sh \
  --model gpt-5-nano \
  --dataset human_ocr_ensembl \
  --iterations 5 \
  --val-metric ACC \
```
Run `./run.sh --help` for more information.

## Custom user prompt
The default prompt: 
`Create the best possible machine learning model that will generalize to new unseen data.`

You can overwrite it with your own user prompt for the agent by passing the `--user-prompt` argument.
```
/run.sh --user-prompt "Only create simple ML models like logistic regression and shallow decision trees"
```

## Local mode (no-docker)
<div style="border:2px solid red; background:#ee2400; padding:10px; border-radius:6px;">
  <strong>⚠️ Warning:</strong> Only run local mode inside a secure environment (like your own docker container with read-only mounts or google colab)! The agent tools can exectute arbitrary bash commands!
</div>

If you can't create your own docker container, you can run in local mode with significantly decreased security by adding the `--local` flag.

`./run.sh --local`

## Running scripts separately
If you want to have more fine-grained control over the agent runs, follow these steps:
### Dataset preparation
To prepare datasets (using data from the Agentomics-ML/datasets directory) for the agent, run:
```
conda env create -f environment_prepare.yaml
conda activate agentomics-prepare-env
python src/prepare_datasets.py
```

Run `python src/prepare_datasets.py --help` for info on more fine-grained control of dataset preparation (e.g. explicitly specifying classification/regression task, explicit positive/negative class, etc..)

### Agent run
To run the agent run:
```
conda env create -f environment.yaml
conda activate agentomics-env
python src/run_agent_interactive.py
```

To run the agent with more logging options and pre-specifying arguments
```
conda env create -f environment.yaml
conda activate agentomics-env
python src/run_agent.py --model <model> --dataset <dataset> --val-metric <val_metric>
```

## Logging
We support logging to W&B, including agent traces, metrics of various model iterations, and generated files.
To enable logging, specify WANDB_* keys in your `.env` file (see `.env.example`)

# Developer information

## Configuration

To modify agent behavior (LLM temperature, timeouts, etc.), edit `src/utils/config.py`

## Build and Push Commands

For developers wanting to build and distribute their own Docker images, use the provided build script:

```bash
# Build and push both images (multi-architecture)
./build.sh myusername

# Build and push specific version
./build.sh myusername v1.0
```

**Features:**

- **Multi-platform Support**: Automatically detects and builds for your platform
- **Dependency Management**: Includes all required Python packages
- **Dataset Processing**: Handles various dataset formats and preprocessing
- **Error Handling**: Docker provides clear error messages for build/push issues

### Multi-Architecture Builds (ARM64 + AMD64)

For production deployments and wide compatibility, you should build and push multi-architecture images that work on both Intel/AMD processors (amd64) and ARM processors (arm64, including Apple Silicon).

#### Prerequisites

First, enable Docker's multi-platform builder:

```bash
# Create and use a multi-platform builder
docker buildx create --name multiplatform --use
docker buildx inspect --bootstrap
```

#### Build Multi-Architecture Images

**Build both architectures for main agent:**

```bash
# Build and push main agent image for multiple architectures
docker buildx build \
  --platform linux/amd64,linux/arm64 \
  -t myusername/agentomics:latest \
  --push .
```

**Build both architectures for preparation image:**

```bash
# Build and push preparation image for multiple architectures
docker buildx build \
  --platform linux/amd64,linux/arm64 \
  -f Dockerfile.prepare \
  -t myusername/agentomics-prepare:latest \
  --push .
```



**With version tags:**

```bash
# Build and push specific versions (multi-arch)
docker buildx build \
  --platform linux/amd64,linux/arm64 \
  -t myusername/agentomics:v1.0 \
  -t myusername/agentomics:latest \
  --push .

docker buildx build \
  --platform linux/amd64,linux/arm64 \
  -f Dockerfile.prepare \
  -t myusername/agentomics-prepare:v1.0 \
  -t myusername/agentomics-prepare:latest \
  --push .
```

#### Architecture-Specific Builds (Optional)

If you need to build for specific architectures only:

```bash
# Build only for ARM64 (Apple Silicon, ARM servers)
docker buildx build \
  --platform linux/arm64 \
  -t myusername/agentomics:arm64 \
  --push .

# Build only for AMD64 (Intel/AMD processors)
docker buildx build \
  --platform linux/amd64 \
  -t myusername/agentomics:amd64 \
  --push .
```

#### Verification

**Check that your images support multiple architectures:**

```bash
# Inspect manifests to see supported architectures
docker buildx imagetools inspect myusername/agentomics:latest
docker buildx imagetools inspect myusername/agentomics-prepare:latest

# Expected output should show both:
# - linux/amd64
# - linux/arm64
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

- http_proxy
- https_proxy
- HTTP_PROXY
- HTTPS_PROXY

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
  -t agentomics .
```

Or use the build script with proxy environment variables set:

```
./build.sh
```

The `./run.sh` script automatically handles proxy settings if environment variables are set.

## GPU settings

If you need to use GPU acceleration with your container, you'll need to configure Docker to access your NVIDIA GPUs.

1. Install the NVIDIA Container Toolkit:

   ```
   # Follow the installation guide at:
   # https://docs.nvidia.com/datacenter/cloud-native/container-toolkit/latest/install-guide.html
   ```

2. Build the Docker image as per above instructions (add proxy arguments if needed)

3. The `./run.sh` script automatically detects and uses GPU access when available:

   ```
   ./run.sh
   ```

   For manual setup with GPU access:

   ```
   docker run -d \
       --name agentomics-agent \
       -e OPENROUTER_API_KEY="your-key" \
       -e WANDB_API_KEY="your-wandb-key" \
       -e DATASET_NAME="sample_dataset" \
       -e MODEL_NAME="openai/gpt-4.1" \
       -e PREPARED_DATASETS_DIR="/repository/prepared_datasets" \
       -v agentomics-workspace:/workspace \
       -v $(pwd):/repository \
       -v $(pwd)/prepared_datasets:/repository/prepared_datasets \
       --gpus all \
       --env NVIDIA_VISIBLE_DEVICES=all \
       agentomics/agentomics:latest
   ```
