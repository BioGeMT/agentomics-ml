# Try out [Google Colab Demo](https://colab.research.google.com/drive/1tCJtTrimw9OviErtKi7FRo5Nx09u7vhv?usp=sharing)

# Agentomics-ML

🤖 **Autonomous AI agent for supervised machine learning model development on omics data**

Given a raw dataset, Agentomics-ML autonomously generates
- A trained model, ready to run inference on new data
- A report summarizing the model development process and evaluation metrics

**⏱️ Typical timing:** 30-120 minutes depending on dataset size and complexity

Agentomics-ML works like a ML engineer
- Explores data before designing a model
- Conciders domain information
- Chooses proper data representation
- Designs and trains models, including custom neural networks
- Works iteratively, reacting to issues like overfitting and underfitting based on validation metrics
- Produces working scripts, including their conda environments

You can use
- any LLM, including local models
- any classification or regression dataset in a csv format (see #Run on your own dataset)
- variety of metrics for validation (accuracy, AUROC, AURPC, MSE, ...)

📖 [Preprint](https://arxiv.org/abs/2506.05542) | 🚀 [Quick Start](#quick-start) | [Website](https://agentomicsml.com/)
## Download

```
git clone https://github.com/BioGeMT/Agentomics-ML.git
cd Agentomics-ML
```

> **Prerequisites:**
>
> - **Docker mode (recommended)**: Docker must be installed. [Get Docker here](https://docs.docker.com/get-docker/) if needed.
> - **Local mode**: Conda must be installed.

## Quick Start

```bash
# 1. Set your API key (get from https://openrouter.ai)
export OPENROUTER_API_KEY="your-key-here"
# OR create a .env file (see .env.example) 

# 2. Run the agent and select one of the sample datasets
./run.sh
```

## Run on your own dataset
Create a folder inside `Agentomics-ML/datasets` and drop your files there

- add `train.csv` - Contains your training data. This will be used by the agent for training and validation
- add `test.csv` - Contains your testing data. This will be hidden from the agent, and used only to evaluate the final model. 
- add `dataset_description.md` - Domain information for the agent. See the sample datasets for examples.

The csv files must contain a 'class' or 'target' column for the classification or regression labels. 

### Create Dataset Structure

```
datasets/
  your_dataset/
    train.csv              # Required: your training data
    test.csv              # Optional: test data
    dataset_description.md # Optional: data description
```
### CSV Format

**Classification example:**

```csv
feature1,feature2,feature3,class
0.5,1.2,0.8,positive
0.3,2.1,0.6,negative
```

**Regression example:**

```csv
feature1,feature2,feature3,target
0.5,1.2,0.8,125.45
0.3,2.1,0.6,187.92
```

## Advanced run parameters
### TODO extra run.sh parameters, including non-interactive runs
<!-- ```bash
./run.sh --list-datasets        # Show available datasets (Docker)
./run.sh --local --list-datasets # Show available datasets (Local)
./run.sh --list-models          # Show available AI models (Docker)
./run.sh --local --list-models   # Show available AI models (Local)
./run.sh --list-metrics         # Show available validation metrics
./run.sh --prepare-only         # Just prepare datasets (Docker)
./run.sh --local --prepare-only  # Just prepare datasets (Local)
./run.sh --help                # Show all options
``` -->
<!-- ```bash
# Pre-built image with specific dataset/model
./run.sh cloudmark -d heart_disease -m "openai/gpt-4.1"

# Custom image with utility commands
./run.sh myorg --list-datasets
./run.sh cloudmark --prepare-only
``` -->
### Local (no-docker) run
<div style="border:2px solid red; background:#ee2400; padding:10px; border-radius:6px;">
  <strong>⚠️ Warning:</strong> When you run outside of the main script (`run.sh`), only run scripts inside a secure environment (like your own docker container)! The agent tools can exectute arbitrary bash commands!
</div>

#### Quickstart
```bash
# 1. Set your API key (get from https://openrouter.ai)
export OPENROUTER_API_KEY="your-key-here"
# OR create a .env file (see .env.example) 

# 2. Run the agent and select one of the sample datasets
./run_local.sh #!Only run in your own secure environment!
```

#### Running scripts separately
If you want to have more fine-grained control over the agent runs, follow these steps:
##### Dataset preparation
To prepare datasets (using data from the Agentomics-ML/datasets directory) for the agent, run:
```
conda env create -f environment_prepare.yaml
conda activate agentomics-prepare-env
python src/prepare_datasets.py
```
##### Agent run
To run the agent and select options interactively, run:
```
conda env create -f environment.yaml
conda activate agentomics-env
python src/run_agent_interactive.py
```

To run the agent directly (pre-specifying arguments)
```
conda env create -f environment.yaml
conda activate agentomics-env
python src/run_agent.py --model <model> --dataset <dataset> --val-metric <val_metric>
```

### Logging
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
