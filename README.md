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

## Quick Start (using a sample dataset)

```bash
# 1. Set your API key (see the PROVIDERS readme section for more API key options)
export OPENROUTER_API_KEY="your-key-here"
# OR create a .env file (see .env.example)

# 2. Run the agent and select one of the sample datasets
./run.sh
```

## Run outputs

After the run is finished, the `outputs` folder contains

- Generated files (training script, inference script, model files, ...)
- Final report (Summary of the model, train/valid/test metrics, ...)

## Run on your own dataset

Create a folder inside `Agentomics-ML/datasets` and drop your files there

- add `train.csv` - Contains your training data. This will be used by the agent for training and validation
- (OPTIONAL) add `validation.csv` - Contains your validation data. If provided, the agent will use this instead of creating its own train/validation split.
- (OPTIONAL) add `test.csv` - Contains your testing data. This will be hidden from the agent, and used to add test set metrics to the final report.
- (OPTIONAL) add `dataset_description.md` - Data description and domain information for the agent. See the sample datasets for examples.

The csv files must contain a column for the classification or regression labels named exactly either `class` or `target`.

**Note:** If you don't provide a `validation.csv`, the agent will automatically create a train/validation split from your `train.csv` during its first iteration.

See the `datasets` folder for examples

After you've added your dataset folder and files, run agentomics:

```bash
# 1. Set your API key (see the PROVIDERS readme section for more API key options)
export OPENROUTER_API_KEY="your-key-here"
# OR create a .env file (see .env.example)

# 2. Run the agent and select your dataset
./run.sh
```

## Predictions

When getting predictions on new data, make sure the data file has the same column names as your training data (except the class/target column).

The output will be a csv file containing a single columns called 'predictions' in the same order as your data.

```
./inference.sh --agent-dir outputs/<agent_id> --input <test_data_file>.csv --output <prediction_file>.csv
```

The `<agent_id>` is the name of the finished run (same as a folder name in `outputs/`)

Optionally pass `--cpu-only` flag if you wish to only run the inference with CPU.

If you wish to run the inference locally and not inside a docker container, use the `--local` flag.

To customize or directly access the inference code, use artifacts in the `outputs/<run_name>/best_run_files` folder.

- `.conda` folder contains the conda environment with packages necessary for the inference
- `inference.py` contains the inference code, and might depend on other artifacts in the folder (like `model.joblib`, tokenizer files, etc..)

# Advanced run parameters

## Providers

We support various providers out-of-the-box. Below is a list of providers, with the specific environment variables names we check for. Those should be provided in the `.env` file, or exported.

- OpenRouter (OPENROUTER_API_KEY)
- OpenAI (OPENAI_API_KEY)
- Anthropic (ANTHROPIC_API_KEY)

If you wish to use other providers, add them to the `src/utils/providers/configured_providers.yaml` configuration. For those, we won't support a "nice" interactive model selection, and you will need to provide the `--model` parameter explicitly.

### Local LLM (Ollama)

We support Ollama for running with local models

If youre running agentomics in docker mode (recommended), additional steps are needed:

- Run `systemctl edit ollama.service` and add the following lines to allow Ollama to listen the requests coming from the container:

  ```
  [Service]
  Environment="OLLAMA_HOST=172.17.0.1:11434"
  ```

- Restart Ollama to propagate this change
  ```
  systemctl daemon-reload
  systemctl restart ollama.service
  ```
- Add the `--ollama` flag when executing the `./run.sh` script.

If youre running agentomics in local mode (unsafe), `OLLAMA_BASE_URL` environment variable needs to be provided (either in the `.env` file or exported, typically `BASE_OLLAMA_URL=http://localhost:11434/v1`).

## Explicit parameters

Running `./run.sh` with no parameters will prompt you to select them interactively.

You can also supply them directly to skip the interactive selection

```
.run.sh \
  --model gpt-5-nano \ # provider-specific
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

## Forking from Previous Runs

**Fork from any step in a previous run to try different approaches!**

Agentomics-ML supports forking from specific steps in previous runs, allowing you to:

- Resume from a specific point in the pipeline
- Try different approaches after a particular step
- Save time by reusing work from earlier steps
- Experiment with variations without starting from scratch

### How Forking Works

When you fork from a step:

1. The agent restores the **exact workspace state** from after that step completed
2. It loads the **conversation history** and outputs from all previous steps
3. It continues execution from the next step (or re-executes the forked step if desired)
4. All work is done in a new run with a unique ID

Forking is powered by:

- **Content-addressed storage**: Files are deduplicated automatically, saving disk space
- **Copy-on-write** (when supported): Instant, zero-copy file restoration on APFS/BTRFS/XFS
- **Step snapshots**: After each step completes, workspace and outputs are automatically saved

### Available Fork Points

You can fork from any of these steps:

- `data_exploration` - After initial data analysis
- `data_split` - After train/validation split
- `data_representation` - After feature engineering/encoding
- `model_architecture` - After model design
- `model_training` - After model training
- `model_inference` - After inference script creation
- `prediction_exploration` - After prediction analysis

### Basic Forking

Fork from the latest iteration of a previous run:

```bash
./run.sh \
  --fork-from-run melodic_recipe_grind \
  --fork-from-step model_architecture \
  --model gpt-4o \
  --dataset breast_cancer \
  --iterations 5 \
  --val-metric ACC
```

This will:

- Skip steps 0-3 (data_exploration through model_architecture)
- Start fresh from step 4 (model_training)
- Use the workspace state from after model_architecture completed

### Fork from Specific Iteration

Or fork from a specific iteration number:

```bash
./run.sh \
  --fork-from-run melodic_recipe_grind \
  --fork-from-step data_representation \
  --fork-from-iteration 2 \
  --model gpt-4o \
  --dataset breast_cancer \
  --iterations 5 \
  --val-metric ACC
```

### Custom Fork ID

By default, forked runs get an auto-generated ID like `melodic_recipe_grind_fork_model_training_iter2_20250112_143022` (includes iteration if specified) or `melodic_recipe_grind_fork_model_training_20250112_143022` (uses latest iteration).

You can specify a custom ID:

```bash
./run.sh \
  --fork-from-run melodic_recipe_grind \
  --fork-from-step model_architecture \
  --agent-id my_custom_fork_experiment \
  --model gpt-4o \
  --dataset breast_cancer \
  --iterations 5 \
  --val-metric ACC
```

### Example Use Cases

**Try a different model architecture:**

```bash
# Original run used a neural network
# Fork after data_representation to try XGBoost instead
./run.sh \
  --fork-from-run original_run_id \
  --fork-from-step data_representation \
  --user-prompt "Use XGBoost with careful hyperparameter tuning" \
  --model gpt-4o \
  --dataset breast_cancer \
  --iterations 3 \
  --val-metric ACC
```

**Fix data representation issues:**

```bash
# Fork from data_split to redo representation with different encoding
./run.sh \
  --fork-from-run original_run_id \
  --fork-from-step data_split \
  --user-prompt "Use target encoding instead of one-hot for categorical features" \
  --model gpt-4o \
  --dataset breast_cancer \
  --iterations 5 \
  --val-metric ACC
```

**Continue after timeout:**

```bash
# Original run timed out during training
# Fork from model_architecture to continue where it left off
./run.sh \
  --fork-from-run timed_out_run_id \
  --fork-from-step model_architecture \
  --model gpt-4o \
  --dataset breast_cancer \
  --iterations 3 \
  --val-metric ACC
```

**Fork from a fork (experiment with variations):**

```bash
# Original run: melodic_recipe_grind
# First fork: melodic_recipe_grind_fork_model_training_iter2_20250112_143022
# Now fork from the first fork to try a different approach

./run.sh \
  --fork-from-run melodic_recipe_grind_fork_model_training_iter2_20250112_143022 \
  --fork-from-step model_architecture \
  --user-prompt "Try ensemble methods combining the previous model with a gradient boosting classifier" \
  --model gpt-4o \
  --dataset breast_cancer \
  --iterations 3 \
  --val-metric ACC
```

This creates a fork-of-fork with ID like: `melodic_recipe_grind_fork_model_training_iter2_20250112_143022_fork_model_architecture_iter1_20250113_150000`

**Note:** Fork IDs can get long with deep fork chains. The system automatically truncates if needed to stay within filesystem limits (255 characters on macOS/Linux).

### How Snapshots Are Stored

Snapshots are stored in the `.agentomics_storage` directory using a git-like content-addressed system:

```
workspace/
  .agentomics_storage/
    objects/          # Deduplicated file contents (by SHA-256 hash)
      a1/
        23f4d5e6...   # Actual file content
      b7/
        89c2e1f3...
    snapshots/        # Lightweight manifests (just path→hash mappings)
      melodic_recipe_grind/
        iteration_0/
          00_data_exploration.json
          01_data_split.json
          02_data_representation.json
          ...
      iteration_1/
          ...
```

**Benefits:**

- **Space efficient**: Identical files are stored only once
- **Fast**: Only changed files are stored after each step
- **Portable**: Copy the entire `.agentomics_storage` directory to share runs

### Storage Statistics

Check how much space your snapshots use:

```python
from utils.content_store import ContentAddressedStore
from pathlib import Path

store = ContentAddressedStore(Path("workspace"))
stats = store.get_storage_stats()
print(f"Total objects: {stats['total_objects']}")
print(f"Total size: {stats['total_size_mb']:.2f} MB")
print(f"Total snapshots: {stats['total_snapshots']}")
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
To enable logging, specify WANDB\_\* keys in your `.env` file (see `.env.example`)

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

GPU support is enabled by default. To use GPU acceleration, you need to configure Docker to access your NVIDIA GPUs:

1. Install the NVIDIA Container Toolkit:

   ```
   # Follow the installation guide at:
   # https://docs.nvidia.com/datacenter/cloud-native/container-toolkit/latest/install-guide.html
   ```

2. Run normally (GPU will be used automatically):

   ```
   ./run.sh
   ```

If you want to disable GPU support, use the `--cpu-only` flag:

```
./run.sh --cpu-only
```
