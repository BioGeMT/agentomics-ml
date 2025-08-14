# Agentomics-ML

🤖 **Autonomous AI agent for machine learning model development on omics data**

Given a classification or regression dataset, Agentomics-ML automatically generates a trained model and inference script, allowing immediate predictions on new data.

📖 [Research Paper](https://arxiv.org/abs/2506.05542) | 🚀 [Quick Start](#quick-start) | 🐳 [Docker Hub](https://hub.docker.com/r/cloudmark/agentomics)

## Download

```
git clone https://github.com/BioGeMT/Agentomics-ML.git
cd Agentomics-ML
```

## Quick Start

**Get running in 3 commands:**

```bash
# 1. Set your API key (get from https://openrouter.ai)
export OPENROUTER_API_KEY="your-key-here"

# 2. Run the agent
./run.sh                    # Docker mode (recommended)
./run.sh --local           # Local mode (faster for development)

# 3. Follow prompts to select dataset, metric, and model
```

**Or skip prompts with direct execution:**

```bash
./run.sh -d sample_dataset -m "openai/gpt-4.1"                    # Classification (auto-detects metric)
./run.sh -d electricity_cost_dataset -m "openai/gpt-4.1"          # Regression (auto-detects metric)
./run.sh -d sample_dataset -m "openai/gpt-4.1" --metric AUROC     # Classification with specific metric
./run.sh -d electricity_cost_dataset -m "openai/gpt-4.1" --metric RMSE # Regression with specific metric
./run.sh --local -d sample_dataset -m "openai/gpt-4.1"            # Local mode
```

> **Prerequisites:**
>
> - **Docker mode**: Docker must be installed. [Get Docker here](https://docs.docker.com/get-docker/) if needed.
> - **Local mode**: Conda environment with `environment.yaml` dependencies installed.

## What Happens

The agent automatically:

- 🔍 **Analyzes your data** and detects task type (classification/regression)
- 🎯 **Lets you choose validation metric** based on task type (interactive selection)
- 🤖 **Lets you select AI model** from available options with pricing
- 🏗️ **Designs ML architecture** optimized for your dataset
- 🎯 **Trains and validates** models using best practices
- 📊 **Evaluates performance** with appropriate metrics
- 💾 **Saves results** to `workspace/runs/[run-id]/` directory

**⏱️ Typical timing:** 5-30 minutes depending on dataset size and model complexity.

## Setup

### 1. Download

```bash
git clone https://github.com/BioGeMT/Agentomics-ML.git
cd Agentomics-ML
```

### 2. API Keys

**Option A: Environment variables (quick)**

```bash
export OPENROUTER_API_KEY="your-openrouter-key"    # Required
export WANDB_API_KEY="your-wandb-key"              # Optional (for tracking)
```

**Option B: .env file (recommended)**

```bash
cp .env.example .env
# Edit .env with your actual API keys
```

### 3. Run

```bash
./run.sh  # Interactive mode - prompts for dataset/model selection
```

## Commands

### Basic Usage

```bash
# Interactive mode (recommended for first-time users)
./run.sh                    # Docker mode
./run.sh --local           # Local mode (faster development)

# Direct execution
./run.sh -d your_dataset -m "openai/gpt-4.1"              # Docker mode
./run.sh --local -d your_dataset -m "openai/gpt-4.1"      # Local mode

# With custom Docker image
./run.sh myusername v1.0 -d dataset -m model
```

### Utility Commands

```bash
./run.sh --list-datasets        # Show available datasets (Docker)
./run.sh --local --list-datasets # Show available datasets (Local)
./run.sh --list-models          # Show available AI models (Docker)
./run.sh --local --list-models   # Show available AI models (Local)
./run.sh --list-metrics         # Show available validation metrics
./run.sh --prepare-only         # Just prepare datasets (Docker)
./run.sh --local --prepare-only  # Just prepare datasets (Local)
./run.sh --help                # Show all options
```

### Local Mode (Development)

For faster development and testing, use local mode to skip Docker overhead:

```bash
# Prerequisites: Install conda environment
conda env create -f environment.yaml
conda activate agentomics-env

# Run in local mode
./run.sh --local                                         # Interactive
./run.sh --local -d sample_dataset -m "openai/gpt-4.1"   # Direct
./run.sh --local --list-datasets                         # List datasets
./run.sh --local --list-models                           # List models

# Custom conda environment
./run.sh --local --conda-env my-custom-env -d dataset -m model
```

**Benefits of Local Mode:**

- 🚀 **Faster startup** - No Docker image pulling or container creation
- 🔍 **Better debugging** - Direct access to Python debugger and logs
- 📝 **Live code editing** - Test changes without rebuilding containers
- 🧪 **Quick iteration** - Immediate feedback for development

### Custom Docker Images

```bash
# Use your organization's images
./run.sh myorg              # myorg/agentomics:latest
./run.sh myorg v2.1         # myorg/agentomics:v2.1

# Build your own
./build.sh myusername       # Builds and pushes to Docker Hub
```

## Docker Images

Agentomics-ML supports both pre-built Docker images (recommended for most users) and custom builds (for developers).

### Pre-built Docker Images (Recommended)

The easiest way to use Agentomics-ML is with our pre-built Docker images. These work perfectly in Google Colab, cluster environments, or any system with Docker installed.

**Default usage (uses cloudmark/agentomics:latest):**

```bash
./run.sh
```

**Using specific pre-built versions:**

```bash
./run.sh cloudmark          # Uses cloudmark/agentomics:latest
./run.sh cloudmark v1.0     # Uses cloudmark/agentomics:v1.0
```

### Custom Docker Images

For developers who want to use their own builds or different Docker Hub accounts:

**Two ways to specify custom images:**

1. **Positional arguments** (concise):

   ```bash
   ./run.sh [username] [version]
   ```

2. **Explicit options** (clear):
   ```bash
   ./run.sh --username [username] --version [version]
   ```

### Examples

**Pre-built images (recommended for most users):**

```bash
# Default pre-built image (easiest)
./run.sh

# Specific pre-built versions
./run.sh cloudmark          # Latest version
./run.sh cloudmark v1.5     # Specific version

# Perfect for Google Colab or cluster environments
./run.sh cloudmark -d heart_disease -m "openai/gpt-4.1"
```

**Custom Docker Hub users:**

```bash
# Use your organization's images
./run.sh myorg              # myorg/agentomics:latest
./run.sh myorg v2.1         # myorg/agentomics:v2.1

# Explicit syntax (same result)
./run.sh --username myorg --version v2.1
```

**Combined with other options:**

```bash
# Pre-built image with specific dataset/model
./run.sh cloudmark -d heart_disease -m "openai/gpt-4.1"

# Custom image with utility commands
./run.sh myorg --list-datasets
./run.sh cloudmark --prepare-only
```

### What Happens

When you specify a custom username/version:

1. **Main Agent**: Uses `username/agentomics:version`
2. **Dataset Preparation**: Uses `username/agentomics-prepare:version`
3. **Utilities**: All commands use the specified image

### Error Handling

If the specified image doesn't exist, you'll get helpful options:

```bash
Failed to pull Docker image: myuser/agentomics:v1.0
Available options:
  1. Build locally:           ./build.sh myuser v1.0
  2. Use CloudMark image:     ./run.sh cloudmark
  3. Use another DockerHub:   ./run.sh <username>
  4. Use local mode:          ./run.sh --local

Recommended: ./build.sh myuser v1.0 (builds with your latest environment.yaml)
```

**Quick Development Option:** If you just want to test or develop quickly, use local mode:

```bash
# Skip Docker entirely for development
./run.sh --local -d your_dataset -m "openai/gpt-4.1"
```

### Default Behavior

- **No username specified**: Uses `cloudmark/agentomics:latest` (pre-built image)
- **No version specified**: Defaults to `latest`
- **Both blank**: `cloudmark/agentomics:latest` (recommended for most users)

This makes it perfect for Google Colab, cluster environments, or any system where you just want to run the agent without building Docker images.

### Building Your Own Images

To create custom images for your organization, use the provided build script:

```bash
# Build and push both images (multi-architecture)
./build.sh myusername

# Build and push specific version
./build.sh myusername v1.0
```

The script automatically builds and pushes both images with multi-architecture support (linux/amd64, linux/arm64).

**Use your custom images:**

```bash
# Use your published images
./run.sh myusername

# Use specific versions
./run.sh myusername v1.0
```

## Local Development Setup

For faster development and testing without Docker:

### 1. Create Conda Environment

```bash
# Create environment from environment.yaml
conda env create -f environment.yaml

# Activate environment
conda activate agentomics-env

# Verify installation
python -c "import pandas, torch, transformers; print('Environment ready!')"
```

### 2. Run in Local Mode

```bash
# Interactive mode
./run.sh --local

# Direct execution
./run.sh --local -d sample_dataset -m "openai/gpt-4.1"

# List utilities
./run.sh --local --list-datasets
./run.sh --local --list-models
```

### 3. Local Mode Features

- 🚀 **Faster startup** - No Docker image pulling
- 🔍 **Better debugging** - Direct Python debugging with breakpoints
- 📝 **Live editing** - Modify code without rebuilding containers
- 🧪 **Quick testing** - Immediate feedback during development
- 🏠 **Native paths** - Uses `./datasets/` and `./prepared_datasets/` directly

### 4. Troubleshooting Local Mode

**Common Issues:**

```bash
# Missing conda environment
conda env create -f environment.yaml
conda activate agentomics-env

# Missing dependencies
conda env update -f environment.yaml --prune

# Path issues (ensure you're in project root)
ls -la agentomics-entrypoint.py  # Should exist

# API key issues
export OPENROUTER_API_KEY="your-key-here"
```

## Advanced Setup (For Developers)

If you want to build your own images or contribute to development:

### Build and Push

Use the provided build script for easy multi-architecture builds:

```bash
# Build and push both images (multi-architecture)
./build.sh myusername

# Build and push specific version
./build.sh myusername v1.0
```

## Add Your Dataset

### 1. Create Dataset Structure

```
datasets/
  your_dataset/
    train.csv              # Required: your training data
    test.csv              # Optional: test data
    dataset_description.md # Optional: data description
```

### 2. CSV Format

**Classification example:**

```csv
feature1,feature2,feature3,class
0.5,1.2,0.8,positive
0.3,2.1,0.6,negative
```

**Regression example:**

```csv
feature1,feature2,feature3,price
0.5,1.2,0.8,125.45
0.3,2.1,0.6,187.92
```

### 3. Auto-Detection

The system automatically:

- **Finds target column**: Looks for `class`, `target`, `label`, `y`, or uses last column
- **Detects task type**:
  - **Classification**: Categorical data or ≤10 unique numeric values
  - **Regression**: Numeric data with >10 unique values

### 4. Run

```bash
./run.sh -d your_dataset -m "openai/gpt-4.1"
```

That's it! No manual configuration needed.

### Build and Push Commands

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

#### Complete Multi-Architecture Workflow

**Build and push both images with multiple architectures:**

```bash
# 1. Ensure multi-platform builder is active
docker buildx create --name multiplatform --use --driver docker-container

# 2. Build and push main agent image (multi-arch)
docker buildx build \
  --platform linux/amd64,linux/arm64 \
  -t myusername/agentomics:latest \
  --push .

# 3. Build and push preparation image (multi-arch)
docker buildx build \
  --platform linux/amd64,linux/arm64 \
  -f Dockerfile.prepare \
  -t myusername/agentomics-prepare:latest \
  --push .

# 4. Verify multi-architecture support
docker buildx imagetools inspect myusername/agentomics:latest
docker buildx imagetools inspect myusername/agentomics-prepare:latest
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

#### Benefits of Multi-Architecture Images

- **Universal Compatibility**: Works on Intel, AMD, and ARM processors
- **Apple Silicon Support**: Native performance on M1/M2/M3 Macs
- **Cloud Flexibility**: Compatible with ARM-based cloud instances (AWS Graviton, etc.)
- **Single Image Tag**: Users don't need to worry about architecture-specific tags
- **Automatic Selection**: Docker automatically pulls the right architecture

**Note**: Multi-architecture builds take longer but ensure maximum compatibility for your users across different hardware platforms.

#### Automated Build Script

For convenience, you can use the provided `build.sh` script that handles the complete multi-architecture build and push workflow:

```bash
# Make script executable (first time only)
chmod +x build.sh

# Build and push both images (multi-architecture)
./build.sh myusername

# Build and push specific version
./build.sh myusername v1.0
```

The script automatically:

- Sets up multi-platform builders (linux/amd64, linux/arm64)
- Builds both agentomics and agentomics-prepare images
- Pushes to DockerHub with multi-architecture support

### Utility Scripts

#### Dataset Preparation

Dataset preparation is handled automatically by the Python script:

**Usage:**

```bash
# Automatic via run.sh
./run.sh --prepare-only

# Direct execution (in Docker)
conda run -n agentomics-env python -m src.utils.dataset_preparation --batch
```

**Features:**

- **Rich Table Display**: Beautiful formatted tables showing dataset status
- **Automatic Detection**: Scans `./datasets/` directory
- **Progress Tracking**: Real-time progress with spinners and status updates
- **Metadata Generation**: Creates `metadata.json` for each dataset
- **Error Handling**: Graceful handling of preparation failures

#### `agentomics-entrypoint.py` - Main Entry Point

Main entry point script supporting both Docker and local execution modes.

**Usage:**

```bash
# Docker mode (called automatically by containers)
python agentomics-entrypoint.py

# Local mode (for development and testing)
python agentomics-entrypoint.py --local

# Interactive dataset/model/metric selection
python agentomics-entrypoint.py --local --list-datasets
python agentomics-entrypoint.py --local --list-models
python agentomics-entrypoint.py --local --list-metrics

# Direct execution with parameters
python agentomics-entrypoint.py --local --dataset sample_dataset --model "openai/gpt-4.1"
python agentomics-entrypoint.py --local --dataset sample_dataset --model "openai/gpt-4.1" --val-metric AUROC
```

**Features:**

- **Dual Mode Support**: Works in both Docker containers and local conda environments
- **Environment Setup**: Automatically activates conda environment in local mode
- **Interactive Selection**: Beautiful Rich UI for dataset, metric, and model selection
- **Smart Path Detection**: Uses appropriate paths for Docker (`/repository/*`) vs Local (`./`) modes
- **Task-Aware Metric Selection**: Automatically detects task type and shows relevant metrics
- **Agent Launch**: Starts the main agent with selected parameters
- **Error Recovery**: Graceful handling with fallback mechanisms
- **Direct Function Calls**: Optimized imports with subprocess fallbacks

### Complete Build and Push Workflow

For developers wanting to build and distribute their own images, use the build script:

```bash
# Build and push both images with multi-architecture support
./build.sh myusername

# Build and push specific version
./build.sh myusername v1.0
```

### Manual Docker Setup

If you prefer manual Docker commands:

#### Make sure you have Docker installed

```
docker --version
```

#### Create volume (this will store all agent-generated files)

```
docker volume create agentomics-workspace
```

#### Build docker image

```
./build.sh myusername
```

#### Run the container manually

```
docker run -d \
    --name agentomics-agent \
    -e OPENROUTER_API_KEY="your-key-here" \
    -e WANDB_API_KEY="your-wandb-key" \
    -e DATASET_NAME="sample_dataset" \
    -e MODEL_NAME="openai/gpt-4.1" \
    -e PREPARED_DATASETS_DIR="/repository/prepared_datasets" \
    -v agentomics-workspace:/workspace \
    -v $(pwd):/repository \
    -v $(pwd)/prepared_datasets:/repository/prepared_datasets \
    agentomics:latest
```

**Volume Mounts Explained:**

- `-v $(pwd):/repository` - Mounts your project directory (contains source code and raw datasets)
- `-v $(pwd)/prepared_datasets:/repository/prepared_datasets` - Mounts prepared datasets directory
- `-v agentomics-workspace:/workspace` - Persistent workspace for agent outputs

**Environment Variables:**

- `DATASET_NAME` - Specify which dataset to use (from `./datasets/` directory)
- `MODEL_NAME` - Specify which model to use (e.g., `"openai/gpt-4.1"`)
- `PREPARED_DATASETS_DIR` - Path to prepared datasets inside container
- `OPENROUTER_API_KEY` - Required for LLM access
- `WANDB_API_KEY` - Optional for experiment tracking

**Interactive Mode:**
To run without specifying dataset/model (interactive selection):

```bash
docker run -it --rm \
    -v $(pwd):/repository \
    -v $(pwd)/prepared_datasets:/repository/prepared_datasets \
    -v agentomics-workspace:/workspace \
    -e OPENROUTER_API_KEY="your-key-here" \
    -e WANDB_API_KEY="your-wandb-key" \
    -e PREPARED_DATASETS_DIR="/repository/prepared_datasets" \
    --name agentomics-agent \
    agentomics:latest
```

## API Keys Setup

The `./run.sh` script will interactively prompt for API keys, but you can also set them as environment variables:

```bash
export OPENROUTER_API_KEY="your-openrouter-key"    # Required
export WANDB_API_KEY="your-wandb-key"              # Optional
./run.sh
```

**Required:**

- `OPENROUTER_API_KEY` - Get from https://openrouter.ai

**Optional:**

- `WANDB_API_KEY` - Get from https://wandb.ai for experiment tracking and cost analysis
  - If not provided: Agent runs normally but saves results locally only
  - If provided: Enables experiment tracking, run comparison, and cost monitoring via Weave
- `WANDB_ENTITY` - Override default Wandb entity (default: ceitec-ai)
- `WANDB_PROJECT` - Override default Wandb project (default: Agentomics-ML)
- `WEAVE_PROJECT` - Override default Weave project (default: {WANDB_ENTITY}/{WANDB_PROJECT})

## API Key Setup

Agentomics-ML requires API keys for model access and experiment tracking. You can set these up in several ways:

### Option 1: Using .env File (Recommended)

Create a `.env` file in the project root with your API keys:

```bash
# Copy the example file
cp .env.example .env

# Edit the file with your actual API keys
nano .env  # or use your preferred editor
```

**Example `.env.example` file:**

```bash
# Required: OpenRouter API Key for model access and listing
# Get your key from: https://openrouter.ai/keys
OPENROUTER_API_KEY=your_openrouter_api_key_here

# Optional: Weights & Biases API Key for experiment tracking
# Get your key from: https://wandb.ai/settings
WANDB_API_KEY=your_wandb_api_key_here
```

The script will automatically load these environment variables when needed.

**Why use a .env file?** This is a standard practice in software development for managing environment variables and API keys. The `.env` file allows you to store sensitive configuration (like API keys) separately from your code, making it easier to manage different environments and keeping secrets out of version control. This pattern is widely used in frameworks like Django, Rails, Node.js, and many others. See [12-factor app methodology](https://12factor.net/config) for more details on this practice.

**Security Note**: We provide `.env.example` (not `.env`) to prevent accidental API key commits. The `.env` file is in `.gitignore` to ensure your real API keys are never committed to version control.

### Option 2: Environment Variables

Set the API keys as environment variables in your shell:

```bash
# Required for model access
export OPENROUTER_API_KEY="your-openrouter-api-key"

# Optional for experiment tracking
export WANDB_API_KEY="your-wandb-api-key"
```

### Option 3: Interactive Prompts

If no API keys are found, the script will prompt you interactively to enter them.

### Getting API Keys

- **OpenRouter API Key**: Get your key from [https://openrouter.ai/keys](https://openrouter.ai/keys)
- **WANDB API Key**: Get your key from [https://wandb.ai/settings](https://wandb.ai/settings)

## Environment Variables (Advanced)

For proper cost tracking with Weave and PydanticAI, you need to set up the following environment variables:

### Required for Weave Cost Tracking

```bash
# Required: Your Wandb API key
export WANDB_API_KEY="your-wandb-api-key"

# Standard Wandb/Weave variables (used to auto-construct project ID)
export WANDB_ENTITY="your-entity"
export WANDB_PROJECT="your-project"

# Optional: Manual Weave project override (if not set, will use WANDB_ENTITY/WANDB_PROJECT)
# export WEAVE_PROJECT="your-entity/your-project"
```

### Weave Dependencies

The system uses Weave for LLM cost and performance tracking. All required dependencies are included in the environment.yaml file.

### Troubleshooting Cost Tracking

If cost data is not appearing in your Weave dashboard:

1. **Check Environment Variables**: Ensure `WANDB_API_KEY` is set correctly, and either `WANDB_ENTITY`/`WANDB_PROJECT` or `WEAVE_PROJECT`
2. **Project Format**: If using manual `WEAVE_PROJECT`, should be in format `"entity/project"` (with quotes)
3. **API Key**: Use your actual Wandb API key, not a placeholder
4. **LLM Provider**: Ensure your LLM provider (OpenRouter, OpenAI, etc.) supports cost tracking

The system will display setup status during initialization:

- ✅ Weave Cost Tracking: Enabled - Cost tracking is working
- ❌ Weave Cost Tracking: Disabled - Check environment variables

## Prepare your dataset

### Add your files

Create a folder for your dataset in the `datasets` folder and add your files. Follow the `datasets/sample_dataset` structure. For easiest use follow these rules:

- Name your files exactly `train.csv`, `test.csv` and `dataset_description.md`
- In your csv files, provide a target column that will contain labels or values:
  - **Classification**: Column with categorical labels (strings or integers like `"positive"`, `"negative"` or `0`, `1`)
  - **Regression**: Column with continuous numeric values (like `125.50`, `1847.2`)
  - **Column Names**: Preferably use `class`, `target`, `label`, or `y` for auto-detection, but any column name works (system will use the last column as fallback)

Possible customizations:

<!-- - providing `test.csv` is optional. Without it, test-set metrics will not be provided to the user at the end of the run. TODO implement -->

- providing `dataset_description.md` is optional. Without it, the agent will be slightly limited but functional.

### Dataset Preparation (Automatic)

**Dataset preparation is now automatic!** When you run `./run.sh`, it automatically detects and prepares any unprepared datasets in your `./datasets/` directory.

**Smart Auto-Detection:**
The preparation system automatically detects both the target column and task type:

**Target Column Detection (in priority order):**

1. Looks for columns named: `class`, `target`, `label`, or `y`
2. Falls back to the **last column** if none of the above are found
3. Works with any column name (e.g., `electricity_cost`, `price`, `outcome`)

**Task Type Detection:**

- **Regression**: Numeric targets with >10 unique values
  - Examples: `125.50`, `1847.2`, `0.95` (continuous values)
  - Use cases: Price prediction, energy consumption, stock prices
- **Classification**: All other cases
  - Categorical: `"positive"`, `"negative"`, `"class_A"`, `"class_B"`
  - Numeric with ≤10 unique values: `0`, `1`, `2` (discrete categories)
  - Use cases: Disease diagnosis, sentiment analysis, image classification

**Real Examples:**

- `electricity_cost_dataset` → Target: `electricity_cost` (numeric, >10 values) → **Regression**
- `sample_dataset` → Target: `class` (categorical) → **Classification**
- `heart_disease` → Target: `diagnosis` (0/1) → **Classification**

This means you can simply place your CSV files in `./datasets/your_dataset/` and the system will figure out the rest!

**Manual preparation (optional):**
If you want to prepare datasets manually or need custom settings, you can still use:

```bash
# Prepare all datasets at once
./run.sh --prepare-only                      # Uses Docker preparation
./run.sh --local --prepare-only              # Uses local Python preparation

# Or prepare individual datasets
python src/utils/dataset_preparation.py --dataset-dir datasets/sample_dataset
```

**Advanced options:**

- Custom output directory: `--output-dir <your/path/prepared_datasets>`
- Custom label column: `--class-col your_label_column`
- Binary classification labels: `--positive-class yes --negative-class no`
- Get help: `python src/utils/prepare_dataset.py --help`

### Manual Commands (Alternative to ./run.sh)

If your label column has a different name than `target`, or you want to specify your own label mapping for binary datasets, run `python src/utils/prepare_dataset.py --help` for more info. 
For users who prefer full control or need to run individual steps manually, here are the complete command sequences:

**Prerequisites:**

```bash
# Set your API keys
export OPENROUTER_API_KEY="your-openrouter-key"
export WANDB_API_KEY="your-wandb-key"  # Optional

# Activate the conda environment
source activate agentomics-env
# OR if using conda run:
# conda run -n agentomics-env [command]
```

**Classification Example (sample_dataset):**

```bash
# Step 1: Prepare the dataset (with explicit task type)
source activate agentomics-env && python src/utils/dataset_preparation.py --dataset-dir datasets/sample_dataset --task-type classification

# Step 2: Run the agent
source activate agentomics-env && python src/run_agent.py --dataset-name sample_dataset --model "openai/gpt-4.1" --val-metric AUROC
```

**Regression Example (electricity_cost_dataset):**

```bash
# Step 1: Prepare the dataset (with explicit task type)
source activate agentomics-env && python src/utils/dataset_preparation.py --dataset-dir datasets/electricity_cost_dataset --task-type regression

# Step 2: Run the agent
source activate agentomics-env && python src/run_agent.py --dataset-name electricity_cost_dataset --model "openai/gpt-4.1" --val-metric RMSE
```

**Auto-Detection Example (recommended):**

```bash
# Step 1: Prepare the dataset (auto-detects task type)
source activate agentomics-env && python src/utils/dataset_preparation.py --dataset-dir datasets/your_dataset

# Step 2: Run the agent (auto-detects appropriate metrics)
source activate agentomics-env && python src/run_agent.py --dataset-name your_dataset --model "openai/gpt-4.1"
```

**Using conda run (alternative syntax):**

```bash
# Classification
conda run -n agentomics-env python src/utils/dataset_preparation.py --dataset-dir datasets/sample_dataset --task-type classification
conda run -n agentomics-env python src/run_agent.py --dataset-name sample_dataset --model "openai/gpt-4.1" --val-metric AUROC

# Regression
conda run -n agentomics-env python src/utils/dataset_preparation.py --dataset-dir datasets/electricity_cost_dataset --task-type regression
conda run -n agentomics-env python src/run_agent.py --dataset-name electricity_cost_dataset --model "openai/gpt-4.1" --val-metric RMSE
```

**Jupyter Notebook/Colab Commands:**

```python
# For Jupyter/Colab environments (use ! prefix)
!source activate agentomics-env && python src/utils/dataset_preparation.py --dataset-dir datasets/sample_dataset --task-type classification
!source activate agentomics-env && python src/run_agent.py --dataset-name sample_dataset --model "openai/gpt-4.1" --val-metric AUROC

# Regression example
!source activate agentomics-env && python src/utils/dataset_preparation.py --dataset-dir datasets/electricity_cost_dataset --task-type regression
!source activate agentomics-env && python src/run_agent.py --dataset-name electricity_cost_dataset --model "openai/gpt-4.1" --val-metric RMSE
```

**All Available Metrics:**

```bash
# Classification metrics: ACC, AUROC, AUPRC
python src/run_agent.py --dataset-name sample_dataset --model "openai/gpt-4.1" --val-metric ACC
python src/run_agent.py --dataset-name sample_dataset --model "openai/gpt-4.1" --val-metric AUPRC

# Regression metrics: MSE, RMSE, MAE, R2
python src/run_agent.py --dataset-name electricity_cost_dataset --model "openai/gpt-4.1" --val-metric MSE
python src/run_agent.py --dataset-name electricity_cost_dataset --model "openai/gpt-4.1" --val-metric MAE
python src/run_agent.py --dataset-name electricity_cost_dataset --model "openai/gpt-4.1" --val-metric R2
```

**Manual Task Type Control:**

For cases where you want to override auto-detection, both the preparation script and `./run.sh` support explicit task type specification:

```bash
# Force classification (even if data looks like regression)
./run.sh -d your_dataset -m "openai/gpt-4.1" --task-type classification

# Force regression (even if data looks like classification)
./run.sh -d your_dataset -m "openai/gpt-4.1" --task-type regression

# Let auto-detection work (recommended)
./run.sh -d your_dataset -m "openai/gpt-4.1"
```

**Note:** The `--task-type` parameter is optional everywhere - the system auto-detects based on your data when not specified:

```bash
# Auto-detection (recommended)
python src/utils/dataset_preparation.py --dataset-dir datasets/your_dataset
python src/run_agent.py --dataset-name your_dataset --model "openai/gpt-4.1"
```

## Run the agent

### Unified Script (Recommended)

The easiest way to run Agentomics-ML is using the unified `./run.sh` script which handles everything automatically:

```bash
# Docker mode (default) - fully interactive
./run.sh

# Local mode - fully interactive (faster for development)
./run.sh --local

# The script will:
# 1. Check/prompt for API keys (.env file, environment, or interactive prompts)
# 2. Let you select from available datasets and models
# 3. Automatically prepare datasets if needed
# 4. Run the agent with optimal settings (Docker container or local conda environment)
# 5. Provide clear output about where results are saved
```

**Choose Your Mode:**

| Mode       | When to Use                                   | Pros                                                                              | Cons                                                          |
| ---------- | --------------------------------------------- | --------------------------------------------------------------------------------- | ------------------------------------------------------------- |
| **Docker** | Production, reproducibility, first-time users | ✅ Isolated environment<br/>✅ No setup required<br/>✅ Consistent across systems | ❌ Slower startup<br/>❌ Harder debugging                     |
| **Local**  | Development, testing, debugging               | ✅ Fast startup<br/>✅ Easy debugging<br/>✅ Live code editing                    | ❌ Requires conda setup<br/>❌ Environment conflicts possible |

**Result Location:**

- **With WANDB_API_KEY**: Results tracked online at https://wandb.ai/ceitec-ai/Agentomics-ML
- **Without WANDB_API_KEY**: Results saved locally in `workspace/runs/[agent-id]` directory

### Advanced Usage (Docker)

For manual Docker usage, once your container is running via `./run.sh`:

```bash
# Attach to the running container (conda environment auto-activates)
docker exec -it agentomics-agent bash

# Run the agent (replace sample_dataset with your dataset name)
python src/run_agent.py --dataset-name sample_dataset
```

### Direct Python Execution

If running in your own environment:

```bash
python src/run_agent.py --dataset-name sample_dataset
```

## Available Models

Agentomics-ML dynamically fetches and filters the best coding and reasoning models from leading AI providers through the OpenRouter API.

**Top Models Include:**

- **OpenAI:** `openai/gpt-4o`, `openai/gpt-4o-mini`, `openai/gpt-4.1`, `openai/o1-preview`
- **Anthropic:** `anthropic/claude-3.5-sonnet`, `anthropic/claude-3-opus`
- **Google:** `google/gemini-1.5-pro`, `google/gemini-1.5-flash`
- **Meta:** `meta-llama/llama-3.1-405b-instruct`, `meta-llama/llama-3.1-70b-instruct`

```bash
./run.sh --list-models          # See all available models with pricing (Docker)
./run.sh --local --list-models  # See all available models with pricing (Local)
```

## Validation Metrics

**Classification:** `ACC` (Accuracy), `AUROC` (ROC), `AUPRC` (Precision-Recall)  
**Regression:** `MSE`, `RMSE`, `MAE`, `R2`

```bash
# Specify metric (optional - auto-selected based on task type)
./run.sh -d dataset -m model --metric AUROC  # Classification
./run.sh -d dataset -m model --metric RMSE   # Regression
```

## Results & Tracking

### Local Results

Results are always saved to `workspace/runs/[run-id]/` with:

- Trained model files
- Performance metrics
- Inference scripts
- Execution logs

### Online Tracking (Optional)

With `WANDB_API_KEY` set, get enhanced tracking:

- 📊 **Experiment Dashboard**: https://wandb.ai/ceitec-ai/Agentomics-ML
- 🔍 **LLM Traces**: Detailed interaction logs via Weave
- 💰 **Cost Tracking**: Token usage and costs per experiment
- 📈 **Performance Comparison**: Compare multiple runs

**Logging Features:**

- 🤖 **Immediate LLM Interactions**: See model requests and responses as they happen
- 📊 **Step-by-Step Progress**: Track each phase (Data Exploration → Model Architecture → Training → Validation)
- ⏱️ **Timestamped Steps**: Know exactly when each phase starts and ends
- 🎯 **Iteration Tracking**: Clear indication of which iteration is running and its status
- 📈 **Performance Updates**: Real-time metrics and best performance tracking
- ✅ **Success/Failure Indicators**: Visual status for each step and overall completion

**🔗 Weave-Wandb Integration & LLM Tracing:**

Agentomics-ML integrates [Weights & Biases Weave](https://wandb.ai/site/weave) with your Wandb experiments for **unified observability**:

- 🔗 **Automatic Linking**: Weave traces are automatically linked to your Wandb runs for seamless navigation
- 🔍 **Automatic Tracing**: All LLM interactions are automatically logged and traced
- 🌲 **Hierarchical Traces**: View complete execution trees showing data flow through agent iterations
- 💰 **Comprehensive Cost Tracking**: Monitor token usage and costs across all LLM providers with detailed breakdowns
- 🐛 **Debugging Tools**: Interactive UI for exploring LLM inputs, outputs, and intermediate steps
- 📊 **Performance Analytics**: Latency analysis and success rate monitoring
- 🔄 **Agent Flow Visualization**: See how data flows through each agent (exploration → architecture → training → validation)
- 🏷️ **Step & Iteration Tracking**: Every step is tagged with both step number and iteration for complete traceability
- 🎯 **Cross-Platform Navigation**: Jump between experiment dashboards and detailed trace analysis

**Unified Dashboard Experience:**
When you run the agent, traces are automatically linked to your Wandb run. You'll see integration info like:

```
🔗 Weave-Wandb Integration Active:
   • Wandb Project: ceitec-ai/Agentomics-ML
   • Weave Project: ceitec-ai/Agentomics-ML
   • View traces: https://weave.wandb.ai/ceitec-ai/Agentomics-ML
   • View run: https://wandb.ai/ceitec-ai/Agentomics-ML/runs/xyz123
```

Access your traces at: `https://weave.wandb.ai/` or directly from your Wandb run pages. Traces are organized by dataset name for easy navigation.

**💰 Comprehensive Cost Tracking:**

The system uses Weave's native cost tracking with stacked bar chart visualization:

- **Native Weave Integration**: Uses built-in cost tracking with automatic LLM provider detection
- **Stacked Bar Charts**: Each iteration shows total cost/tokens broken down by steps
- **Step-Level Breakdown**: See which steps (Data Exploration, Model Training, etc.) consume most resources
- **Iteration Comparison**: Compare costs across iterations to identify efficiency trends
- **Token Usage**: Monitors input/output tokens with step-level granularity
- **Consolidated Metrics**: Clean dashboard without metric clutter
- **Real-time Logging**: Costs appear in both Weave traces and Wandb dashboards
- **Provider Support**: Works with OpenAI, Anthropic, and other Weave-supported LLM providers

Example cost output:

```
💰 Iteration 1 Cost Breakdown:
   • Total Cost: $0.1234
   • Total Tokens: 5,432
   • Steps Breakdown:
     - Step 1 (data_exploration): $0.0123, 543 tokens
     - Step 3 (data_representation): $0.0456, 1,876 tokens
     - Step 4 (model_architecture): $0.0234, 1,234 tokens
     - Step 5 (model_training): $0.0421, 1,779 tokens

💰 Experiment Summary:
   • Total Cost: $0.8765
   • Total Tokens: 45,321
   • Total Requests: 67
   • Iterations: 8
   • Average Cost per Iteration: $0.1096
   • Average Tokens per Iteration: 5,665
   • Average Cost per Token: $0.000019
```

**Cost Tracking Troubleshooting:**
If you see "No cost data available in Weave traces":

- Ensure your LLM provider supports Weave cost tracking (OpenAI, Anthropic, etc.)
- Verify your API keys are properly configured
- Check that autopatch settings are enabled (done automatically by `init_weave_with_costs()`)
- Note: Some local/self-hosted models may not provide cost information

**Example workflow:**

```bash
# 1. List available datasets
./run.sh --list-datasets                    # Docker mode
./run.sh --local --list-datasets            # Local mode (shows datasets from ./datasets/ folder)

# 2. List available models
./run.sh --list-models                      # Docker mode
./run.sh --local --list-models              # Local mode

# 3. List available metrics
./run.sh --list-metrics                     # Show all metrics with descriptions

# 4. Set up conda environment (for local mode only)
conda env create -f environment.yaml       # First time only
conda activate agentomics-env              # Each session

# 5. Run with interactive selection
./run.sh                                    # Docker mode
./run.sh --local                            # Local mode

# 6. Follow prompts to select dataset, metric, and model
# 7. The script handles the rest automatically
```

### Dataset and Model Examples

Here are practical examples of running Agentomics-ML with specific datasets and models:

**Docker Mode Examples:**

```bash
# Basic run with specific dataset and model
./run.sh -d heart_disease -m "openai/gpt-4.1"

# Different models with same dataset
./run.sh -d sample_dataset -m "anthropic/claude-3.5-sonnet"
./run.sh -d sample_dataset -m "google/gemini-pro-1.5"
./run.sh -d sample_dataset -m "meta-llama/llama-3.1-70b-instruct"

# Using custom Docker images
./run.sh myusername -d heart_disease -m "openai/gpt-4.1"
./run.sh --username myorg --version v2.1 -d cancer_data -m "anthropic/claude-3.5-sonnet"

# Classification with additional parameters
./run.sh -d cancer_data -m "openai/gpt-4.1" --metric AUROC --tags experiment1,baseline

# Regression examples (continuous numeric targets)
./run.sh -d electricity_cost_dataset -m "openai/gpt-4.1" --metric RMSE --tags energy_prediction
./run.sh -d house_prices -m "openai/gpt-4.1" --metric RMSE --tags regression_baseline
./run.sh -d stock_prediction -m "anthropic/claude-3.5-sonnet" --metric MAE --tags timeseries
```

**Local Mode Examples:**

```bash
# Basic local run
./run.sh --local -d sample_dataset -m "openai/gpt-4.1"

# With custom workspace directory
./run.sh --local -d heart_disease -m "openai/gpt-4.1" --workspace-dir /custom/workspace

# Multiple experiments for comparison (classification)
./run.sh --local -d sample_dataset -m "openai/gpt-4.1" --tags gpt4_baseline
./run.sh --local -d sample_dataset -m "anthropic/claude-3.5-sonnet" --tags claude_comparison

# Regression experiments (continuous numeric targets)
./run.sh --local -d electricity_cost_dataset -m "openai/gpt-4.1" --metric MSE --tags energy_cost_prediction
./run.sh --local -d house_prices -m "openai/gpt-4.1" --metric RMSE --tags regression_gpt4
./run.sh --local -d temperature_data -m "google/gemini-pro-1.5" --metric R2 --tags weather_prediction
```

**Available Model Formats:**

- OpenAI: `"openai/gpt-4o"`, `"openai/gpt-4o-mini"`, `"openai/gpt-4.1"`, `"openai/o1-preview"`
- Anthropic: `"anthropic/claude-3.5-sonnet"`, `"anthropic/claude-3-opus"`
- Google: `"google/gemini-1.5-pro"`, `"google/gemini-1.5-flash"`
- Meta: `"meta-llama/llama-3.1-405b-instruct"`, `"meta-llama/llama-3.1-70b-instruct"`

**Note:** Models are dynamically fetched from OpenRouter API. Use `./run.sh --list-models` to see current available models with pricing.

**Dataset Requirements:**

- Dataset folder in `./datasets/your_dataset_name/`
- Required file: `train.csv` with a target column (any name, auto-detected)
  - **Classification**: categorical labels or numeric with ≤10 unique values
  - **Regression**: numeric values with >10 unique values
- Optional files: `test.csv`, `dataset_description.md`
- **Auto-Detection**: Automatically finds target column (`class`, `target`, `label`, `y`, or last column) and determines task type

### Classification vs Regression Examples

Here are real examples showing how the system automatically detects task types:

**Classification Example (`sample_dataset`):**

```csv
# train.csv structure
feature1,feature2,feature3,class
0.5,1.2,0.8,positive
0.3,2.1,0.6,negative
0.7,1.8,0.9,positive
```

- ✅ **Target Column**: `class` (auto-detected)
- ✅ **Task Type**: Classification (categorical values: `positive`, `negative`)
- ✅ **Metrics**: ACC, AUROC, AUPRC
- ✅ **Command**: `./run.sh -d sample_dataset -m "openai/gpt-4.1" --metric AUROC`

**Regression Example (`electricity_cost_dataset`):**

```csv
# train.csv structure
feature1,feature2,feature3,electricity_cost
0.5,1.2,0.8,125.45
0.3,2.1,0.6,187.92
0.7,1.8,0.9,203.78
```

- ✅ **Target Column**: `electricity_cost` (auto-detected, not standard name)
- ✅ **Task Type**: Regression (numeric values >10 unique: `125.45`, `187.92`, `203.78`, ...)
- ✅ **Metrics**: MSE, RMSE, MAE, R2
- ✅ **Command**: `./run.sh -d electricity_cost_dataset -m "openai/gpt-4.1" --metric RMSE`

**Auto-Detection Logic:**

1. **Target Column Search**: `class` → `target` → `label` → `y` → last column
2. **Task Type Analysis**:
   - Count unique values in target column
   - If numeric & >10 unique values → **Regression**
   - Otherwise → **Classification**
3. **No Manual Configuration Required**: Just place your CSV files and run!

### Output and Monitoring

- **Files**: Agent outputs are stored in the `/workspace/runs/` directory inside the container
- **Monitoring**: If you provided a WANDB_API_KEY, monitor experiments at https://wandb.ai/ceitec-ai/Agentomics-ML
- **Logs**: View real-time logs with `docker logs -f agentomics-agent`
- **Duration**: Runs can take several hours depending on dataset complexity

### Validation Metrics

Agentomics-ML supports both classification and regression tasks with appropriate metrics:

**Classification Metrics:**

- `ACC` (Accuracy) - Percentage of correctly classified instances
- `AUPRC` (Area Under Precision-Recall Curve) - Good for imbalanced datasets
- `AUROC` (Area Under ROC Curve) - Overall classification performance

**Regression Metrics:**

- `MSE` (Mean Squared Error) - Average squared differences
- `RMSE` (Root Mean Squared Error) - Square root of MSE, same units as target
- `MAE` (Mean Absolute Error) - Average absolute differences
- `R2` (R-squared) - Explained variance ratio (0-1, higher is better)

**Interactive Metric Selection:**

The system intelligently detects your dataset's task type and shows only relevant metrics:

- **Classification datasets**: Shows only ACC, AUPRC, AUROC
- **Regression datasets**: Shows only MSE, RMSE, MAE, R2
- **Auto-detection**: Automatically suggests the best default metric for your task type
- **Smart defaults**: ACC for classification, R2 for regression

**List Available Metrics:**

```bash
./run.sh --list-metrics         # Show all available metrics with descriptions
python agentomics-entrypoint.py --list-metrics  # Direct Python execution
```

**Usage Examples:**

```bash
# Classification with different metrics (categorical targets)
./run.sh -d heart_disease -m "openai/gpt-4.1" --metric ACC     # Binary: 0/1 diagnosis
./run.sh -d cancer_data -m "openai/gpt-4.1" --metric AUROC     # Multi-class: benign/malignant
./run.sh -d sample_dataset -m "openai/gpt-4.1" --metric AUPRC  # Auto-detects class column

# Regression with different metrics (continuous numeric targets)
./run.sh -d electricity_cost_dataset -m "openai/gpt-4.1" --metric MSE   # Energy cost prediction
./run.sh -d house_prices -m "openai/gpt-4.1" --metric RMSE             # Price prediction
./run.sh -d stock_prediction -m "openai/gpt-4.1" --metric R2            # Financial forecasting
./run.sh -d temperature_data -m "openai/gpt-4.1" --metric MAE           # Weather prediction
```

### Advanced Options

```bash
# Run without root privileges (for restricted environments)
python src/run_agent.py --dataset-name sample_dataset --no-root-privileges

# Custom prepared datasets directory
python src/run_agent.py --dataset-name sample_dataset --prepared-datasets-dir /custom/path

# Custom Wandb/Weave configuration
python src/run_agent.py --dataset-name sample_dataset --wandb-entity myorg --wandb-project myproject

# Custom Weave project (independent of Wandb)
python src/run_agent.py --dataset-name sample_dataset --weave-project myorg/custom-weave-project

# See all options
python src/run_agent.py --help
```

### Configuration

To modify agent behavior (LLM temperature, timeouts, etc.), edit `src/utils/config.py`

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

## Known Limitations & Future Improvements

### Target Column Selection Limitation

**Current Behavior:**

- The system auto-detects task type (classification vs regression) based on the first suitable target column found
- For regression datasets, users can select which numeric column to use as the target
- However, if a dataset has both categorical and numeric columns that could serve as targets, the system doesn't let users switch between classification and regression modes

**Example Scenario:**

```csv
house_id,size,bedrooms,neighborhood,price,sold_within_30_days
1,1200,2,"Downtown",250000,1
2,1800,3,"Suburbs",350000,0
```

This dataset could be used for:

- **Regression**: Predict `price` (continuous target) → Task type: regression
- **Classification**: Predict `sold_within_30_days` (binary target) → Task type: classification

**Current Limitation:**

- System detects one task type and only shows appropriate columns for that task
- Users cannot easily switch to use the same dataset for a different ML task type

### Single-Target Regression Only

**Current Behavior:**

- The system only supports single-target regression (predicting one continuous variable)
- Multi-target regression (predicting multiple continuous outputs simultaneously) is not supported

**Future Enhancement:**

- Support for multi-target regression scenarios
- Enhanced evaluation metrics for multi-dimensional predictions

## Developer guidelines and tips

Whatever packages you add, add them to the environment.yaml file

If they are conda packages, you can just re-run
`conda env export --from-history > environment.yaml`

If they are pip packages, you have to manually add them.

You can update your existing environment by running this command
`conda env update --file environment.yaml --prune`
