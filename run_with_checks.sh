#!/usr/bin/env bash

# Unified run script for Agentomics-ML
# Supports both Docker and local execution modes
set -euo pipefail

# Colors for output
RED='\033[0;31m'
GREEN='\033[0;32m'
YELLOW='\033[1;33m'
BLUE='\033[0;34m'
PURPLE='\033[0;35m'
CYAN='\033[0;36m'
NC='\033[0m' # No Color

# Configuration
SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
SRC_DIR="$SCRIPT_DIR/src"
DATASETS_DIR="$SCRIPT_DIR/datasets"
WORKSPACE_DATASETS_DIR="$SCRIPT_DIR/workspace/datasets"
WORKSPACE_RUNS_DIR="$SCRIPT_DIR/workspace/runs"
PREPARED_DATASETS_DIR="$SCRIPT_DIR/prepared_datasets"

# Docker configuration
USERNAME=""
TAG="latest"
IMAGE_NAME="agentomics"
CONTAINER_NAME="agentomics-agent"

# Available validation metrics
VALIDATION_METRICS=("ACC" "AUPRC" "AUROC")

# Execution modes
LOCAL_MODE=false

# Load environment variables early
load_environment_variables() {
    print_colored "$BLUE" "Loading environment variables..."
    
    # Load from .env file if it exists
    if [[ -f "$SCRIPT_DIR/.env" ]]; then
        print_colored "$CYAN" "Loading environment variables from .env file..."
        export $(grep -v '^#' "$SCRIPT_DIR/.env" | xargs)
        print_colored "$GREEN" "Environment variables loaded from .env file"
    else
        print_colored "$YELLOW" "No .env file found. Using system environment variables."
    fi
}

# Function to check and prepare datasets (delegated to prepare_datasets_docker)

# Function to print colored output
print_colored() {
    local color=$1
    local message=$2
    echo -e "${color}${message}${NC}"
}

# Function to check if we have an interactive TTY
check_tty() {
    if [[ ! -t 0 ]]; then
        return 1  # No TTY available
    fi
    return 0
}

# Function to validate TTY requirement for interactive operations
require_tty_for_interactive() {
    local operation="$1"
    local dataset="${2:-}"
    local model="${3:-}"
    
    # Check if we need interactivity
    local needs_interactive=false
    
    if [[ -z "$dataset" || -z "$model" ]]; then
        needs_interactive=true
    fi
    
    # If we need interactivity but don't have TTY, exit with error
    if [[ "$needs_interactive" == true ]] && ! check_tty; then
        print_colored "$RED" "❌ Interactive TTY required for $operation but not available"
        print_colored "$CYAN" "💡 For non-interactive use, specify both --dataset and --model arguments"
        print_colored "$CYAN" "   Example: $0 --dataset heart_disease --model 'openai/gpt-4'"
        if [[ "$LOCAL_MODE" != true ]]; then
            print_colored "$CYAN" "💡 Ensure you're running in an interactive terminal with: $0 --local"
        fi
        return 1
    fi
    
    return 0
}

# Function to show usage
show_usage() {
    echo "Usage: $0 [options] [username] [tag]"
    echo ""
    echo "🐳 Agentomics-ML Docker Runner (Simplified)"
    echo "============================================"
    echo ""
    echo "Docker Image Arguments:"
    echo "       $0                      # Uses local agentomics:latest, falls back to cloudmark/agentomics:latest"
    echo "       $0 username             # Uses username/agentomics:latest from DockerHub"
    echo "       $0 username v1.0        # Uses username/agentomics:v1.0 from DockerHub"
    echo "       $0 --username myuser --version v2.0  # Explicit Docker image specification"
    echo ""
    echo "Options:"
    echo "  -h, --help                  Show this help message"
    echo "  -d, --dataset NAME          Dataset name (will prompt if not provided)"
    echo "  -m, --model NAME            Model name (will prompt if not provided)"
    echo "  --local                     Run in local mode (faster for development)"
    echo "  --conda-env NAME            Conda environment to use in local mode (default: agentomics-env)"
    echo "  --username USER             Docker Hub username (default: cloudmark)"
    echo "  --version TAG               Docker image version/tag (default: latest)"
    echo "  --metric METRIC             Validation metric (ACC, AUPRC, AUROC)"
    echo "  --task-type TYPE            Force task type (classification, regression) instead of auto-detection"
    echo "  --target-column COLUMN      Target column for regression (skips interactive selection)"
    echo "  --tags TAG1,TAG2            Comma-separated tags for wandb"
    echo "  --list-datasets             List available datasets and exit"
    echo "  --list-models               List top AI models with pricing and exit"
    echo "  --list-volumes              List Docker volumes and exit"
    echo "  --prepare-only              Only prepare datasets, don't run the agent"
    echo ""
    echo "Environment Variables:"
    echo "  OPENROUTER_API_KEY          Required for agent operation, optional for --list-models (live pricing)"
    echo "  WANDB_API_KEY               Optional for experiment tracking and logging"
    echo "  WANDB_ENTITY                Override default Wandb entity (ceitec-ai)"
    echo "  WANDB_PROJECT               Override default Wandb project (Agentomics-ML)"
    echo "  WEAVE_PROJECT               Override default Weave project ({WANDB_ENTITY}/{WANDB_PROJECT})"
    echo ""
    echo "Examples:"
    echo "  # Basic usage:"
    echo "  $0                                      # Interactive mode (Docker: cloudmark/agentomics:latest)"
    echo "  $0 --local                              # Interactive mode (Local: conda environment)"
    echo "  $0 myuser                               # Download and use myuser/agentomics:latest"
    echo "  $0 cloudmark v2.1                       # Download and use cloudmark/agentomics:v2.1"
    echo "  $0 --username myuser --version v1.5     # Explicit Docker image specification"
    echo ""
    echo "  # Direct execution:"
    echo "  $0 -d heart_disease -m \"openai/gpt-4.1\"     # Docker mode with parameters"
    echo "  $0 --local -d heart_disease -m \"openai/gpt-4.1\" # Local mode with parameters"
    echo "  $0 --username myuser -d sample_dataset   # Custom Docker image + dataset"
    echo ""
    echo "  # Utility commands:"
    echo "  $0 --list-datasets                      # Show available datasets (Docker)"
    echo "  $0 --local --list-datasets              # Show available datasets (Local)"
    echo "  $0 --list-models                        # Show available models (Docker)"
    echo "  $0 --local --list-models                # Show available models (Local)"
    echo "  $0 --list-volumes                       # Show Docker volumes"
    echo "  $0 --prepare-only                       # Only prepare datasets (Docker)"
    echo "  $0 --local --prepare-only               # Only prepare datasets (Local)"
    echo ""
    echo "Prerequisites:"
    echo "  • Docker mode: Docker must be installed and running"
    echo "  • Local mode: Conda environment with environment.yaml dependencies"
    echo "  • OPENROUTER_API_KEY environment variable or .env file"
    echo ""
}





# Function to prompt for API key with validation
prompt_for_api_key() {
    local var_name="$1"
    local description="$2"
    local is_required="$3"
    local current_value="${!var_name:-}"
    
    if [ -n "$current_value" ]; then
        echo "$var_name already set"
        return 0
    fi
    
    echo ""
    echo "$description"
    if [ "$is_required" = "true" ]; then
        echo "   This is REQUIRED for the agent to work."
    else
        echo "   This is OPTIONAL (press Enter to skip)."
    fi
    echo ""
    
    while true; do
        if [ "$is_required" = "true" ]; then
            read -p "Enter your $var_name: " -r api_key
        else
            read -p "Enter your $var_name (optional): " -r api_key
        fi
        
        # If optional and empty, that's fine
        if [ "$is_required" = "false" ] && [ -z "$api_key" ]; then
            echo "   Skipping $var_name"
            break
        fi
        
        # If required and empty, ask again
        if [ "$is_required" = "true" ] && [ -z "$api_key" ]; then
            echo "   Error: $var_name is required. Please enter a value."
            continue
        fi
        
        # If we have a value, validate it looks like an API key
        if [ -n "$api_key" ]; then
            if [[ ${#api_key} -lt 10 ]]; then
                echo "   Warning: API key seems too short. Please check and try again."
                continue
            fi
            # Export the variable
            export "$var_name"="$api_key"
            echo "   $var_name set successfully"
            break
        fi
    done
}



# Function to select or create workspace volume (Docker mode only)
select_workspace_volume() {
    local default_volume="agentomics-workspace"
    local selected=""
    echo ""
    # Use rich UI for volume selection
    if command -v python3 >/dev/null 2>&1 && python3 -c "import rich" >/dev/null 2>&1; then
        local sel_output
        sel_output=$(python3 "$SCRIPT_DIR/src/utils/volume_manager.py" --select-only)
        if [[ -n "$sel_output" ]]; then
            selected="$sel_output"
        else
            print_colored "$RED" "Volume selection cancelled"
            exit 1
        fi
    else
        print_colored "$RED" "Error: Python 3 with rich library is required for volume selection"
        print_colored "$CYAN" "Please install rich: pip install rich"
        exit 1
    fi

    WORKSPACE_VOLUME="$selected"
    print_colored "$GREEN" "Using workspace volume: $WORKSPACE_VOLUME"
    
    # Debug: check if volume name contains problematic characters
    if [[ "$WORKSPACE_VOLUME" == *":"* ]]; then
        print_colored "$YELLOW" "Warning: Volume name contains colon, this may cause issues"
    fi
}

# Function to check if a dataset is prepared
check_dataset_prepared() {
    local dataset_path="$1"
    local dataset_name=$(basename "$dataset_path")
    local prepared_path="./prepared_datasets/$dataset_name"
    [ -f "$prepared_path/metadata.json" ] && [ -f "$prepared_path/train.csv" ]
}



# Function to prepare datasets in Docker mode
prepare_datasets_docker() {
    print_colored "$BLUE" "Preparing datasets for Docker mode..."
    echo ""
    
    # Check if datasets directory exists
    if [ ! -d "./datasets" ]; then
        print_colored "$YELLOW" "No datasets directory found. Creating ./datasets/"
        mkdir -p "./datasets"
        echo ""
        print_colored "$CYAN" "Add your datasets to ./datasets/ directory"
        echo "   Each dataset should be in its own folder with train.csv"
        return 0
    fi
    
    # Check if we have any datasets to prepare
    local has_unprepared=false
    for dataset_dir in ./datasets/*/; do
        if [ -d "$dataset_dir" ] && [ -f "$dataset_dir/train.csv" ] && ! check_dataset_prepared "$dataset_dir"; then
            has_unprepared=true
            break
        fi
    done
    
    if [ "$has_unprepared" = false ]; then
        print_colored "$GREEN" "All datasets are already prepared"
        echo ""
        return 0
    fi
    
    # Use dedicated prepare Docker image for dataset preparation
    PREPARE_IMAGE=$(get_docker_image_name "prepare" "$USERNAME" "$TAG")
    
    # Only pull if it's a remote image (contains username)
    if [[ "$PREPARE_IMAGE" == *"/"* ]]; then
        print_colored "$CYAN" "Pulling Docker image: $PREPARE_IMAGE"
        docker pull "$PREPARE_IMAGE" >/dev/null 2>&1
    else
        print_colored "$CYAN" "Using local Docker image: $PREPARE_IMAGE"
    fi
    
    # Run preparation container
    print_colored "$BLUE" "Running dataset preparation..."
    echo ""
    
    # Run preparation directly with working command
    print_colored "$BLUE" "Running dataset preparation container..."
    
    # Build environment variables
    ENV_VARS=()
    if [[ -n "$TASK_TYPE" ]]; then
        ENV_VARS+=("-e" "TASK_TYPE=$TASK_TYPE")
    fi
    
    # Run the container directly with the known working command
    if docker run --rm \
        ${ENV_VARS[@]+"${ENV_VARS[@]}"} \
        -v "$(pwd):/repository" \
        -v "$(pwd)/datasets:/repository/datasets" \
        -v "$(pwd)/prepared_datasets:/repository/prepared_datasets" \
        "$PREPARE_IMAGE" \
        conda run -n agentomics-env python -m src.utils.dataset_preparation --batch; then
        print_colored "$GREEN" "Dataset preparation completed successfully"
    else
        print_colored "$RED" "Dataset preparation failed"
        exit 1
    fi
    
    echo ""
}



# Helper function to save API key to .env file
save_api_key_to_env() {
    local key_name="$1"
    local key_value="$2"
    
    if [[ -f "$SCRIPT_DIR/.env" ]]; then
        # Update existing .env file
        if grep -q "^${key_name}=" "$SCRIPT_DIR/.env"; then
            # Key exists, update it
            sed -i.bak "s/^${key_name}=.*/${key_name}=${key_value}/" "$SCRIPT_DIR/.env"
        else
            # Key does not exist, append it
            echo "${key_name}=${key_value}" >> "$SCRIPT_DIR/.env"
        fi
        print_colored "$GREEN" "${key_name} saved to .env file"
    else
        # Create new .env file
        echo "${key_name}=${key_value}" > "$SCRIPT_DIR/.env"
        print_colored "$GREEN" "${key_name} saved to new .env file"
    fi
}


# Function to get Docker image name with fallback logic
get_docker_image_name() {
    local image_type="${1:-agentomics}"
    local username="${2:-}"
    local tag="${3:-latest}"
    
    # For agentomics image
    if [[ "$image_type" == "agentomics" ]]; then
        # Check if local image exists
        if docker images --format "table {{.Repository}}:{{.Tag}}" | grep -q "^agentomics:latest$"; then
            echo "agentomics:latest"
            return 0
        else
            # Use cloudmark as default if username is empty
            local final_username="${username:-cloudmark}"
            echo "${final_username}/agentomics:${tag}"
            return 0
        fi
    fi
    
    # For prepare image
    if [[ "$image_type" == "prepare" ]]; then
        # Check if local prepare image exists
        if docker images --format "table {{.Repository}}:{{.Tag}}" | grep -q "^agentomics-prepare:latest$"; then
            echo "agentomics-prepare:latest"
            return 0
        else
            # Use cloudmark as default if username is empty
            local final_username="${username:-cloudmark}"
            echo "${final_username}/agentomics-prepare:${tag}"
            return 0
        fi
    fi
    
    # Default fallback
    echo "${username:-cloudmark}/agentomics:${tag}"
    return 0
}

# Function to prepare Docker image (simplified)
prepare_docker_image() {
    local image_name="$1"
    
    # Only pull if it's a remote image (contains username)
    if [[ "$image_name" == *"/"* ]]; then
        print_colored "$CYAN" "Pulling Docker image: $image_name"
        docker pull "$image_name" >/dev/null 2>&1
    else
        print_colored "$CYAN" "Using local Docker image: $image_name"
    fi
}

# Function to run a Docker command with common setup
run_docker_command() {
    local image_name="$1"
    local command="$2"
    local env_vars="${3:-}"
    local volumes="${4:-}"
    local interactive="${5:-false}"
    local container_name="${6:-}"
    local fallback_command="${7:-}"
    
    # Prepare Docker image if fallback command is provided
    if [[ -n "$fallback_command" ]]; then
        prepare_docker_image "$image_name"
    fi
    
    # Detect platform
    local platform="linux/$(uname -m)"
    if [[ "$platform" == "linux/x86_64" ]]; then
        platform="linux/amd64"
    fi
    
    # Build docker run command array
    local docker_args=("run" "--rm")
    
    # Add interactive flag if needed
    if [[ "$interactive" == "true" ]]; then
        docker_args+=("-it")
    else
        docker_args+=("-t")
    fi
    
    # Add platform
    docker_args+=("--platform" "$platform")
    
    # Add entrypoint (empty entrypoint for utility commands)
    if [[ -n "$fallback_command" ]]; then
        docker_args+=("--entrypoint" "")
    fi
    
    # Add environment variables (always include color support and API keys)
    docker_args+=("-e" "TERM=xterm-256color" "-e" "FORCE_COLOR=1")
    
    # Add API keys if they exist
    if [[ -n "${OPENROUTER_API_KEY:-}" ]]; then
        docker_args+=("-e" "OPENROUTER_API_KEY=$OPENROUTER_API_KEY")
    fi
    if [[ -n "${WANDB_API_KEY:-}" ]]; then
        docker_args+=("-e" "WANDB_API_KEY=$WANDB_API_KEY")
    fi
    
    # Add custom environment variables if provided
    if [[ -n "$env_vars" ]]; then
        # Split env_vars into individual arguments
        while IFS= read -r -d ' ' arg; do
            if [[ -n "$arg" ]]; then
                docker_args+=("$arg")
            fi
        done <<< "$env_vars "
    fi
    
    # Add volumes if provided
    if [[ -n "$volumes" ]]; then
        # Split volumes into individual arguments
        while IFS= read -r -d ' ' arg; do
            if [[ -n "$arg" ]]; then
                docker_args+=("$arg")
            fi
        done <<< "$volumes "
    fi
    
    # Add container name if provided
    if [[ -n "$container_name" ]]; then
        docker_args+=("--name" "$container_name")
    fi
    
    # Add image
    docker_args+=("$image_name")
    
    # Execute the command
    if [[ "$command" == *"bash -c"* ]]; then
        # For bash commands, split properly
        docker_args+=("bash" "-c" "${command#bash -c }")
    else
        # For simple commands, split by spaces
        while IFS= read -r -d ' ' arg; do
            if [[ -n "$arg" ]]; then
                docker_args+=("$arg")
            fi
        done <<< "$command "
    fi
    
    # Execute the command
    docker "${docker_args[@]}"
    return $?
}



# Function to check conda environment
check_conda_environment() {
    if ! command -v conda >/dev/null 2>&1; then
        print_colored "$RED" "Error: conda is not installed or not in PATH"
        print_colored "$CYAN" "Please install conda/miniconda and try again"
        return 1
    fi
    
    # Check if environment exists
    if ! conda env list | grep -q "^${CONDA_ENV} "; then
        print_colored "$YELLOW" "Conda environment '$CONDA_ENV' not found"
        print_colored "$CYAN" "Creating environment from environment.yaml..."
        if ! conda env create -f environment.yaml -n "$CONDA_ENV"; then
            print_colored "$RED" "Failed to create conda environment"
            return 1
        fi
        print_colored "$GREEN" "Conda environment '$CONDA_ENV' created successfully"
    fi
    
    return 0
}

# Function to run command in conda environment while preserving TTY
# This fixes the TTY inheritance issue with 'conda run'
run_in_conda_env() {
    local conda_env="$1"
    shift  # Remove first argument (env name)
    local cmd_args=("$@")  # Remaining arguments are the command
    
    # For interactive commands, use conda activation instead of 'conda run'
    # This preserves TTY attributes that 'conda run' can break
    if check_tty; then
        print_colored "$CYAN" "Running with conda activation (preserves TTY): ${cmd_args[*]}"
        # Use env to pass environment variables and bash -c with conda activation to preserve TTY
        env \
            OPENROUTER_API_KEY="${OPENROUTER_API_KEY:-}" \
            WANDB_API_KEY="${WANDB_API_KEY:-}" \
            WANDB_ENTITY="${WANDB_ENTITY:-}" \
            WANDB_PROJECT="${WANDB_PROJECT:-}" \
            WEAVE_PROJECT="${WEAVE_PROJECT:-}" \
            DATASET_NAME="${DATASET_NAME:-}" \
            MODEL_NAME="${MODEL_NAME:-}" \
            TARGET_COLUMN="${TARGET_COLUMN:-}" \
            bash -c "
            source \$(conda info --base)/etc/profile.d/conda.sh &&
            conda activate '$conda_env' &&
            exec \"\$@\"
        " -- "${cmd_args[@]}"
    else
        print_colored "$CYAN" "Running with conda run (non-interactive): ${cmd_args[*]}"
        # For non-interactive environments, conda run is fine and preserves env vars
        conda run -n "$conda_env" "${cmd_args[@]}"
    fi
}

# Unified execution function that routes between Docker and Conda
execute_command() {
    local command_type="$1"  # "agent", "list-datasets", "list-models", "prepare-datasets"
    shift  # Remove first argument
    local extra_args=("$@")  # Remaining arguments
    
    # For agent execution, check TTY requirements early
    if [[ "$command_type" == "agent" ]]; then
        if ! require_tty_for_interactive "$command_type" "$SELECTED_DATASET" "$SELECTED_MODEL"; then
            exit 1
        fi
    fi
    
    # Setup API keys
    print_colored "$CYAN" "Checking API Keys..."
    prompt_for_api_key "OPENROUTER_API_KEY" "OpenRouter API key for LLM access" "true"
    prompt_for_api_key "WANDB_API_KEY" "WandB API key for experiment tracking and visualization" "false"
    print_colored "$GREEN" "API key setup complete!"
    echo ""
    
    # Choose execution environment
    if [[ "$LOCAL_MODE" == true ]]; then
        if [[ ${#extra_args[@]} -gt 0 ]]; then
            execute_local "$command_type" "${extra_args[@]}"
        else
            execute_local "$command_type"
        fi
    else
        if [[ ${#extra_args[@]} -gt 0 ]]; then
            execute_docker "$command_type" "${extra_args[@]}"
        else
            execute_docker "$command_type"
        fi
    fi
}

# Execute command in local conda environment
execute_local() {
    local command_type="$1"
    shift  # Remove first argument
    local extra_args=("$@")  # Remaining arguments
    
    print_colored "$BLUE" "Running in Local mode"
    echo "=============================="
    
    # Check conda environment
    if ! check_conda_environment; then
        exit 1
    fi
    
    # Export environment variables
    export OPENROUTER_API_KEY
    export WANDB_API_KEY
    export CONDA_DEFAULT_ENV="$CONDA_ENV"
    
    # Route to appropriate command
    case "$command_type" in
        "agent")
            prepare_datasets_local
            if [[ ${#extra_args[@]} -gt 0 ]]; then
                run_agent_local "${extra_args[@]}"
            else
                run_agent_local
            fi
            ;;
        "list-datasets")
            run_in_conda_env "$CONDA_ENV" python agentomics-entrypoint.py --list-datasets
            ;;
        "list-models")
            run_in_conda_env "$CONDA_ENV" python agentomics-entrypoint.py --list-models
            ;;
        "prepare-datasets")
            prepare_datasets_local
            ;;
        *)
            print_colored "$RED" "Unknown command type: $command_type"
            exit 1
            ;;
    esac
}

# Execute command in Docker environment
execute_docker() {
    local command_type="$1"
    shift  # Remove first argument
    local extra_args=("$@")  # Remaining arguments
    
    print_colored "$BLUE" "Running in Docker mode"
    echo "=============================="
    
    # Get Docker image
    DOCKER_IMAGE=$(get_docker_image_name "agentomics" "$USERNAME" "$TAG")
    
    # Route to appropriate command
    case "$command_type" in
        "agent")
            prepare_datasets_docker
            select_workspace_volume
            if [[ ${#extra_args[@]} -gt 0 ]]; then
                run_agent_docker "${extra_args[@]}"
            else
                run_agent_docker
            fi
            ;;
        "list-datasets")
            run_command_docker "cd /repository && python -m src.utils.list_datasets --format display"
            ;;
        "list-models")
            run_command_docker "bash -c \"cd /repository && python /repository/agentomics-entrypoint.py --list-models\""
            ;;
        "prepare-datasets")
            prepare_datasets_docker
            ;;
        *)
            print_colored "$RED" "Unknown command type: $command_type"
            exit 1
            ;;
    esac
}

# Helper function to run agent in local mode
run_agent_local() {
    local extra_args=("$@")
    
    # Build command arguments
    CMD_ARGS=()
    if [[ -n "$SELECTED_DATASET" ]]; then
        CMD_ARGS+=("--dataset" "$SELECTED_DATASET")
    fi
    if [[ -n "$SELECTED_MODEL" ]]; then
        CMD_ARGS+=("--model" "$SELECTED_MODEL")
    fi
    if [[ -n "$VALIDATION_METRIC" && "$VALIDATION_METRIC" != "ACC" ]]; then
        CMD_ARGS+=("--val-metric" "$VALIDATION_METRIC")
    fi
    if [[ -n "$TARGET_COLUMN" ]]; then
        CMD_ARGS+=("--target-column" "$TARGET_COLUMN")
    fi
    
    # Add any extra arguments
    if [[ ${#extra_args[@]} -gt 0 ]]; then
        CMD_ARGS+=("${extra_args[@]}")
    fi
    
    show_execution_info
    
    # Run the entry point with TTY preservation
    if [[ ${#CMD_ARGS[@]} -gt 0 ]]; then
        print_colored "$PURPLE" "Running: agentomics-entrypoint.py ${CMD_ARGS[*]}"
        echo ""
        run_in_conda_env "$CONDA_ENV" python agentomics-entrypoint.py "${CMD_ARGS[@]}"
    else
        print_colored "$PURPLE" "Running: agentomics-entrypoint.py (interactive mode)"
        echo ""
        run_in_conda_env "$CONDA_ENV" python agentomics-entrypoint.py
    fi
    
    show_completion_message $?
}

# Helper function to run agent in Docker mode
run_agent_docker() {
    local extra_args=("$@")
    
    # Environment variables will be handled directly in the docker run command
    
    show_execution_info
    
    # Stop and remove existing container if it exists
    if docker ps -a --format 'table {{.Names}}' | grep -q "^${CONTAINER_NAME}$"; then
        echo ""
        echo "Stopping and removing existing container..."
        docker stop "$CONTAINER_NAME" 2>/dev/null || true
        docker rm "$CONTAINER_NAME" 2>/dev/null || true
    fi
    
    # Pull the Docker image if it's remote
    echo ""
    echo "Preparing Docker image..."
    if [[ "$DOCKER_IMAGE" == *"/"* ]]; then
        print_colored "$CYAN" "Pulling Docker image: $DOCKER_IMAGE"
        docker pull "$DOCKER_IMAGE" >/dev/null 2>&1
    else
        print_colored "$CYAN" "Using local Docker image: $DOCKER_IMAGE"
    fi
    
    echo ""
    print_colored "$PURPLE" "Quick Commands (in another terminal):"
    echo "  Attach to container:  docker exec -it $CONTAINER_NAME bash"
    echo "  View logs:           docker logs -f $CONTAINER_NAME"
    echo ""
    
    # Run container directly with working command
    print_colored "$BLUE" "Starting agentomics agent container..."
    
    # Build environment variables array
    ENV_VARS=()
    if [[ -n "${OPENROUTER_API_KEY:-}" ]]; then
        ENV_VARS+=("-e" "OPENROUTER_API_KEY=$OPENROUTER_API_KEY")
    fi
    if [[ -n "${WANDB_API_KEY:-}" ]]; then
        ENV_VARS+=("-e" "WANDB_API_KEY=$WANDB_API_KEY")
    fi
    if [[ -n "$SELECTED_DATASET" ]]; then
        ENV_VARS+=("-e" "DATASET_NAME=$SELECTED_DATASET")
    fi
    if [[ -n "$SELECTED_MODEL" ]]; then
        ENV_VARS+=("-e" "MODEL_NAME=$SELECTED_MODEL")
    fi
    
    # Determine if we should run in interactive mode
    DOCKER_FLAGS=("--rm" "--name" "$CONTAINER_NAME")
    if check_tty; then
        DOCKER_FLAGS+=("-it")
    else
        DOCKER_FLAGS+=("-t")
        print_colored "$YELLOW" "Warning: No TTY detected, running in non-interactive mode"
    fi
    
    # Run the container directly with overridden entrypoint
    docker run \
        --entrypoint="" \
        "${DOCKER_FLAGS[@]}" \
        ${ENV_VARS[@]+"${ENV_VARS[@]}"} \
        -v "$WORKSPACE_VOLUME:/workspace" \
        -v "$(pwd):/repository" \
        -v "$(pwd)/datasets:/workspace/datasets" \
        -v "$(pwd)/prepared_datasets:/workspace/prepared_datasets" \
        "$DOCKER_IMAGE" \
        bash -c "
        source \$(conda info --base)/etc/profile.d/conda.sh &&
        conda activate agentomics-env &&
        exec python /repository/agentomics-entrypoint.py
        "
    
    echo ""
    echo "Container stopped!"
}

# Helper function to run simple commands in local mode
run_command_local() {
    local command="$1"
    print_colored "$BLUE" "Running: python $command"
    # Split the command properly for execution
    read -ra cmd_parts <<< "$command"
    run_in_conda_env "$CONDA_ENV" python "${cmd_parts[@]}"
}

# Helper function to run simple commands in Docker mode  
run_command_docker() {
    local command="$1"
    
    # Pull image if it's remote
    if [[ "$DOCKER_IMAGE" == *"/"* ]]; then
        print_colored "$CYAN" "Pulling Docker image: $DOCKER_IMAGE"
        docker pull "$DOCKER_IMAGE" >/dev/null 2>&1
    else
        print_colored "$CYAN" "Using local Docker image: $DOCKER_IMAGE"
    fi
    
    # Run command directly in container with TTY-preserving conda activation
    docker run --rm \
        --entrypoint="" \
        -v "$(pwd):/repository" \
        "$DOCKER_IMAGE" \
        bash -c "
        source \$(conda info --base)/etc/profile.d/conda.sh &&
        conda activate agentomics-env &&
        $command
        "
}

# Helper function to show execution info
show_execution_info() {
    print_colored "$BLUE" "Starting agentomics agent..."
    
    # Show different message based on whether dataset/model are pre-configured
    if [[ -n "$SELECTED_DATASET" && -n "$SELECTED_MODEL" ]]; then
        print_colored "$GREEN" "Using pre-configured settings:"
        print_colored "$CYAN" "   Dataset: $SELECTED_DATASET"
        print_colored "$CYAN" "   Model: $SELECTED_MODEL"
    elif [[ -n "$SELECTED_DATASET" ]]; then
        print_colored "$GREEN" "Dataset pre-configured: $SELECTED_DATASET"
        print_colored "$CYAN" "Will handle model selection interactively"
    elif [[ -n "$SELECTED_MODEL" ]]; then
        print_colored "$GREEN" "Model pre-configured: $SELECTED_MODEL"
        print_colored "$CYAN" "Will handle dataset selection interactively"
    else
        print_colored "$CYAN" "Interactive mode - will handle dataset and model selection"
    fi
    
    echo ""
    print_colored "$YELLOW" "   Press Ctrl+C to stop the agent"
    echo ""
    print_colored "$CYAN" "Monitor experiments: https://wandb.ai/ceitec-ai/Agentomics-ML"
    echo ""
}

# Helper function to show completion message
show_completion_message() {
    local exit_code=$1
    echo ""
    if [[ $exit_code -eq 0 ]]; then
        print_colored "$GREEN" "Agent completed successfully!"
    else
        print_colored "$RED" "Agent exited with error code: $exit_code"
    fi
    return $exit_code
}

# Function to prepare datasets in local mode
prepare_datasets_local() {
    print_colored "$BLUE" "Preparing datasets for local mode..."
    echo ""
    
    # Check if datasets directory exists
    if [ ! -d "./datasets" ]; then
        print_colored "$YELLOW" "No datasets directory found. Creating ./datasets/"
        mkdir -p "./datasets"
        echo ""
        print_colored "$CYAN" "Add your datasets to ./datasets/ directory"
        echo "   Each dataset should be in its own folder with train.csv"
        return 0
    fi
    
    # Check if we have any datasets to prepare
    local has_unprepared=false
    for dataset_dir in ./datasets/*/; do
        if [ -d "$dataset_dir" ] && [ -f "$dataset_dir/train.csv" ] && ! check_dataset_prepared "$dataset_dir"; then
            has_unprepared=true
            break
        fi
    done
    
    if [ "$has_unprepared" = false ]; then
        print_colored "$GREEN" "All datasets are already prepared"
        echo ""
        return 0
    fi
    
    # Run preparation locally
    print_colored "$BLUE" "Running dataset preparation (local mode)..."
    echo ""
    
    # Export environment variables
    export OPENROUTER_API_KEY
    export WANDB_API_KEY
    
    # Build preparation command - run as module to handle relative imports
    # Use conda run for batch operations (non-interactive, so TTY inheritance not needed)
    PREP_CMD=("conda" "run" "-n" "$CONDA_ENV" "python" "-m" "src.utils.dataset_preparation" "--batch")
    
    # Run with error output visible
    if "${PREP_CMD[@]}"; then
        print_colored "$GREEN" "Dataset preparation completed successfully"
        return 0
    else
        print_colored "$RED" "Dataset preparation failed"
        return 1
    fi
}






list_datasets_with_status() {
    local datasets_dir="$1"
    local prepared_datasets_dir="$2"
    local title="$3"
    
    print_colored "$YELLOW" "$title"
    if [[ -d "$datasets_dir" ]]; then
        datasets=()
        while IFS= read -r -d '' dataset_dir; do
            datasets+=("$(basename "$dataset_dir")")
        done < <(find "$datasets_dir" -mindepth 1 -maxdepth 1 -type d -print0 2>/dev/null)
        
        if [[ ${#datasets[@]} -gt 0 ]]; then
            print_colored "$CYAN" "  Found ${#datasets[@]} dataset(s):"
            for dataset in "${datasets[@]}"; do
                if [[ -d "$prepared_datasets_dir/$dataset" ]]; then
                    status="Prepared"
                    status_color="$GREEN"
                else
                    status="Not prepared"
                    status_color="$YELLOW"
                fi
                print_colored "$status_color" "    $dataset: $status"
            done
        else
            print_colored "$YELLOW" "    No datasets found"
        fi
    else
        print_colored "$RED" "    Directory does not exist"
    fi
    echo ""
}

check_system_info() {
    print_colored "$YELLOW" "System Information:"
    echo "   OS: $(uname -s) $(uname -r)"
    echo "   Architecture: $(uname -m)"
    echo "   Working Directory: $(pwd)"
    echo ""
}

check_environment_variables() {
    print_colored "$YELLOW" "Environment Variables:"
    echo "   OPENROUTER_API_KEY: ${OPENROUTER_API_KEY:+'set'}"
    echo "   WANDB_API_KEY: ${WANDB_API_KEY:+'set'}"
    echo ""
}

check_directory_structure() {
    print_colored "$YELLOW" "Directory Structure:"
    
    directories=(
        "$SCRIPT_DIR/datasets:Raw datasets"
        "$PREPARED_DATASETS_DIR:Prepared datasets"
        "$WORKSPACE_DATASETS_DIR:Workspace datasets"
        "$WORKSPACE_RUNS_DIR:Workspace runs"
    )
    
    for dir_info in "${directories[@]}"; do
        IFS=":" read -r dir_path description <<< "$dir_info"
        if [[ -d "$dir_path" ]]; then
            count=$(find "$dir_path" -mindepth 1 -maxdepth 1 -type d 2>/dev/null | wc -l)
            print_colored "$GREEN" "  $description: $dir_path ($count items)"
        else
            print_colored "$RED" "  $description: $dir_path (missing)"
        fi
    done
    echo ""
}

check_python_environment() {
    print_colored "$YELLOW" "Python Environment:"
    
    if command -v python3 >/dev/null 2>&1; then
        python_version=$(python3 --version 2>&1)
        python_location=$(which python3)
        print_colored "$GREEN" "  Python: $python_version"
        print_colored "$CYAN" "  Location: $python_location"
    else
        print_colored "$RED" "  Python: Not found"
    fi
    
    if command -v conda >/dev/null 2>&1; then
        conda_version=$(conda --version 2>&1)
        conda_location=$(which conda)
        print_colored "$GREEN" "  Conda: $conda_version"
        print_colored "$CYAN" "  Location: $conda_location"
        if [[ -n "${CONDA_DEFAULT_ENV:-}" ]]; then
            print_colored "$CYAN" "  Active environment: $CONDA_DEFAULT_ENV"
        fi
    else
        print_colored "$YELLOW" "  Conda: Not available"
    fi
    echo ""
}

check_key_files() {
    print_colored "$YELLOW" "Key Files:"
    
    key_files=(
        "src/run_agent.py"
        "agentomics-entrypoint.py"
        "environment.yaml"
        ".env"
    )
    
    for file in "${key_files[@]}"; do
        if [[ -f "$SCRIPT_DIR/$file" ]]; then
            print_colored "$GREEN" "  $file: Exists"
        else
            print_colored "$RED" "  $file: Missing"
        fi
    done
    echo ""
}






# Parse command line arguments
SELECTED_DATASET=""
SELECTED_MODEL=""
VALIDATION_METRIC="ACC"
WANDB_TAGS=""
TASK_TYPE=""
TARGET_COLUMN=""
SHOW_DATASETS=false
SHOW_MODELS=false
SHOW_VOLUMES=false
PREPARE_ONLY=false
CONDA_ENV="agentomics-env"

WORKSPACE_DIR=""

# Store original arguments for positional parsing
ORIG_ARGS=("$@")

# Parse arguments
while [[ $# -gt 0 ]]; do
    case $1 in

        -h|--help)
            show_usage
            exit 0
            ;;
        -d|--dataset)
            SELECTED_DATASET="$2"
            shift 2
            ;;
        -m|--model)
            SELECTED_MODEL="$2"
            shift 2
            ;;
        --username)
            USERNAME="$2"
            shift 2
            ;;
        --version)
            TAG="$2"
            shift 2
            ;;
        --metric)
            if [[ " ${VALIDATION_METRICS[@]} " =~ " $2 " ]]; then
                VALIDATION_METRIC="$2"
            else
                print_colored "$RED" "Invalid validation metric: $2. Use one of: ${VALIDATION_METRICS[*]}"
                exit 1
            fi
            shift 2
            ;;
        --tags)
            WANDB_TAGS="$2"
            TAGS="$2"
            shift 2
            ;;
        --task-type)
            if [[ "$2" == "classification" || "$2" == "regression" ]]; then
                TASK_TYPE="$2"
            else
                print_colored "$RED" "Invalid task type: $2. Use 'classification' or 'regression'"
                exit 1
            fi
            shift 2
            ;;
        --workspace-dir)
            WORKSPACE_DIR="$2"
            shift 2
            ;;
        --list-datasets)
            SHOW_DATASETS=true
            shift
            ;;
        --list-models)
            SHOW_MODELS=true
            shift
            ;;
        --list-volumes)
            SHOW_VOLUMES=true
            shift
            ;;
        --prepare-only)
            PREPARE_ONLY=true
            shift
            ;;
        --local)
            LOCAL_MODE=true
            shift
            ;;
        --conda-env)
            CONDA_ENV="$2"
            shift 2
            ;;
        --target-column)
            TARGET_COLUMN="$2"
            shift 2
            ;;

        -*)
            echo "Unknown option: $1"
            show_usage
            exit 1
            ;;
        *)
            # Handle positional arguments (username, tag)
            if [[ ! "$1" =~ ^- ]]; then
                if [[ -z "$USERNAME" ]]; then
                    USERNAME="$1"
                elif [[ "$TAG" == "latest" ]]; then
                    TAG="$1"
                fi
            fi
            shift
            ;;
    esac
done

# Load environment variables early for all commands
load_environment_variables

# Handle list commands early
if [[ "$SHOW_DATASETS" == true ]]; then
    print_colored "$BLUE" "Listing available datasets..."
    execute_command "list-datasets"
    print_colored "$GREEN" "Dataset listing completed"
    exit 0
fi

if [[ "$SHOW_MODELS" == true ]]; then
    print_colored "$BLUE" "Listing available models..."
    execute_command "list-models"
    print_colored "$GREEN" "Model listing completed"
    exit 0
fi

# Handle prepare-only mode
if [[ "$PREPARE_ONLY" == true ]]; then
    print_colored "$BLUE" "Preparing datasets..."
    execute_command "prepare-datasets"
    print_colored "$GREEN" "Dataset preparation completed"
    exit $?
fi


# Main execution
print_colored "$BLUE" "Starting Agentomics Agent"
execute_command "agent"