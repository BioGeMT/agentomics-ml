#!/bin/bash
set -e # Exit immediately if a command exits with a non-zero status.

# Helper functions for colored output
echo_green() { echo -e "\033[0;32m$1\033[0m"; }
echo_yellow() { echo -e "\033[1;33m$1\033[0m"; }
echo_red() { echo -e "\033[0;31m$1\033[0m"; }

# --- Configuration ---
SELA_CONDA_ENV_PATH="/tmp/sela_env"
METAGPT_FORK_DIR="/tmp/MetaGPT_fork_sela"
METAGPT_REPO_URL="https://github.com/davidcechak/MetaGPT.git"
SCRIPT_SOURCE_DIR="$( cd "$( dirname "${BASH_SOURCE[0]}" )" &> /dev/null && pwd )"
UTILS_FIXED_SOURCE="$SCRIPT_SOURCE_DIR/utils_fixed.py"
DATASET_FIXED_SOURCE="$SCRIPT_SOURCE_DIR/dataset_fixed.py"

# --- 0. Clean Up Previous Environment ---
echo_yellow "Step 0: Checking for and removing existing SELA conda environment..."
if [ -d "$SELA_CONDA_ENV_PATH" ]; then rm -rf "$SELA_CONDA_ENV_PATH"; echo_green " Removed $SELA_CONDA_ENV_PATH."; else echo_green " No previous env found."; fi

# --- 1. Create and Activate Conda Environment ---
echo_green "Step 1: Creating conda environment $SELA_CONDA_ENV_PATH..."; conda create -p "$SELA_CONDA_ENV_PATH" python=3.9 -y
if [ $? -ne 0 ]; then echo_red "Failed create env."; exit 1; fi
echo_green "Activating conda environment..."; source /opt/conda/etc/profile.d/conda.sh; conda activate "$SELA_CONDA_ENV_PATH"
if [ $? -ne 0 ]; then echo_red "Failed activate env."; exit 1; fi; echo_green " Env activated."
PYTHON_EXEC="$SELA_CONDA_ENV_PATH/bin/python"; PIP_EXEC="$SELA_CONDA_ENV_PATH/bin/pip"
if [ ! -x "$PYTHON_EXEC" ]; then echo_red "ERROR: Python not found: $PYTHON_EXEC"; exit 1; fi
if [ ! -x "$PIP_EXEC" ]; then echo_red "ERROR: Pip not found: $PIP_EXEC"; exit 1; fi

# --- 2. MetaGPT Installation (Clone/Pull & Patch using cp) ---
echo_green "Step 2: Handling MetaGPT fork ($METAGPT_REPO_URL)..."
if [ -d "$METAGPT_FORK_DIR/.git" ]; then echo_yellow " Existing repo found. Pulling..."; (cd "$METAGPT_FORK_DIR" && git pull origin main || git pull); echo_green " Pull attempted."; else echo_yellow " Cloning fresh..."; if [ -d "$METAGPT_FORK_DIR" ]; then rm -rf "$METAGPT_FORK_DIR"; fi; git clone "$METAGPT_REPO_URL" "$METAGPT_FORK_DIR"; if [ $? -ne 0 ]; then echo_red "Clone failed."; exit 1; fi; echo_green " Cloned successfully."; fi

# ---> Patch files using cp <---
UTILS_TARGET_PATH="$METAGPT_FORK_DIR/metagpt/ext/sela/utils.py"
DATASET_TARGET_PATH="$METAGPT_FORK_DIR/metagpt/ext/sela/data/dataset.py"

echo_yellow "Copying corrected utils.py..."
if [ -f "$UTILS_FIXED_SOURCE" ]; then cp "$UTILS_FIXED_SOURCE" "$UTILS_TARGET_PATH"; echo_green " Overwrote $UTILS_TARGET_PATH"; else echo_red "ERROR: $UTILS_FIXED_SOURCE not found."; exit 1; fi
echo_yellow "Copying corrected dataset.py..."
if [ -f "$DATASET_FIXED_SOURCE" ]; then cp "$DATASET_FIXED_SOURCE" "$DATASET_TARGET_PATH"; echo_green " Overwrote $DATASET_TARGET_PATH"; else echo_red "ERROR: $DATASET_FIXED_SOURCE not found."; exit 1; fi
# ---> End of patching using cp <---

# ---> Append dataset config to datasets.yaml <---
DATASETS_YAML_PATH="$METAGPT_FORK_DIR/metagpt/ext/sela/datasets.yaml"
DATASET_NAME_TO_ADD="human_nontata_promoters"
TARGET_COL_TO_ADD="class"
METRIC_TO_ADD="f1_binary" # Assuming binary classification

echo_yellow "Appending config for '$DATASET_NAME_TO_ADD' to $DATASETS_YAML_PATH..."
if [ -f "$DATASETS_YAML_PATH" ] && grep -q "  $DATASET_NAME_TO_ADD:" "$DATASETS_YAML_PATH"; then
    echo_yellow " Entry for '$DATASET_NAME_TO_ADD' already exists. Skipping append."
else
    echo "" >> "$DATASETS_YAML_PATH"; echo "  $DATASET_NAME_TO_ADD:" >> "$DATASETS_YAML_PATH"
    echo "    dataset: $DATASET_NAME_TO_ADD" >> "$DATASETS_YAML_PATH"
    echo "    metric: $METRIC_TO_ADD" >> "$DATASETS_YAML_PATH"
    echo "    target_col: $TARGET_COL_TO_ADD" >> "$DATASETS_YAML_PATH"
    USER_REQ_TO_ADD="\"This is a $DATASET_NAME_TO_ADD dataset. Your goal is to predict the target column \`$TARGET_COL_TO_ADD\`. Perform data analysis, data preprocessing, feature engineering, and modeling to predict the target. Report $METRIC_TO_ADD on the eval data. Do not plot or make any visualizations.\""
    echo "    user_requirement: $USER_REQ_TO_ADD" >> "$DATASETS_YAML_PATH"
    echo_green " Appended config for '$DATASET_NAME_TO_ADD'."
fi
# ---> End of appending to datasets.yaml <---

# Install the patched MetaGPT fork
echo_green "Installing MetaGPT from $METAGPT_FORK_DIR (uses patched files)..."
cd "$METAGPT_FORK_DIR"; if [ $? -ne 0 ]; then echo_red "Failed cd $METAGPT_FORK_DIR."; exit 1; fi
"$PIP_EXEC" install .; if [ $? -ne 0 ]; then echo_red "pip install . failed."; exit 1; fi
echo_green "Patched MetaGPT installed successfully."
echo "Verifying metagpt command..."; if ! command -v metagpt &> /dev/null; then echo_yellow "Warn: 'metagpt' cmd not found."; else echo_green "'metagpt' cmd available: $(which metagpt)"; fi

# --- 3. Install SELA-specific requirements ---
SELA_EXT_REQUIREMENTS_PATH="$METAGPT_FORK_DIR/metagpt/ext/sela/requirements.txt"
echo_green "Step 3: Checking SELA requirements at $SELA_EXT_REQUIREMENTS_PATH..."
if [ -f "$SELA_EXT_REQUIREMENTS_PATH" ]; then echo_green "Installing SELA requirements..."; "$PIP_EXEC" install -r "$SELA_EXT_REQUIREMENTS_PATH"; if [ $? -ne 0 ]; then echo_red "Failed install SELA reqs."; else echo_green "SELA reqs installed."; fi
else echo_yellow "SELA requirements.txt not found. Skipping."; fi

# --- 4. Install other dependencies with verification ---
echo_green "Step 4: Installing other dependencies with verification..."
install_and_verify() {
    local pkg="$1"; local mod="$2"; local ver="${3:-}"; local opts="";
    echo_yellow "Install/Verify: $pkg$ver..."; if "$PIP_EXEC" install $opts "$pkg$ver"; then echo_green " $pkg OK."; else echo_red " $pkg install FAILED."; exit 1; fi
    echo " Verifying $mod import..."; if "$PYTHON_EXEC" -c "import $mod" &> /dev/null; then VER=$("$PYTHON_EXEC" -c "import $mod; print(getattr($mod,'__version__','N/A'))"); echo_green " $mod import OK. Ver: $VER"; else echo_red " $mod import FAILED."; exit 1; fi
}
install_and_verify "protobuf" "google.protobuf" "<5"; install_and_verify "pyyaml" "yaml" "==6.0.1"; install_and_verify "hrid" "hrid" "==0.2.4"; install_and_verify "agentops" "agentops" "==0.4.9"; install_and_verify "wandb" "wandb"; install_and_verify "python-dotenv" "dotenv"; install_and_verify "pandas" "pandas"; install_and_verify "scikit-learn" "sklearn"; install_and_verify "openml" "openml"
echo_green "All specified dependencies processed and verified."

# --- 5. Setup /tmp/sela working directory ---
SELA_WORK_DIR="/tmp/sela"; echo_green "Step 5: Setting up $SELA_WORK_DIR..."; cd /tmp; mkdir -p "$SELA_WORK_DIR"; cd "$SELA_WORK_DIR"
if [ $? -ne 0 ]; then echo_red "Failed cd $SELA_WORK_DIR."; exit 1; fi; echo_green "CWD: $(pwd)"; echo "Initializing MetaGPT config..."; conda activate "$SELA_CONDA_ENV_PATH"
if metagpt --init-config; then echo_green "MetaGPT config init OK."; else echo_yellow "Warning: metagpt --init-config failed."; fi

# --- 6. Copy SELA utility scripts (run.py, set_config.py, etc.) ---
echo_green "Step 6: Copying SELA runtime utilities from $SCRIPT_SOURCE_DIR to $SELA_WORK_DIR..."
for script_file in run.py set_config.py overwrite_data_yaml.sh; do
    src="$SCRIPT_SOURCE_DIR/$script_file"; dst="$SELA_WORK_DIR/$script_file"
    if [ -f "$src" ]; then cp "$src" "$dst"; echo " Copied $script_file."; if [[ "$script_file" == *.sh ]]; then chmod +x "$dst"; echo "  Made executable."; fi; else echo_red " Warning: Source $src not found."; fi
done; echo_green "Utility scripts copied."

# --- 7. Set permissions ---
echo_green "Step 7: Ensuring permissions for $SELA_WORK_DIR & $SELA_CONDA_ENV_PATH..."; chmod -R 777 "$SELA_WORK_DIR" || echo_yellow "Warn: chmod $SELA_WORK_DIR failed."; chmod -R 777 "$SELA_CONDA_ENV_PATH" || echo_yellow "Warn: chmod $SELA_CONDA_ENV_PATH failed."

echo_green "------------------ SETUP COMPLETE (Using Patched SELA Library Files) ------------------"
