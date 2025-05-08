#!/bin/bash
set -e # Exit immediately if a command exits with a non-zero status.

echo_green() {
    echo -e "\033[0;32m$1\033[0m"
}

echo_red() {
    echo -e "\033[0;31m$1\033[0m"
}

# --- 1. Create and Activate Conda Environment ---
echo_green "Step 1: Creating conda environment at /tmp/sela_env..."
conda create -p /tmp/sela_env python=3.9 -y
if [ $? -ne 0 ]; then
    echo_red "Failed to create conda environment. Exiting."
    exit 1
fi

echo_green "Activating conda environment /tmp/sela_env..."
source /opt/conda/etc/profile.d/conda.sh
conda activate /tmp/sela_env
if [ $? -ne 0 ]; then
    echo_red "Failed to activate conda environment. Exiting."
    exit 1
fi
echo_green "Conda environment /tmp/sela_env activated."

# --- 2. MetaGPT Installation (cloning to a writable /tmp location) ---
METAGPT_FORK_DIR="/tmp/MetaGPT_fork_sela" # Using a distinct name for clarity
echo_green "Step 2: Cloning MetaGPT fork (davidcechak/MetaGPT) to $METAGPT_FORK_DIR..."

if [ -d "$METAGPT_FORK_DIR" ]; then
  echo "Removing existing $METAGPT_FORK_DIR to ensure a fresh clone..."
  rm -rf "$METAGPT_FORK_DIR"
fi

git clone https://github.com/davidcechak/MetaGPT.git "$METAGPT_FORK_DIR"
if [ $? -ne 0 ]; then
    echo_red "Failed to clone MetaGPT repository. Exiting."
    exit 1
fi
echo_green "MetaGPT repository cloned successfully to $METAGPT_FORK_DIR."

echo_green "Installing MetaGPT in editable mode from $METAGPT_FORK_DIR..."
cd "$METAGPT_FORK_DIR"
if [ ! -f "setup.py" ] && [ ! -f "pyproject.toml" ]; then
    echo_red "ERROR: Cloned MetaGPT repository ($METAGPT_FORK_DIR) does not contain setup.py or pyproject.toml at its root."
    echo "Contents of $METAGPT_FORK_DIR:"
    ls -la
    exit 1
fi
pip install -e .
if [ $? -ne 0 ]; then
    echo_red "Failed to 'pip install -e .' for MetaGPT. Exiting."
    exit 1
fi
echo_green "MetaGPT installed successfully in editable mode."

# --- 3. Install SELA-specific requirements (if they are inside the MetaGPT fork) ---
SELA_EXT_REQUIREMENTS_PATH="$METAGPT_FORK_DIR/metagpt/ext/sela/requirements.txt"
echo_green "Step 3: Checking for SELA-specific requirements at $SELA_EXT_REQUIREMENTS_PATH..."
if [ -f "$SELA_EXT_REQUIREMENTS_PATH" ]; then
    echo_green "Found SELA requirements.txt. Installing..."
    pip install -r "$SELA_EXT_REQUIREMENTS_PATH"
    if [ $? -ne 0 ]; then
        echo_red "Failed to install SELA requirements from $SELA_EXT_REQUIREMENTS_PATH. Please check the file and dependencies."
        # Consider exiting if these are critical: exit 1
    else
        echo_green "SELA requirements installed successfully."
    fi
else
    echo "SELA requirements.txt not found at $SELA_EXT_REQUIREMENTS_PATH. Skipping this step."
    echo "(This is okay if SELA dependencies are installed individually below or are already covered by main MetaGPT setup)."
fi

# --- 4. Install other common dependencies for SELA ---
echo_green "Step 4: Installing other specified SELA dependencies..."
pip install agentops==0.4.9
pip install wandb
pip install python-dotenv
pip install pyyaml
pip install hrid==0.2.4
pip install pandas
pip install scikit-learn
pip install openml
echo_green "Other SELA dependencies installed."

# --- 5. Setup /tmp/sela working directory and initialize MetaGPT config ---
SELA_WORK_DIR="/tmp/sela"
echo_green "Step 5: Setting up SELA working directory at $SELA_WORK_DIR..."
mkdir -p "$SELA_WORK_DIR" # -p avoids error if directory exists
cd "$SELA_WORK_DIR"
if [ $? -ne 0 ]; then
    echo_red "Failed to cd to $SELA_WORK_DIR. Exiting."
    exit 1
fi
echo_green "Changed directory to $SELA_WORK_DIR."

echo_green "Initializing MetaGPT config in $SELA_WORK_DIR/.metagpt/ (if not already present)..."
metagpt --init-config
if [ $? -ne 0 ]; then
    echo_red "Warning: 'metagpt --init-config' failed. This might be an issue if configuration is expected here."
    echo_red "However, set_config.py might handle the configuration later."
fi

# --- 6. Copy SELA utility scripts to the working directory ---
# The set_up.sh script is located at /repository/Agentomics-ML/src/competitors/SELA/
# Other scripts (run.py, set_config.py, overwrite_data_yaml.sh) are assumed to be in the same directory.
SCRIPT_SOURCE_DIR="/repository/Agentomics-ML/src/competitors/SELA"
echo_green "Step 6: Copying SELA utility scripts from $SCRIPT_SOURCE_DIR to $SELA_WORK_DIR..."

for script_file in run.py set_config.py overwrite_data_yaml.sh; do
    if [ -f "$SCRIPT_SOURCE_DIR/$script_file" ]; then
        cp "$SCRIPT_SOURCE_DIR/$script_file" "$SELA_WORK_DIR/$script_file"
        echo "Copied $script_file to $SELA_WORK_DIR."
        if [[ "$script_file" == *.sh ]]; then
            chmod +x "$SELA_WORK_DIR/$script_file"
            echo "Made $SELA_WORK_DIR/$script_file executable."
        fi
    else
        echo_red "Warning: Script $SCRIPT_SOURCE_DIR/$script_file not found. It will not be copied."
    fi
done
echo_green "SELA utility scripts copied."

# --- 7. Set permissions for /tmp directories ---
# These directories in /tmp should generally be writable by the user running the script.
echo_green "Step 7: Ensuring permissions for /tmp/sela and /tmp/sela_env..."
chmod -R 777 "$SELA_WORK_DIR" || echo_red "Warning: chmod on $SELA_WORK_DIR failed. This might be okay if already writable."
chmod -R 777 "/tmp/sela_env" || echo_red "Warning: chmod on /tmp/sela_env failed. This might be okay if already writable."

echo_green "-----------------------------------------------------"
echo_green "SELA Setup Script Completed Successfully!"
echo_green "-----------------------------------------------------"
echo "To use the SELA environment:"
echo "1. Ensure you are in a shell where 'conda activate' is available (e.g., after 'source /opt/conda/etc/profile.d/conda.sh')."
echo "2. Activate the environment: conda activate /tmp/sela_env"
echo "3. Change to the working directory: cd $SELA_WORK_DIR"
echo "You can then run your SELA scripts (e.g., bash run.sh or python run.py)."