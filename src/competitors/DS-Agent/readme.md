# Complete Step-by-Step Plan to Adapt DS-Agent for DNA Sequence Classification

## Step 1: Environment Setup
```bash
# Navigate to the development directory
cd /path/to/DS-Agent/development

# Install DS-Agent and dependencies
pip install -e .
pip install -r requirements.txt

# Fix OpenAI API compatibility issue (critical)
pip install openai==0.28

pip install tensorflow
pip install keras
```

## Step 2: Download and Extract Case Data (Critical)
```bash
# Extract data.zip to data folder (contains case-based reasoning data)
unzip data.zip -d data/

# If necessary, create a biological sequence case directory
mkdir -p data/cases/biological_sequence
```

## Step 3: Create Proper Benchmark Directory Structure
```bash
# Create the main benchmark directory (from development directory)
mkdir -p MLAgentBench/benchmarks/dna_classification

# Create required subdirectories
mkdir -p MLAgentBench/benchmarks/dna_classification/env
mkdir -p MLAgentBench/benchmarks/dna_classification/scripts
```

## Step 4: Create Configuration File
```bash
# Create config.json in the main benchmark directory
cat > MLAgentBench/benchmarks/dna_classification/config.json << EOF
{
    "task_name": "dna_classification",
    "task_type": "classification",
    "data_modality": "biological_sequence",
    "evaluation_metric": "auc",
    "higher_is_better": true,
    "time_limit_hours": 2,
    "description": "Classify DNA sequences as non-TATA promoters (class=1) or non-promoters (class=0)"
}
EOF
```

## Step 5: Create Environment Files with Dummy Scripts
```bash
# Copy dataset files to env/ directory
cp /path/to/your/train.csv MLAgentBench/benchmarks/dna_classification/env/

# Create initial scripts in env/ directory
touch MLAgentBench/benchmarks/dna_classification/env/train.py


## Step 6: Create Script Files (in scripts/ directory)
```bash
# Create research_problem.txt in scripts/ directory
cat > MLAgentBench/benchmarks/dna_classification/scripts/research_problem.txt << EOF
Build the best classifier that generates the new unseen data. End produce much be the inference data file at logs/*
EOF

# Create required 'prepared' empty file in scripts/ directory
touch MLAgentBench/benchmarks/dna_classification/scripts/prepared
```

## Step 7: Register Your Task in tasks.json
```bash
# Create or update tasks.json file
cat > MLAgentBench/benchmarks/tasks.json << EOF
{
  "dna_classification": {
    "benchmark_folder_name": "dna_classification",
    "research_problem_file": "research_problem.txt"
  }
}
EOF
```

## Step 8: Configure OpenAI API Key
```bash
# Edit LLM.py to add your OpenAI API key
nano MLAgentBench/LLM.py
# Replace "FILL IN YOUR KEY HERE." with your actual key
```

## Step 9: Fix GPU Configuration (Critical)
```bash
# Edit the retrieval.py file to use the correct CUDA device
nano MLAgentBench/retrieval.py

# Find the line that sets self.device (around line 16-18)
# Change:
# self.device = torch.device("cuda:1" if torch.cuda.is_available() else "cpu")
# To:
# self.device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
```

## Step 10: Run DS-Agent
```bash
cd MLAgentBench
python runner.py --task dna_classification --llm-name gpt-4o-2024-08-06 --edit-script-llm-name gpt-4o-2024-08-06
```

## Step 11: Find Results
```bash
# Results will be saved in:
ls -la ./logs/{date}/env_log/traces/step_final_files  # For logs of the process
ls ./DS-Agent/development/MLAgentBench/workspace/{date}/dna_classification/  # For generated solution files
```
