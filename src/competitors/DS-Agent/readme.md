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
```

## Step 2: Download and Extract Case Data (Critical)
```bash
# Download benchmark and case data from Google Drive
# Go to https://drive.google.com/file/d/1xUd1nvCsMLfe-mv9NBBHOAtuYnSMgBGx/view?usp=sharing

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
cp /path/to/your/test.csv MLAgentBench/benchmarks/dna_classification/env/

# Create initial train.py in env/ directory with dummy content
cat > MLAgentBench/benchmarks/dna_classification/env/train.py << 'EOF'
import pandas as pd
import numpy as np
from sklearn.metrics import roc_auc_score
import os
import sys

# Function to make random predictions (baseline)
def random_guess(n_samples):
    return np.random.random(n_samples)

def main():
    print("Loading DNA sequence data...")
    
    # Load training data
    train_data = pd.read_csv("train.csv")
    print(f"Loaded {len(train_data)} training samples")
    
    # Basic data inspection
    print("\nData columns:", train_data.columns.tolist())
    print("\nClass distribution:")
    print(train_data['class'].value_counts())
    
    # Check sequence properties
    seq_lengths = train_data['sequence'].str.len()
    print(f"\nSequence length: min={seq_lengths.min()}, max={seq_lengths.max()}, mean={seq_lengths.mean():.1f}")
    
    # Make random predictions (baseline model)
    print("\nGenerating random baseline predictions...")
    y_true = train_data['class'].values
    y_pred = random_guess(len(train_data))
    
    # Evaluate baseline
    baseline_auc = roc_auc_score(y_true, y_pred)
    print(f"Baseline model AUC: {baseline_auc:.4f}")
    
    print("\nInitial script complete. Ready for DS-Agent to improve.")

if __name__ == "__main__":
    main()
EOF

# Create submission.py in env/ directory with dummy content
cat > MLAgentBench/benchmarks/dna_classification/env/submission.py << 'EOF'
import pandas as pd
import numpy as np
from sklearn.metrics import roc_auc_score
import sys

def evaluate(test_csv, prediction_csv):
    """
    Evaluate predictions using AUC score.
    
    Args:
        test_csv: Path to the test data with true labels
        prediction_csv: Path to the predictions file with 'prediction' column
        
    Returns:
        AUC score
    """
    # Load test data with ground truth
    test_data = pd.read_csv(test_csv)
    
    # Load predictions
    predictions = pd.read_csv(prediction_csv)
    
    if 'prediction' not in predictions.columns:
        raise ValueError("Prediction file must contain 'prediction' column")
    
    # Calculate AUC
    auc = roc_auc_score(test_data['class'], predictions['prediction'])
    
    return auc

if __name__ == "__main__":
    if len(sys.argv) != 3:
        print("Usage: python submission.py <test_csv> <predictions_csv>")
        sys.exit(1)
    
    test_csv = sys.argv[1]
    prediction_csv = sys.argv[2]
    
    try:
        auc = evaluate(test_csv, prediction_csv)
        print(f"AUC Score: {auc:.4f}")
    except Exception as e:
        print(f"Error during evaluation: {e}")
        sys.exit(1)
EOF
```

## Step 6: Create Script Files (in scripts/ directory)
```bash
# Create research_problem.txt in scripts/ directory
cat > MLAgentBench/benchmarks/dna_classification/scripts/research_problem.txt << EOF
Classify DNA sequences as non-TATA promoters (class=1) or non-promoters (class=0). The sequences are 251 nucleotides long containing 'A', 'G', 'T', 'C', and 'N'. The goal is to build a model with high AUC score.
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

# Save the file
```

## Step 10: Run DS-Agent
```bash
cd MLAgentBench
python runner.py --task dna_classification --llm-name gpt-4o-2024-08-06 --edit-script-llm-name gpt-4o-2024-08-06
```

## Step 11: Find Results
```bash
# Results will be saved in:
ls -la ./logs/  # For logs of the process
ls -la ./workspace/  # For generated solution files
```

## Important Notes:
1. The case-based reasoning data is essential for DS-Agent to function properly
2. The dummy scripts provide minimal functionality as a starting point
3. Make sure your train.csv and test.csv files contain 'sequence' and 'class' columns
4. The GPU configuration fix is critical if you have a GPU and want to use it for case-based reasoning
5. If you continue to have GPU issues, you can run with `--no-retrieval true` flag, but this will limit the agent's capabilities


# What happened step-by-step
Stage | Contents of train.py | Outcome
Before DSAgent | Your own “default” script (never executed). | Never run, so no metrics from it.
Agent edit #1–2 | Place-holder lines like Here is the python code. and an Optuna template. | Immediate SyntaxError or ModuleNotFoundError; no AUC.
Agent edit #3 | Fresh Logistic-Regression script the agent wrote (read data → StratifiedKFold). | First successful run → mean AUC 0.7836.
Agent edit #4 | Same model with GridSearchCV tuning. | Best AUC 0.7839.
Agent edit #5 | Added calibration via a Pipeline. | Best AUC 0.7840.
