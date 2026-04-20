# Agent Architecture

Agentomics-ML uses a multi-step iterative architecture where each iteration refines the ML solution.

## The Iteration Loop

Each iteration consists of 7 sequential steps:

```
┌─────────────────────────────────────────────────────────────┐
│                       ITERATION N                            │
├─────────────────────────────────────────────────────────────┤
│  1. Data Exploration    → Analyze data characteristics       │
│  2. Data Split          → Create train/validation split      │
│  3. Data Representation → Define feature encoding            │
│  4. Model Architecture  → Select and configure model         │
│  5. Training            → Train the model                    │
│  6. Inference           → Create prediction pipeline         │
│  7. Prediction Analysis → Evaluate and analyze results       │
├─────────────────────────────────────────────────────────────┤
│                    ↓ Feedback Agent ↓                        │
│              (Guides next iteration)                         │
└─────────────────────────────────────────────────────────────┘
                           ↓
                    ITERATION N+1
```

## Step Details

### 1. Data Exploration

The agent analyzes the dataset:

- Feature distributions and statistics
- Missing values and data quality
- Correlations between features
- Class balance (for classification)
- Domain-specific insights

**Output:** Captured in structured outputs and iteration reports

### 2. Data Split

Creates or modifies train/validation split:

- Stratified splitting for classification
- Considers data distribution
- May adjust split based on previous iterations

**Output:** Captured in structured outputs and iteration reports

!!! note
    The agent can modify splits during early iterations (controlled by `--split-allowed-iterations`).

### 3. Data Representation

Defines how data is encoded:

- Numeric normalization/standardization
- Categorical encoding (one-hot, label encoding)
- Feature selection
- Custom transformations for omics data

**Output:** Captured in structured outputs and iteration reports

### 4. Model Architecture

Selects and configures the model:

- Model type (sklearn, neural network, etc.)
- Hyperparameters
- Architecture details for neural networks
- Regularization strategies

**Output:** Captured in structured outputs and iteration reports

### 5. Training

Implements and runs training:

- Generates `train.py` script
- Sets up conda environment
- Executes training
- Monitors for issues (overfitting, convergence)

**Output:** `train.py` and training artifacts

### 6. Inference

Creates prediction pipeline:

- Generates `inference.py` script
- Ensures reproducibility
- Handles data preprocessing
- Validates predictions

**Output:** `inference.py` script

### 7. Prediction Analysis

Evaluates model performance:

- Calculates validation metrics
- Analyzes prediction distributions
- Identifies potential issues
- Compares to previous iterations

**Output:** Metrics and analysis

## Agent Tools

The agent has access to these tools:

| Tool | Purpose |
|------|---------|
| **Bash** | Execute shell commands |
| **Write Python** | Create Python files |
| **Run Python** | Execute Python scripts |
| **Replace** | Modify code sections |
| **Foundation Models Info** | Get domain model information |

## How Decisions Are Made

Each step uses an LLM agent that:

1. Receives context (data info, previous results, feedback)
2. Generates a structured output (validated by Pydantic)
3. Validates the output meets requirements
4. Retries if validation fails (up to 10 times)

## Iteration Flow

```
Iteration 0 (Baseline)
├── Full exploration
├── Initial split
├── Basic representation
├── Simple model
└── Baseline metrics

Iteration 1-N (Refinement)
├── Focused exploration (guided by feedback)
├── Split adjustment (if allowed)
├── Improved representation
├── Better model/hyperparameters
└── Compare to best so far

Final
├── Select best iteration
├── Evaluate on test set
└── Generate reports
```

## Snapshot System

The agent tracks the best-performing iteration:

- After each iteration, metrics are compared
- If better than previous best, a snapshot is saved
- Snapshots include model, scripts, and environment
- Final evaluation uses the best snapshot

## Configuration

Key architecture parameters in `src/agentomics/utils/config.py`:

| Parameter | Default | Description |
|-----------|---------|-------------|
| `temperature` | 1.0 | LLM creativity level |
| `max_steps` | 100 | Max steps per agent |
| `max_validation_retries` | 10 | Output validation retries |
| `llm_response_timeout` | 900s | LLM response timeout |
| `bash_tool_timeout` | 300s | Bash command timeout |
| `run_python_tool_timeout` | 21600s | Training timeout (6 hours, configurable via `--run-python-timeout`) |

## Next Steps

- [Feedback Loop](feedback-loop.md) - How iterations improve
- [Evaluation](evaluation.md) - Metrics and testing
