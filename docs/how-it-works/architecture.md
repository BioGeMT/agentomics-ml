# Agent Architecture

Agentomics-ML uses a multi-step iterative architecture where each iteration refines the ML solution.

## The Iteration Loop

Each iteration consists of 9 sequential steps:

```
┌─────────────────────────────────────────────────────────────┐
│                       ITERATION N                            │
├─────────────────────────────────────────────────────────────┤
│  1. Iteration Plan      → Create step-by-step guidance       │
│  2. Data Exploration    → Analyze data characteristics       │
│  3. Data Split          → Create train/validation split      │
│  4. Data Representation → Define feature encoding            │
│  5. Model Architecture  → Select and configure model         │
│  6. Model Training      → Train the model                    │
│  7. Model Inference     → Create prediction pipeline         │
│  8. Prediction Exploration → Analyze validation predictions  │
│  9. Validation Evaluation → Score the iteration              │
└─────────────────────────────────────────────────────────────┘
```

## Step Details

### 1. Iteration Plan

The iteration-planning step analyzes archived results and produces concrete instructions for the rest of the iteration:

- Reviews prior step outputs and validation metrics
- Accounts for split-version changes and time budget
- Decides whether to explore new combinations or refine promising ones
- Produces per-step instructions in `IterationPlanOutput`

**Output:** Structured iteration-plan instructions stored with the iteration

### 2. Data Exploration

The agent analyzes the dataset:

- Feature distributions and statistics
- Missing values and data quality
- Correlations between features
- Class balance (for classification)
- Domain-specific insights

**Output:** Captured in structured outputs and iteration reports

### 3. Data Split

Creates, reuses, or modifies train/validation split files:

- Stratified splitting for classification
- Considers data distribution
- May adjust split based on previous iterations
- Reuses supplied `validation.csv` when present

**Output:** Captured in structured outputs and iteration reports

!!! note
    The agent can modify splits during early iterations (controlled by `--split-allowed-iterations`).

### 4. Data Representation

Defines how data is encoded:

- Numeric normalization/standardization
- Categorical encoding (one-hot, label encoding)
- Feature selection
- Custom transformations for omics data

**Output:** Captured in structured outputs and iteration reports

### 5. Model Architecture

Selects and configures the model:

- Model type (sklearn, neural network, etc.)
- Hyperparameters
- Architecture details for neural networks
- Regularization strategies

**Output:** Captured in structured outputs and iteration reports

### 6. Model Training

Implements and runs training:

- Generates `model_training/train.py`
- Sets up conda environment
- Executes training
- Monitors for issues (overfitting, convergence)

**Output:** `model_training/train.py` and sibling `training_artifacts/`

### 7. Model Inference

Creates prediction pipeline:

- Generates `model_inference/inference.py`
- Ensures reproducibility
- Handles data preprocessing
- Validates predictions

**Output:** `model_inference/inference.py`

### 8. Prediction Exploration

Analyzes the validation-set predictions produced by the current inference pipeline:

- Generates prediction summaries
- Analyzes prediction distributions
- Identifies potential issues
- Surfaces patterns worth addressing in later iterations

**Output:** Structured prediction analysis and iteration reports

### 9. Validation Evaluation

Runs the final validation pass for the iteration:

- Executes the generated inference pipeline on train and validation splits
- Computes metrics used for model selection
- Decides whether the iteration becomes the new best iteration

**Output:** Validation metrics and best-iteration status

## Agent Tools

The agent has access to these tools:

| Tool | Purpose |
|------|---------|
| **Bash** | Execute shell commands |
| **Write Python** | Create Python files |
| **Run Python** | Execute Python scripts |
| **Replace** | Modify code sections |

## How Decisions Are Made

Each step uses an LLM agent that:

1. Receives context (data info, previous results, and the iteration plan when applicable)
2. Generates a structured output (validated by Pydantic)
3. Validates the output meets requirements
4. Retries if validation fails (up to 5 times by default)

## Iteration Flow

```
Iteration 0 (Baseline)
├── Iteration planning
├── Full exploration
├── Initial split
├── Basic representation
├── Simple model
└── Baseline validation metrics

Iteration 1-N (Refinement)
├── Iteration planning from archived results
├── Focused exploration (guided by the iteration plan)
├── Split adjustment (if allowed)
├── Improved representation
├── Better model/hyperparameters
└── Compare to best so far

Final
├── Select best iteration
└── Generate reports
```

## Snapshot System

The agent tracks the best-performing iteration:

- After each iteration, metrics are compared
- If better than previous best, a snapshot is saved
- Snapshots include model, scripts, and environment
- The best snapshot is exported to `best_iteration_snapshot/` for inference and re-training

## Configuration

Key architecture parameters in `src/utils/config.py`:

| Parameter | Default | Description |
|-----------|---------|-------------|
| `temperature` | 0.7 | LLM creativity level |
| `max_steps` | 100 | Max steps per agent |
| `max_validation_retries` | 5 | Output validation retries |
| `llm_response_timeout` | 600s | LLM response timeout |
| `bash_tool_timeout` | 180s | Bash command timeout |
| `run_python_tool_timeout` | 21600s | Training timeout (6 hours, configurable via `--run-python-timeout`) |

## Next Steps

- [Iteration Planning](iteration-planning.md) - How iterations improve
- [Evaluation](evaluation.md) - Metrics and testing
