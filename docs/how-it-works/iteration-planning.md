# Iteration Planning

The iteration-plan step guides improvements between iterations.

## How It Works

At the start of each iteration:

1. **Archived Results Loaded** - Metrics, model outputs, split history, and analysis from prior iterations
2. **Plan Generated** - The iteration-planning model analyzes what worked and what did not
3. **Instructions Created** - Specific guidance is produced for the remaining steps in the current iteration
4. **Step Agents Run** - Later steps receive the iteration plan as part of their context

## Iteration-Plan Structure

The iteration-planning step produces instructions for each step:

```
IterationPlanOutput
├── data_exploration_instructions
├── data_split_instructions
├── data_representation_instructions
├── model_architecture_instructions
├── model_training_instructions
├── model_inference_instructions
├── prediction_exploration_instructions
└── other_instructions
```

## What Iteration Planning Considers

### Performance Metrics

- Current vs. previous validation metrics
- Best iteration so far
- Metric trends across iterations

### Issues Detected

- Overfitting (train >> validation performance)
- Underfitting (poor performance on both)
- Class imbalance effects
- Feature problems

### Model Behavior

- Training convergence
- Prediction distribution
- Error patterns

### Time Constraints

- Remaining time (if timeout set)
- Training duration of previous models

## Iteration-Plan Examples

### Overfitting Detected

> **Model Architecture Instructions:**
> The previous model overfit significantly (train ACC 0.98, val ACC 0.72).
> Try stronger regularization: increase dropout to 0.5, add L2 regularization,
> or use a simpler architecture with fewer layers.

### Underfitting Detected

> **Model Architecture Instructions:**
> The model underfits (train ACC 0.65, val ACC 0.63).
> Consider a more complex model: deeper network, more trees,
> or additional features in the representation.

### Good Progress

> **Model Training Instructions:**
> Good improvement from 0.75 to 0.82 AUROC.
> Continue with similar architecture but try fine-tuning
> learning rate (try 1e-4) and increasing epochs.

## Exploration vs. Optimization

The iteration-planning step manages two phases:

### Exploration Phase

Early iterations (controlled by `--exploration-iterations`):

- Try diverse approaches
- Test different model types
- Explore data representations
- Build baseline understanding

### Optimization Phase

Later iterations:

- Focus on best-performing approaches
- Fine-tune hyperparameters
- Improve on winning strategy
- Polish the solution

## Combination Tracking

The iteration-planning step tracks which combinations have been tried:

```
Representation × Architecture Combinations:
✓ StandardScaler + LogisticRegression → 0.72 ACC
✓ StandardScaler + RandomForest → 0.78 ACC
✓ PCA + RandomForest → 0.75 ACC
✗ MinMaxScaler + GradientBoosting → Not tried yet
```

This prevents repeating failed approaches and encourages exploration.

## Split Management

For early iterations, the agent may suggest split changes:

> **Data Split Instructions:**
> Validation set may be too small for reliable metrics.
> Consider increasing validation size to 30% for better estimates.

After `--split-allowed-iterations`, or after `--split-timeout` when set, splits are frozen to ensure fair comparison.

## Iteration-Planning Model

The iteration-planning step uses `--iteration-plan-model` when provided. If it is omitted, it uses the same model as `--model`.

## Viewing Iteration Plans

Iteration-plan guidance is included in iteration reports:

```
reports/markdown/run_report_iter_N.md
```

Each report shows:
- What iteration plan was received
- How the agent responded
- Results of the iteration

## Configuration

| Parameter | Default | Description |
|-----------|---------|-------------|
| `--exploration-iterations` | 4 | Iterations for broad exploration |
| `--split-allowed-iterations` | 1 | Iterations that can modify splits |
| `--split-timeout` | None | Time deadline after which splits cannot change |

## Next Steps

- [Agent Architecture](architecture.md) - Full iteration loop
- [Evaluation](evaluation.md) - How metrics are calculated
