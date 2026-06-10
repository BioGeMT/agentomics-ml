# Evaluation

How Agentomics-ML evaluates model performance.

## Evaluation Stages

Models are evaluated at multiple stages:

| Stage | Data Used | Purpose |
|-------|-----------|---------|
| **Dry Run** | Prepared training data without labels | Validate inference script shape and metrics compatibility |
| **Validation** | Validation set | Guide optimization |
| **Train** | Training set | Detect overfitting |
| **Test** | Hidden test set | Final unbiased evaluation |

## Validation Evaluation

After each iteration:

1. Model makes predictions on validation set
2. Metrics calculated based on task type
3. Compared to previous iterations
4. Best iteration snapshot updated if improved

## Test Evaluation

At the end of the run:

1. Best iteration's model is loaded
2. Predictions made on test set (never seen during training)
3. Final metrics reported
4. Results saved to final report

!!! note
    Test evaluation only occurs if you provide a `test.csv` file.

## Classification Metrics

| Metric | Code | Description |
|--------|------|-------------|
| Accuracy | `ACC` | Correct predictions / Total predictions |
| AUROC | `AUROC` | Area Under ROC Curve |
| AUPRC | `AUPRC` | Area Under Precision-Recall Curve |
| F1 Score | `F1` | Harmonic mean of precision and recall (macro) |
| Log Loss | `LOG_LOSS` | Negative log-likelihood |
| MCC | `MCC` | Matthews Correlation Coefficient |

### When to Use Each

| Metric | Best For |
|--------|----------|
| `ACC` | Balanced classes, general performance |
| `AUROC` | Comparing models, imbalanced data |
| `AUPRC` | Highly imbalanced data |
| `F1` | Balance of precision and recall |
| `LOG_LOSS` | Probability calibration |
| `MCC` | Imbalanced data, overall quality |

## Regression Metrics

| Metric | Code | Description |
|--------|------|-------------|
| Mean Squared Error | `MSE` | Average squared difference |
| Root MSE | `RMSE` | Square root of MSE |
| Mean Absolute Error | `MAE` | Average absolute difference |
| Mean Absolute Percentage Error | `MAPE` | Average percent error |
| R-squared | `R2` | Proportion of variance explained |
| Pearson Correlation | `PEARSON` | Linear correlation coefficient |
| Spearman Correlation | `SPEARMAN` | Rank correlation coefficient |

### When to Use Each

| Metric | Best For |
|--------|----------|
| `MSE`/`RMSE` | Penalizing large errors |
| `MAE` | Robust to outliers |
| `MAPE` | Relative error (positive targets) |
| `R2` | Understanding explained variance |
| `PEARSON` | Linear relationship strength |
| `SPEARMAN` | Monotonic relationship strength |

## Selecting Validation Metric

Choose with `--val-metric`:

```bash
./run.sh --val-metric AUROC
```

The agent optimizes for this metric when selecting the best iteration.

## Metric Calculation

Metrics are calculated using scikit-learn:

```python
from sklearn.metrics import accuracy_score, roc_auc_score, f1_score

# Classification
accuracy = accuracy_score(y_true, y_pred)
auroc = roc_auc_score(y_true, y_proba)
f1 = f1_score(y_true, y_pred, average='macro')

# Regression
mse = mean_squared_error(y_true, y_pred)
r2 = r2_score(y_true, y_pred)
```

## Multi-class Handling

For multi-class classification:

- **AUROC**: One-vs-rest
- **AUPRC**: Macro average
- **F1**: Macro average

## Overfitting Detection

The agent monitors for overfitting by comparing:

- Training metrics vs. validation metrics
- Large gaps indicate overfitting

Example iteration-plan guidance when overfitting is detected:

> Train ACC: 0.98, Validation ACC: 0.72
> Significant overfitting detected. Consider regularization.

## Best Iteration Selection

The best iteration is selected by:

1. Comparing validation metric across all iterations
2. Higher is better for most metrics (ACC, AUROC, F1, R2, PEARSON, SPEARMAN)
3. Lower is better for error metrics (MSE, RMSE, MAE, MAPE, LOG_LOSS)

## Viewing Results

### During Run

Progress shows current metrics:

```
Iteration 5: Validation ACC = 0.847 (Best: 0.852 at iter 3)
```

### In Reports

```
outputs/<agent_id>/reports/pdf/iteration_N.pdf
```

Contains:
- All iteration metrics
- Best iteration details
- Test set results
- Metric comparisons

### In W&B

If configured, metrics are logged to Weights & Biases:

- Metric plots over iterations
- Comparison tables
- Artifact tracking

## Custom Evaluation

For advanced use, modify metric definitions in:

```
src/utils/metrics.py
```

## Next Steps

- [Metrics Reference](../reference/metrics.md) - Complete metric list
- [Agent Architecture](architecture.md) - How iterations work
