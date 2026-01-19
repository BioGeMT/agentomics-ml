# Metrics Reference

Complete list of available metrics for model evaluation.

## Classification Metrics

### Accuracy (ACC)

The proportion of correct predictions.

```
ACC = (TP + TN) / (TP + TN + FP + FN)
```

| Property | Value |
|----------|-------|
| Range | 0 to 1 |
| Best | Higher |
| Use case | Balanced datasets |

---

### AUROC

Area Under the Receiver Operating Characteristic Curve.

| Property | Value |
|----------|-------|
| Range | 0 to 1 |
| Best | Higher |
| Use case | Ranking quality, imbalanced data |

**Multi-class:** Uses one-vs-rest with macro averaging.

---

### AUPRC

Area Under the Precision-Recall Curve.

| Property | Value |
|----------|-------|
| Range | 0 to 1 |
| Best | Higher |
| Use case | Highly imbalanced data |

Better than AUROC when positive class is rare.

---

### F1 Score (F1)

Harmonic mean of precision and recall.

```
F1 = 2 * (Precision * Recall) / (Precision + Recall)
```

| Property | Value |
|----------|-------|
| Range | 0 to 1 |
| Best | Higher |
| Use case | Balance precision and recall |

**Multi-class:** Uses weighted averaging by default.

---

### Precision (PRECISION)

Proportion of positive predictions that are correct.

```
Precision = TP / (TP + FP)
```

| Property | Value |
|----------|-------|
| Range | 0 to 1 |
| Best | Higher |
| Use case | Minimize false positives |

---

### Recall (RECALL)

Proportion of actual positives that are correctly identified.

```
Recall = TP / (TP + FN)
```

| Property | Value |
|----------|-------|
| Range | 0 to 1 |
| Best | Higher |
| Use case | Minimize false negatives |

---

### Matthews Correlation Coefficient (MCC)

Correlation between predicted and actual classifications.

| Property | Value |
|----------|-------|
| Range | -1 to 1 |
| Best | Higher (1 is perfect) |
| Use case | Imbalanced data, overall quality |

Considered one of the best single metrics for binary classification.

---

### Balanced Accuracy (BALANCED_ACC)

Average recall across all classes.

| Property | Value |
|----------|-------|
| Range | 0 to 1 |
| Best | Higher |
| Use case | Multi-class with imbalance |

---

## Regression Metrics

### Mean Squared Error (MSE)

Average of squared prediction errors.

```
MSE = (1/n) * Σ(y_true - y_pred)²
```

| Property | Value |
|----------|-------|
| Range | 0 to ∞ |
| Best | Lower |
| Use case | Penalizing large errors |

---

### Root Mean Squared Error (RMSE)

Square root of MSE.

```
RMSE = √MSE
```

| Property | Value |
|----------|-------|
| Range | 0 to ∞ |
| Best | Lower |
| Use case | Same units as target |

---

### Mean Absolute Error (MAE)

Average of absolute prediction errors.

```
MAE = (1/n) * Σ|y_true - y_pred|
```

| Property | Value |
|----------|-------|
| Range | 0 to ∞ |
| Best | Lower |
| Use case | Robust to outliers |

---

### R-squared (R2)

Proportion of variance explained by the model.

```
R² = 1 - (SS_res / SS_tot)
```

| Property | Value |
|----------|-------|
| Range | -∞ to 1 |
| Best | Higher (1 is perfect) |
| Use case | Model explanatory power |

Can be negative if model is worse than predicting the mean.

---

### Pearson Correlation (PEARSON)

Linear correlation between predictions and true values.

| Property | Value |
|----------|-------|
| Range | -1 to 1 |
| Best | Higher absolute value |
| Use case | Linear relationship strength |

---

## Choosing a Metric

### Classification

| Scenario | Recommended Metric |
|----------|-------------------|
| Balanced classes | ACC, F1 |
| Imbalanced classes | AUROC, AUPRC, MCC |
| Minimize false positives | PRECISION |
| Minimize false negatives | RECALL |
| Overall quality | MCC |
| Model ranking | AUROC |

### Regression

| Scenario | Recommended Metric |
|----------|-------------------|
| General performance | RMSE, R2 |
| Outlier-robust | MAE |
| Relative performance | R2, PEARSON |
| Same units as target | RMSE, MAE |

---

## Using Metrics

### CLI

```bash
./run.sh --val-metric AUROC
```

### Listing Available Metrics

```bash
./run.sh --list-metrics
```

---

## Metric Abbreviations

| Abbreviation | Full Name |
|--------------|-----------|
| ACC | Accuracy |
| AUROC | Area Under ROC Curve |
| AUPRC | Area Under Precision-Recall Curve |
| F1 | F1 Score |
| MCC | Matthews Correlation Coefficient |
| MSE | Mean Squared Error |
| RMSE | Root Mean Squared Error |
| MAE | Mean Absolute Error |
| R2 | R-squared |
