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

**Multi-class:** Uses one-vs-rest.

---

### AUPRC

Area Under the Precision-Recall Curve.

| Property | Value |
|----------|-------|
| Range | 0 to 1 |
| Best | Higher |
| Use case | Highly imbalanced data |

**Multi-class:** Macro average.

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

**Multi-class:** Macro average.

---

### Log Loss (LOG_LOSS)

Negative log-likelihood of the true labels given predicted probabilities.

| Property | Value |
|----------|-------|
| Range | 0 to ∞ |
| Best | Lower |
| Use case | Probability calibration |

---

### Matthews Correlation Coefficient (MCC)

Correlation between predicted and actual classifications.

| Property | Value |
|----------|-------|
| Range | -1 to 1 |
| Best | Higher (1 is perfect) |
| Use case | Imbalanced data, overall quality |

---

## Regression Metrics

### Mean Squared Error (MSE)

Average of squared prediction errors.

```
MSE = (1/n) * Σ(y_true - y_pred)^2
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
RMSE = sqrt(MSE)
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

### Mean Absolute Percentage Error (MAPE)

Average absolute percent error.

| Property | Value |
|----------|-------|
| Range | 0 to ∞ |
| Best | Lower |
| Use case | Relative error on positive targets |

---

### R-squared (R2)

Proportion of variance explained by the model.

```
R2 = 1 - (SS_res / SS_tot)
```

| Property | Value |
|----------|-------|
| Range | -∞ to 1 |
| Best | Higher (1 is perfect) |
| Use case | Model explanatory power |

---

### Pearson Correlation (PEARSON)

Linear correlation between predictions and true values.

| Property | Value |
|----------|-------|
| Range | -1 to 1 |
| Best | Higher |
| Use case | Linear relationship strength |

---

### Spearman Correlation (SPEARMAN)

Rank correlation between predictions and true values.

| Property | Value |
|----------|-------|
| Range | -1 to 1 |
| Best | Higher |
| Use case | Monotonic relationship strength |

---

## Choosing a Metric

### Classification

| Scenario | Recommended Metric |
|----------|-------------------|
| Balanced classes | ACC, F1 |
| Imbalanced classes | AUROC, AUPRC, MCC |
| Probability calibration | LOG_LOSS |
| Overall quality | MCC |

### Regression

| Scenario | Recommended Metric |
|----------|-------------------|
| General performance | RMSE, MAE |
| Relative performance | R2, PEARSON, SPEARMAN |
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
| LOG_LOSS | Log Loss |
| MCC | Matthews Correlation Coefficient |
| MSE | Mean Squared Error |
| RMSE | Root Mean Squared Error |
| MAE | Mean Absolute Error |
| MAPE | Mean Absolute Percentage Error |
| R2 | R-squared |
| PEARSON | Pearson Correlation |
| SPEARMAN | Spearman Correlation |
