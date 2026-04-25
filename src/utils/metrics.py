from sklearn.metrics import average_precision_score, roc_auc_score, accuracy_score, mean_squared_error, mean_absolute_error, r2_score, f1_score, log_loss, matthews_corrcoef, mean_absolute_percentage_error
from scipy.stats import pearsonr, spearmanr
import numpy as np
from typing import Optional

from utils.task_types import TaskTypes


class Metric:
    """
    A metric class that encapsulates:
    - The computation function
    - Whether it needs probabilities (True) or class predictions (False)  
    - Whether higher values are better (True) or worse (False)
    """
    
    def __init__(self, function, needs_probabilities: bool, higher_is_better: bool):
        self.function = function
        self.needs_probabilities = needs_probabilities
        self.higher_is_better = higher_is_better
    
    def __call__(self, y_true, y_pred_or_prob):
        return self.function(y_true, y_pred_or_prob)

def _pcc(y_true, y_pred):
    r = pearsonr(np.asarray(y_true, float).ravel(), np.asarray(y_pred, float).ravel())[0]
    return float(r) if np.isfinite(r) else 0.0

def _scc(y_true, y_pred):
    """Spearman correlation coefficient."""
    r = spearmanr(np.asarray(y_true, float).ravel(), np.asarray(y_pred, float).ravel())[0]
    return float(r) if np.isfinite(r) else 0.0

def _auroc_metric(y_true, y_prob):
    """Handle AUROC for both binary and multiclass cases."""
    if y_prob.shape[1] == 2:
        # Binary classification
        return roc_auc_score(y_true, y_prob[:, 1])
    else:
        # Multiclass classification - use 'ovr' (one-vs-rest) strategy
        return roc_auc_score(y_true, y_prob, multi_class='ovr')

def _auprc_metric(y_true, y_prob):
    """Handle AUPRC for both binary and multiclass cases."""
    if y_prob.shape[1] == 2:
        # Binary classification
        return average_precision_score(y_true, y_prob[:, 1])
    else:
        # Multiclass classification - use macro average
        return average_precision_score(y_true, y_prob, average='macro')

def get_classification_metrics_functions():
    """Returns a dictionary mapping metric names to Metric objects."""
    return {
        "ACC": Metric(
            function=lambda y_true, y_pred: accuracy_score(y_true, y_pred),
            needs_probabilities=False,
            higher_is_better=True
        ),
        "AUPRC": Metric(
            function=_auprc_metric,
            needs_probabilities=True,
            higher_is_better=True
        ),
        "AUROC": Metric(
            function=_auroc_metric,
            needs_probabilities=True,
            higher_is_better=True
        ),
        "F1": Metric(
            function=lambda y_true, y_pred: f1_score(y_true, y_pred, average='macro'),
            needs_probabilities=False,
            higher_is_better=True
        ),
        "LOG_LOSS": Metric(
            function=lambda y_true, y_prob: log_loss(y_true, np.clip(y_prob, 1e-15, 1-1e-15)),
            needs_probabilities=True,
            higher_is_better=False
        ),
        "MCC": Metric(
            function=lambda y_true, y_pred: matthews_corrcoef(y_true, y_pred),
            needs_probabilities=False,
            higher_is_better=True
        ),
    }

def get_regression_metrics_functions():
    """Returns a dictionary mapping metric names to Metric objects."""
    return {
        "MSE": Metric(
            function=lambda y_true, y_pred: mean_squared_error(y_true, y_pred),
            needs_probabilities=False,  # Regression always uses predictions
            higher_is_better=False
        ),
        "RMSE": Metric(
            function=lambda y_true, y_pred: np.sqrt(mean_squared_error(y_true, y_pred)),
            needs_probabilities=False,
            higher_is_better=False
        ),
        "MAE": Metric(
            function=lambda y_true, y_pred: mean_absolute_error(y_true, y_pred),
            needs_probabilities=False,
            higher_is_better=False
        ),
        "MAPE": Metric(
            function=lambda y_true, y_pred: mean_absolute_percentage_error(y_true, y_pred),
            needs_probabilities=False,
            higher_is_better=False
        ),
        "PEARSON": Metric(
            function=lambda y_true, y_pred: _pcc(y_true, y_pred),
            needs_probabilities=False,
            higher_is_better=True
        ),
        "SPEARMAN": Metric(
            function=lambda y_true, y_pred: _scc(y_true, y_pred),
            needs_probabilities=False,
            higher_is_better=True
        ),
        "R2": Metric(
            function=lambda y_true, y_pred: r2_score(y_true, y_pred),
            needs_probabilities=False,
            higher_is_better=True
        ),
    }

def get_classification_metrics_names():
    return list(get_classification_metrics_functions().keys())

def get_regression_metrics_names():
    return list(get_regression_metrics_functions().keys())

def get_task_to_metrics_names():
    return {
        TaskTypes.CLASSIFICATION: get_classification_metrics_names(),
        TaskTypes.REGRESSION: get_regression_metrics_names(),
    }

def get_higher_is_better_map():
    """Return a dictionary mapping metric names to whether higher values are better."""
    all_metrics = {
        **get_classification_metrics_functions(),
        **get_regression_metrics_functions(),
    }
    return {name: metric.higher_is_better for name, metric in all_metrics.items()}

def get_default_val_metric(task_type: str) -> str:
    if task_type == TaskTypes.CLASSIFICATION:
        return "AUROC"
    if task_type == TaskTypes.REGRESSION:
        return "MAE"
    raise ValueError(f"Unknown task_type: {task_type}. Expected one of {TaskTypes}.")

def resolve_val_metric(task_type: str, val_metric: Optional[str] = None) -> str:
    """Resolve a validation metric for a task, applying defaults when omitted."""
    if not val_metric:
        return get_default_val_metric(task_type)

    allowed = get_task_to_metrics_names().get(task_type)
    if not allowed:
        raise ValueError(f"Unknown task_type: {task_type}. Expected one of {TaskTypes}.")
    if val_metric not in allowed:
        raise ValueError(
            f"Validation metric '{val_metric}' is invalid for task type '{task_type}'. "
            f"Allowed metrics: {', '.join(allowed)}"
        )
    return val_metric
