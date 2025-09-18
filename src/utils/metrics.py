from sklearn.metrics import average_precision_score, roc_auc_score, accuracy_score, mean_squared_error, mean_absolute_error, r2_score, f1_score, log_loss, matthews_corrcoef, mean_absolute_percentage_error
from scipy.stats import pearsonr
import numpy as np

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
        "POS_PCC": Metric(
            function=lambda y_true, y_pred: _pcc(y_true, y_pred),
            needs_probabilities=False,
            higher_is_better=True
        ),
        "NEG_PCC": Metric(
            function=lambda y_true, y_pred: _pcc(y_true, y_pred),
            needs_probabilities=False,
            higher_is_better=False  # Negative correlation
        ),
        "R2": Metric(
            function=lambda y_true, y_pred: r2_score(y_true, y_pred),
            needs_probabilities=False,
            higher_is_better=True
        ),
    }

def get_classification_metrics_requiring_probabilities():
    """Return a set of metric names that require probability scores instead of hard predictions."""
    metrics = get_classification_metrics_functions()
    return {name for name, metric in metrics.items() if metric.needs_probabilities}

def get_classification_metrics_requiring_predictions():
    """Return a set of metric names that require hard predictions instead of probabilities."""
    metrics = get_classification_metrics_functions()
    return {name for name, metric in metrics.items() if not metric.needs_probabilities}

def get_classification_metrics_names():
    return list(get_classification_metrics_functions().keys())

def get_regression_metrics_names():
    return list(get_regression_metrics_functions().keys())

def get_task_to_metrics_names():
    return {
        "classification": get_classification_metrics_names(),
        "regression": get_regression_metrics_names(),
    }

def get_higher_is_better_map():
    """Return a dictionary mapping metric names to whether higher values are better."""
    all_metrics = {
        **get_classification_metrics_functions(),
        **get_regression_metrics_functions(),
    }
    return {name: metric.higher_is_better for name, metric in all_metrics.items()}
