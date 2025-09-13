from sklearn.metrics import average_precision_score, roc_auc_score, accuracy_score, mean_squared_error, mean_absolute_error, r2_score, f1_score, log_loss, matthews_corrcoef, mean_absolute_percentage_error
from scipy.stats import pearsonr
import numpy as np

def pcc(y_true, y_pred):
    r = pearsonr(np.asarray(y_true, float).ravel(), np.asarray(y_pred, float).ravel())[0]
    return float(r) if np.isfinite(r) else 0.0

def get_classification_metrics_functions(acc_threshold=0.5):
    metric_to_fn = {
        "ACC": lambda y_true, y_pred: accuracy_score(y_true, (y_pred >= acc_threshold).astype(int)),
        "AUPRC": lambda y_true, y_pred: average_precision_score(y_true, y_pred),
        "AUROC": lambda y_true, y_pred: roc_auc_score(y_true, y_pred),
        "F1": lambda y_true, y_pred: f1_score(y_true, (y_pred >= acc_threshold).astype(int)),
        "LOG_LOSS": lambda y_true, y_pred: log_loss(y_true, np.clip(y_pred, 1e-15, 1-1e-15)),
        "MCC": lambda y_true, y_pred: matthews_corrcoef(y_true, (y_pred >= acc_threshold).astype(int)),
    }
    return metric_to_fn

def get_regression_metrics_functions():
    metric_to_fn = {
        "MSE":  lambda y_true, y_pred: mean_squared_error(y_true, y_pred),
        "RMSE": lambda y_true, y_pred: np.sqrt(mean_squared_error(y_true, y_pred)),
        "MAE": lambda y_true, y_pred: mean_absolute_error(y_true, y_pred),
        "MAPE": lambda y_true, y_pred: mean_absolute_percentage_error(y_true, y_pred),
        "POS_PCC": lambda y_true, y_pred: max(pcc(y_true, y_pred), 0.0),
        "NEG_PCC": lambda y_true, y_pred: min(pcc(y_true, y_pred), 0.0),
        "R2": lambda y_true, y_pred: r2_score(y_true, y_pred),
    }
    return metric_to_fn

def get_classification_metrics_names():
    return list(get_classification_metrics_functions().keys())

def get_regression_metrics_names():
    return list(get_regression_metrics_functions().keys())

def get_task_to_metrics_names():
    return {
        "classification": get_classification_metrics_names(),
        "regression": get_regression_metrics_names(),
    }
_HIGHER_IS_BETTER = {"ACC": True, "AUPRC": True, "AUROC": True, "F1": True, "LOG_LOSS": False, "MCC": True, "MSE": False, "RMSE": False, "MAE": False, "MAPE": False, "POS_PCC": True, "NEG_PCC": False, "R2": True}

def get_higher_is_better_map():
    return _HIGHER_IS_BETTER.copy()
