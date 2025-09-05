from sklearn.metrics import average_precision_score, roc_auc_score, accuracy_score, mean_squared_error, mean_absolute_error, r2_score
import numpy as np

def get_classification_metrics_functions(acc_threshold=0.5):
    metric_to_fn = {
        "ACC": lambda y_true, y_pred: accuracy_score(y_true, (y_pred >= acc_threshold).astype(int)),
        "AUPRC": lambda y_true, y_pred: average_precision_score(y_true, y_pred),
        "AUROC": lambda y_true, y_pred: roc_auc_score(y_true, y_pred),
    }
    return metric_to_fn

def get_regression_metrics_functions():
    metric_to_fn = {
        "MSE":  lambda y_true, y_pred: mean_squared_error(y_true, y_pred),
        "RMSE": lambda y_true, y_pred: np.sqrt(mean_squared_error(y_true, y_pred)),
        "MAE": lambda y_true, y_pred: mean_absolute_error(y_true, y_pred),
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