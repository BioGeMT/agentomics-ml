from sklearn.metrics import average_precision_score, roc_auc_score, accuracy_score, mean_squared_error, mean_absolute_error, r2_score, f1_score, log_loss, matthews_corrcoef, mean_absolute_percentage_error
from scipy.stats import pearsonr, spearmanr
import numpy as np
from typing import Optional

from agentomics.utils.cell_eval2_scoring import score_vcc2026
from agentomics.utils.task_types import TaskTypes


class Metric:
    """
    A metric class that encapsulates:
    - The computation function
    - Whether it needs probabilities (True) or class predictions (False)  
    - Whether higher values are better (True) or worse (False)
    - An optional description, shown to the agents next to the metric name
    """
    
    def __init__(self, function, needs_probabilities: bool, higher_is_better: bool, description: Optional[str] = None):
        self.function = function
        self.needs_probabilities = needs_probabilities
        self.higher_is_better = higher_is_better
        self.description = description
    
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

# name -> (cell-eval2 member, description)
CONDITIONAL_GENERATION_METRICS = {
    "AVG_SCORE": (
        "avg_score",
        "Virtual Cell Challenge 2026 score: the average of PDS, EXPR_MSE, DIRECTION_FIDELITY, DIRECTION_REACH, "
        "SIG_JACCARD and LFC_NMAE, computed by cell-eval2 (--preset vcc2026) separately for each context and averaged "
        "over contexts. Every metric is rescaled so that 0 = predicting the context-mean perturbation effect "
        "(baseline) and 1 = an experimental replicate (split halves of the real cells); higher is better and values "
        "can be negative or above 1. To compute it, use the installed cell-eval2 CLI per context: "
        "baseline --save-pred, prep-real-bundle, run, then score --real-bundle (scaled values in the from_replicate "
        "column), with --pert-col target_gene --control non-targeting; the predicted h5ad passed to cell-eval2 must "
        "also contain the context's real control cells. Each context is scored only on the genes observed (not NaN) "
        "in all its real cells, and without perturbations that have fewer than 2 real cells (on both sides). A context "
        "whose prep-real-bundle fails with 'no usable replicate scale' has no valid scale for any prediction and is "
        "left out of the average.",
    ),
    "PDS": (
        "pds_cosine",
        "Perturbation discrimination: how close (cosine) each predicted pseudobulk effect is to its own real effect, "
        "ranked against the real effects of all other perturbations.",
    ),
    "EXPR_MSE": (
        "expr_mse_unbiased_capped_norm",
        "Error between predicted and real pseudobulk expression, corrected for sampling noise. Unlike the other "
        "metrics it is clamped to [0, 1]: 0 means no better than the context-mean baseline.",
    ),
    "DIRECTION_FIDELITY": (
        "de_wilcoxon_direction_fidelity_yield_raw",
        "Genes differentially expressed (Wilcoxon) with the correct up/down direction, as min(precision, recall) "
        "against the real differentially expressed genes.",
    ),
    "DIRECTION_REACH": (
        "de_wilcoxon_direction_reach_raw",
        "How far down the predicted ranking of differentially expressed genes the predicted directions stay correct, "
        "relative to the number of confident real differentially expressed genes.",
    ),
    "SIG_JACCARD": (
        "de_wilcoxon_sig_jaccard",
        "Jaccard overlap between the predicted and real sets of significantly differentially expressed genes.",
    ),
    "LFC_NMAE": (
        "de_wilcoxon_lfc_nmae",
        "Normalised mean absolute error of the log2 fold changes on the real significantly differentially "
        "expressed genes.",
    ),
}

def get_conditional_generation_metrics_functions():
    return {
        name: Metric(
            function=lambda real, pred, member=member: score_vcc2026(real, pred)[member],
            needs_probabilities=False,
            higher_is_better=True,
            description=description,
        )
        for name, (member, description) in CONDITIONAL_GENERATION_METRICS.items()
    }

# name -> (ve_t1_scoring member, higher is better, description)
TEMPORAL_GENERATION_METRICS = {
    "VE_SCORE": (
        "SCORE",
        True,
        "Virtual Embryo Challenge Task 1 score, reconstructed locally from the published definitions (the official "
        "scorer is not public; the reconstruction reproduces the published E10.5 floor and ceiling anchors closely, "
        "but values only approximate the leaderboard): 100 * (0.25 DE_SCORE + 0.25 DE_DIRECTION + "
        "0.30 MMD + 0.20 VARIOGRAM skills), averaged over target stages. Each metric becomes a skill in (0, 1]: 0.5 for "
        "copy_last (predicting the latest observed stage unchanged), 1 for an experimental replicate (half of the real "
        "target cells scored against the other half), decaying towards 0 below copy_last. So 50 = copy_last, "
        "100 = replicate; higher is better. A population collapsed to an average cell scores below copy_last.",
    ),
    "VE_DE_SCORE": (
        "DE_SCORE",
        True,
        "DE gene recovery: real DE genes of the target against the latest observed stage (two-sided Mann-Whitney U, "
        "BH FDR 0.05, |mean log1p difference| >= 0.25; all genes; target cells subsampled to 10%). The prediction proposes the "
        "same numbers of up and down genes, ranked by its own mean change against the latest observed stage; overlap "
        "rescaled so 0 = ranking genes by expression level alone and 1 = exact.",
    ),
    "VE_DE_DIRECTION": (
        "DE_DIRECTION",
        True,
        "Change direction: rank partial correlation between the predicted and the real mean change per gene (against "
        "the latest observed stage), controlling for the latest observed stage's expression rank.",
    ),
    "VE_MMD": (
        "MMD",
        False,
        "Cell-state distribution: unbiased MMD^2 between up to 2000 predicted and real cells in a 30-component PCA "
        "fitted on the real target cells, averaged over RBF kernels exp(-s * d^2 / median d^2), s in {0.25, 0.5, 1, 2, 4}. "
        "Lower is better.",
    ),
    "VE_VARIOGRAM": (
        "VARIOGRAM",
        False,
        "Gene-gene co-variation: mean squared difference between predicted and real variograms (order 0.5, mean of "
        "|x_i - x_j|^0.5 over cells) on 20,000 random gene pairs, up to 2000 cells each. Lower is better.",
    ),
}

def get_temporal_generation_metrics_functions():
    from agentomics.utils.ve_t1_scoring import score_t1_split

    return {
        name: Metric(
            function=lambda split_path, pred, member=member: score_t1_split(split_path, pred)[member],
            needs_probabilities=False,
            higher_is_better=higher_is_better,
            description=description,
        )
        for name, (member, higher_is_better, description) in TEMPORAL_GENERATION_METRICS.items()
    }

def get_classification_metrics_names():
    return list(get_classification_metrics_functions().keys())

def get_regression_metrics_names():
    return list(get_regression_metrics_functions().keys())

def get_conditional_generation_metrics_names():
    return list(get_conditional_generation_metrics_functions().keys())

def get_temporal_generation_metrics_names():
    return list(TEMPORAL_GENERATION_METRICS.keys())

def get_task_to_metrics_names():
    return {
        TaskTypes.CLASSIFICATION: get_classification_metrics_names(),
        TaskTypes.REGRESSION: get_regression_metrics_names(),
        TaskTypes.CONDITIONAL_GENERATION: get_conditional_generation_metrics_names(),
        TaskTypes.TEMPORAL_GENERATION: get_temporal_generation_metrics_names(),
    }

def get_higher_is_better_map():
    """Return a dictionary mapping metric names to whether higher values are better."""
    all_metrics = {
        **get_classification_metrics_functions(),
        **get_regression_metrics_functions(),
        **get_conditional_generation_metrics_functions(),
        **get_temporal_generation_metrics_functions(),
    }
    return {name: metric.higher_is_better for name, metric in all_metrics.items()}

def describe_metric(name: str) -> str:
    """Return the metric name, followed by its description when it has one."""
    all_metrics = {
        **get_classification_metrics_functions(),
        **get_regression_metrics_functions(),
        **get_conditional_generation_metrics_functions(),
        **get_temporal_generation_metrics_functions(),
    }
    description = all_metrics[name].description
    return f"{name} ({description})" if description else name

def get_default_val_metric(task_type: str) -> str:
    if task_type == TaskTypes.CLASSIFICATION:
        return "AUROC"
    if task_type == TaskTypes.REGRESSION:
        return "MAE"
    if task_type == TaskTypes.CONDITIONAL_GENERATION:
        return "AVG_SCORE"
    if task_type == TaskTypes.TEMPORAL_GENERATION:
        return "VE_SCORE"
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
