"""Local reconstruction of the Virtual Embryo Challenge Task 1 score.

Follows the published definitions (virtualembryo.ai/challenge/evaluation, section scoring); the official
scorer (starter kit score_h5ad.py) is not public, so absolute values can differ from the leaderboard.

For one target stage, with reference = the latest observed stage (what copy_last predicts):
Like the official scorer, the target stage is first subsampled to 10% of its cells. Changes are measured
against the mean of the full reference stage: subsampling it would put the same sampling noise into the
predicted and the real change, rewarding any prediction made of other reference cells. All genes are used.
- DE_SCORE (25%): truth DE genes (target vs reference) by two-sided Mann-Whitney U, BH FDR 0.05, and
  |mean log1p difference| >= 0.25. The prediction proposes the same numbers of up/down genes, ranked by
  its own log fold change against the reference. Overlap, rescaled so 0 = ranking by reference expression alone, 1 = exact.
- DE_DIRECTION (25%): rank partial correlation between predicted and observed change, controlling for
  the reference expression rank.
- MMD (30%, lower is better): up to 2000 cells each, 30-component PCA fitted on the target, unbiased
  MMD^2 averaged over RBF kernels exp(-s * gamma0 * d^2), s in {0.25, 0.5, 1, 2, 4}, gamma0 = 1 / median d^2.
- VARIOGRAM (20%, lower is better): 20,000 random gene pairs, mean squared difference between the
  predicted and observed variograms of order p = 0.5.

Each metric is turned into a skill in (0, 1]: 0.5 at the floor (copy_last), 1 at the ceiling (split-half
of the truth), linear in between, decaying towards 0 below the floor. SCORE = 100 * weighted skills.

Checked on the E9.5 -> E10.5 board against the published anchors (floor / ceiling, official in brackets):
DE_SCORE 0 / 0.87 (0 / 0.85), DE_DIRECTION 0 / 0.79 (0 / 0.79), MMD 0.084 / 0.0043 (0.084 / 0.0041),
VARIOGRAM 0.0052 / 0.00017 (0.0052 / 0.00016).
"""
from __future__ import annotations

import hashlib
import json
import tempfile
import os
from concurrent.futures import ThreadPoolExecutor
from dataclasses import dataclass, field
from pathlib import Path

import numpy as np

WEIGHTS = {"DE_SCORE": 0.25, "DE_DIRECTION": 0.25, "MMD": 0.30, "VARIOGRAM": 0.20}
HIGHER_IS_BETTER = {"DE_SCORE": True, "DE_DIRECTION": True, "MMD": False, "VARIOGRAM": False}

DE_FDR = 0.05
DE_EFFECT_FLOOR = 0.25
DE_MAX_CELLS = 3000
STAGE_FRACTION = 0.1
# Small local splits (e.g. mini_train) keep at least this many cells per stage.
STAGE_MIN_CELLS = 500
POPULATION_MAX_CELLS = 2000
PCA_COMPONENTS = 30
MMD_SCALES = (0.25, 0.5, 1.0, 2.0, 4.0)
VARIOGRAM_PAIRS = 20_000
VARIOGRAM_P = 0.5
SEED = 0

ANCHOR_CACHE_DIR = Path(tempfile.gettempdir()) / "agentomics_ve_t1_anchors"


@dataclass
class StageData:
    """Dense float32 cell x gene matrices for one target stage, on the same genes."""
    reference: np.ndarray
    truth: np.ndarray
    # What scoring needs from reference and truth alone, computed once and shared by every prediction.
    truth_side: "TruthSide | None" = field(default=None, init=False, repr=False, compare=False)


@dataclass
class TruthSide:
    reference_mean: np.ndarray
    true_change: np.ndarray
    up: np.ndarray
    down: np.ndarray
    population_truth: np.ndarray
    pca: object
    truth_embedding: np.ndarray
    gamma0: float
    truth_kernel_terms: list[float]
    gene_pairs: tuple[np.ndarray, np.ndarray]
    observed_variogram: np.ndarray
    rng_state: dict


def make_stage(reference: np.ndarray, truth: np.ndarray, seed: int = SEED) -> StageData:
    """The scored stage: the full reference and the truth subsampled to STAGE_FRACTION of its cells."""
    rng = np.random.default_rng(seed)

    def keep(matrix: np.ndarray) -> np.ndarray:
        n_keep = max(int(np.ceil(STAGE_FRACTION * len(matrix))), min(len(matrix), STAGE_MIN_CELLS))
        return _subsample(matrix, n_keep, rng)

    return StageData(reference=reference, truth=keep(truth))


def raw_metrics(reference: np.ndarray, truth: np.ndarray, prediction: np.ndarray, seed: int = SEED) -> dict[str, float]:
    """The four unscaled T1 metrics of a prediction."""
    return _prediction_metrics(truth_side(reference, truth, seed), prediction)


def truth_side(reference: np.ndarray, truth: np.ndarray, seed: int = SEED) -> TruthSide:
    """Everything the metrics need that does not depend on the prediction."""
    from scipy.spatial.distance import pdist
    from sklearn.decomposition import PCA

    rng = np.random.default_rng(seed)
    de_reference = _subsample(reference, DE_MAX_CELLS, rng)
    de_truth = _subsample(truth, DE_MAX_CELLS, rng)
    reference_mean = reference.mean(axis=0)
    true_change = de_truth.mean(axis=0) - reference_mean
    up, down = _truth_de_genes(de_reference, de_truth, true_change)

    population_truth = _subsample(truth, POPULATION_MAX_CELLS, rng)
    components = min(PCA_COMPONENTS, len(population_truth) - 1, population_truth.shape[1])
    pca = PCA(n_components=components, svd_solver="randomized", random_state=SEED).fit(population_truth)
    x = pca.transform(population_truth)
    median_squared_distance = np.median(pdist(x, "sqeuclidean"))
    gamma0 = 1.0 / median_squared_distance if median_squared_distance > 0 else 1.0
    d_xx = _squared_distances(x, x)
    n = len(x)
    truth_kernel_terms = [
        float((np.exp(-scale * gamma0 * d_xx).sum() - n) / (n * (n - 1))) for scale in MMD_SCALES
    ]

    pair_rng = np.random.default_rng([seed, 1])
    n_genes = truth.shape[1]
    first = pair_rng.integers(0, n_genes, VARIOGRAM_PAIRS)
    second = (first + pair_rng.integers(1, n_genes, VARIOGRAM_PAIRS)) % n_genes  # never the same gene
    return TruthSide(
        reference_mean=reference_mean,
        true_change=true_change,
        up=up,
        down=down,
        population_truth=population_truth,
        pca=pca,
        truth_embedding=x,
        gamma0=gamma0,
        truth_kernel_terms=truth_kernel_terms,
        gene_pairs=(first, second),
        observed_variogram=_variogram(population_truth, first, second),
        rng_state=rng.bit_generator.state,
    )


def _prediction_metrics(side: TruthSide, prediction: np.ndarray) -> dict[str, float]:
    rng = np.random.default_rng()
    rng.bit_generator.state = side.rng_state
    predicted_change = prediction.mean(axis=0) - side.reference_mean
    population_prediction = _subsample(prediction, POPULATION_MAX_CELLS, rng)
    predicted_variogram = _variogram(population_prediction, *side.gene_pairs)
    return {
        "DE_SCORE": _de_score(predicted_change, side.up, side.down, side.reference_mean),
        "DE_DIRECTION": _partial_rank_correlation(predicted_change, side.true_change, side.reference_mean),
        "MMD": _mmd(side, population_prediction),
        "VARIOGRAM": float(np.mean((predicted_variogram - side.observed_variogram) ** 2)),
    }


def anchors(stage: StageData) -> dict[str, dict[str, float]]:
    """Floor (copy_last: the reference as prediction) and ceiling (half of the truth against the other half)."""
    floor = raw_metrics(stage.reference, stage.truth, stage.reference)
    order = np.random.default_rng(SEED).permutation(len(stage.truth))
    half = len(order) // 2
    ceiling = raw_metrics(stage.reference, stage.truth[order[half:]], stage.truth[order[:half]])
    return {"floor": floor, "ceiling": ceiling}


def skill(name: str, value: float, floor: float, ceiling: float) -> float:
    """0.5 at the floor, 1 at (or beyond) the ceiling, in (0, 0.5) below the floor."""
    span = ceiling - floor
    if not np.isfinite(value) or span == 0 or not np.isfinite(span):
        return 0.5 if value == floor else 0.0
    progress = (value - floor) / span  # sign of span encodes the metric direction
    if progress >= 0:
        return float(0.5 + 0.5 * min(progress, 1.0))
    return float(0.5 * np.exp(progress))


def score_stage(stage: StageData, prediction: np.ndarray, stage_anchors: dict | None = None) -> dict[str, float]:
    stage_anchors = stage_anchors or anchors(stage)
    if stage.truth_side is None:
        stage.truth_side = truth_side(stage.reference, stage.truth)
    metrics = _prediction_metrics(stage.truth_side, prediction)
    skills = {
        name: skill(name, metrics[name], stage_anchors["floor"][name], stage_anchors["ceiling"][name])
        for name in WEIGHTS
    }
    return {
        "SCORE": 100.0 * sum(WEIGHTS[name] * skills[name] for name in WEIGHTS),
        **metrics,
        **{f"{name}_SKILL": value for name, value in skills.items()},
    }


def score_t1_split(split_path: Path, predictions_path: Path) -> dict[str, float]:
    """Scores predictions.h5ad against a temporal-generation split, averaging over its target stages."""
    import anndata as ad

    from agentomics.datasets.conditional_generation_contract import CONTROLS_FILE_NAME
    from agentomics.datasets.data_contract import GENERATION_LABELS_FILE_NAME, INPUT_DIR_NAME
    from agentomics.datasets.temporal_generation_contract import STAGE_COLUMN_NAME, stage_value

    split_path = Path(split_path)
    observed_path = split_path / INPUT_DIR_NAME / CONTROLS_FILE_NAME
    labels_path = split_path / GENERATION_LABELS_FILE_NAME
    observed = ad.read_h5ad(observed_path)
    labels = ad.read_h5ad(labels_path)
    predicted = ad.read_h5ad(predictions_path)
    if list(predicted.var_names) != list(labels.var_names):
        predicted = predicted[:, labels.var_names]

    observed_stages = observed.obs[STAGE_COLUMN_NAME].astype(str)
    latest = max(observed_stages.unique(), key=stage_value)
    reference = _dense(observed[(observed_stages == latest).to_numpy()].X)

    per_stage = []
    label_stages = labels.obs[STAGE_COLUMN_NAME].astype(str)
    predicted_stages = predicted.obs[STAGE_COLUMN_NAME].astype(str)
    for target in sorted(label_stages.unique(), key=stage_value):
        stage = make_stage(reference, _dense(labels[(label_stages == target).to_numpy()].X))
        prediction = _dense(predicted[(predicted_stages == target).to_numpy()].X)
        if len(prediction) == 0:
            raise ValueError(f"predictions contain no cells for target stage {target}")
        stage_anchors = _cached_anchors(stage, (observed_path, labels_path, latest, target))
        per_stage.append(score_stage(stage, prediction, stage_anchors))
    return {name: float(np.mean([scores[name] for scores in per_stage])) for name in per_stage[0]}


def _cached_anchors(stage: StageData, key_parts) -> dict:
    digest = hashlib.sha256()
    for part in key_parts:
        if isinstance(part, Path):
            resolved = part.resolve()
            status = resolved.stat()
            digest.update(f"{resolved}:{status.st_size}:{status.st_mtime_ns}".encode())
        else:
            digest.update(str(part).encode())
    digest.update(json.dumps(
        [WEIGHTS, DE_MAX_CELLS, POPULATION_MAX_CELLS, SEED, VARIOGRAM_PAIRS, STAGE_FRACTION, STAGE_MIN_CELLS, "v5"]
    ).encode())
    cache_path = ANCHOR_CACHE_DIR / f"{digest.hexdigest()}.json"
    if cache_path.is_file():
        return json.loads(cache_path.read_text())
    stage_anchors = anchors(stage)
    ANCHOR_CACHE_DIR.mkdir(parents=True, exist_ok=True)
    temporary_path = cache_path.with_suffix(".tmp")
    temporary_path.write_text(json.dumps(stage_anchors))
    temporary_path.replace(cache_path)
    return stage_anchors


def _dense(matrix) -> np.ndarray:
    from scipy import sparse

    if sparse.issparse(matrix):
        matrix = matrix.toarray()
    return np.asarray(matrix, dtype=np.float32)


def _subsample(matrix: np.ndarray, max_cells: int, rng: np.random.Generator) -> np.ndarray:
    if len(matrix) <= max_cells:
        return matrix
    return matrix[np.sort(rng.choice(len(matrix), max_cells, replace=False))]


def _truth_de_genes(reference: np.ndarray, truth: np.ndarray, true_change: np.ndarray, chunk: int = 2048):
    from scipy.stats import false_discovery_control, mannwhitneyu

    p_values = np.ones(reference.shape[1])
    # Genes zero in every cell of both groups have no test; they keep p = 1.
    detected = np.flatnonzero((reference > 0).any(axis=0) | (truth > 0).any(axis=0))
    chunks = [detected[start:start + chunk] for start in range(0, len(detected), chunk)]

    def test(genes):
        return genes, mannwhitneyu(truth[:, genes], reference[:, genes], axis=0).pvalue

    with ThreadPoolExecutor(max_workers=_workers()) as executor:
        for genes, chunk_p_values in executor.map(test, chunks):
            p_values[genes] = chunk_p_values
    p_values = np.nan_to_num(p_values, nan=1.0)
    significant = (false_discovery_control(p_values) < DE_FDR) & (np.abs(true_change) >= DE_EFFECT_FLOOR)
    return np.flatnonzero(significant & (true_change > 0)), np.flatnonzero(significant & (true_change < 0))


def _de_score(predicted_change: np.ndarray, up: np.ndarray, down: np.ndarray, reference_mean: np.ndarray) -> float:
    """Overlap with the real DE genes, above what ranking genes by reference expression level alone achieves."""
    if len(up) + len(down) == 0 or np.ptp(predicted_change) == 0:
        return 0.0
    overlap = _de_overlap(predicted_change, up, down)
    # Rescaling the reference (e.g. x 0.99 or x 2) ranks genes by expression level, in either direction.
    expression_level = max(_de_overlap(reference_mean, up, down), _de_overlap(-reference_mean, up, down))
    return float((overlap - expression_level) / (1 - expression_level)) if expression_level < 1 else 0.0


def _de_overlap(predicted_change: np.ndarray, up: np.ndarray, down: np.ndarray) -> float:
    """Fraction of the real up and down genes among the same numbers of top and bottom predicted changes."""
    n_genes, n_up, n_down = len(predicted_change), len(up), len(down)
    order = np.argsort(-predicted_change, kind="stable")
    predicted_up = order[:n_up]
    predicted_down = order[n_genes - n_down:] if n_down else order[:0]
    hits = len(np.intersect1d(predicted_up, up)) + len(np.intersect1d(predicted_down, down))
    return hits / (n_up + n_down)


def _partial_rank_correlation(x: np.ndarray, y: np.ndarray, control: np.ndarray) -> float:
    from scipy.stats import rankdata

    if np.ptp(x) == 0 or np.ptp(y) == 0:
        return 0.0
    rx, ry, rz = (rankdata(values) for values in (x, y, control))
    r_xy, r_xz, r_yz = (np.corrcoef(a, b)[0, 1] for a, b in ((rx, ry), (rx, rz), (ry, rz)))
    denominator = np.sqrt((1 - r_xz**2) * (1 - r_yz**2))
    return float((r_xy - r_xz * r_yz) / denominator) if denominator > 0 else 0.0


def _mmd(side: TruthSide, prediction: np.ndarray) -> float:
    """Unbiased MMD^2 in the truth's PCA space, averaged over the RBF kernels exp(-s * gamma0 * d^2)."""
    y = side.pca.transform(prediction)
    d_yy, d_xy = _squared_distances(y, y), _squared_distances(side.truth_embedding, y)
    m = len(y)
    values = []
    for scale, truth_term in zip(MMD_SCALES, side.truth_kernel_terms):
        gamma = scale * side.gamma0
        values.append(truth_term + (np.exp(-gamma * d_yy).sum() - m) / (m * (m - 1)) - 2 * np.exp(-gamma * d_xy).mean())
    return float(np.mean(values))


def _squared_distances(a: np.ndarray, b: np.ndarray) -> np.ndarray:
    from scipy.spatial.distance import cdist

    return cdist(a, b, "sqeuclidean")


def _variogram(cells: np.ndarray, first: np.ndarray, second: np.ndarray) -> np.ndarray:
    return (np.abs(cells[:, first] - cells[:, second]) ** VARIOGRAM_P).mean(axis=0)


def _workers() -> int:
    return max(1, min(16, len(os.sched_getaffinity(0)) if hasattr(os, "sched_getaffinity") else os.cpu_count() or 1))
