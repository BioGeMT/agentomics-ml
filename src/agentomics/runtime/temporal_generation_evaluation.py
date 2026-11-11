"""Scores temporal generation predictions against the measured cells of a split."""
from __future__ import annotations

from pathlib import Path

from agentomics.utils.ve_t1_scoring import score_t1_split


def get_temporal_generation_metrics(predictions_path: Path, labels_path: Path) -> dict[str, float]:
    """Scores predictions.h5ad against labels_path (labels.h5ad) and the observed cells of its split."""
    from agentomics.utils.metrics import TEMPORAL_GENERATION_METRICS

    scores = score_t1_split(Path(labels_path).parent, Path(predictions_path))
    return {name: scores[member] for name, (member, _, _) in TEMPORAL_GENERATION_METRICS.items()}
