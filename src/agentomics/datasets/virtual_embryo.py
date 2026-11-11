"""Reading and writing Virtual Embryo Challenge Task 1 cells (whole-transcriptome RNA, log1p-normalised)."""
from __future__ import annotations

from pathlib import Path

import anndata as ad
import numpy as np
import pandas as pd
from scipy import sparse

from agentomics.datasets.temporal_generation_contract import (
    MAX_LOG_EXPRESSION,
    STAGE_COLUMN_NAME,
    stage_value,
)

CELLTYPE_COLUMN_NAME = "celltype"
# Cells per stage the challenge accepts: E10.5 board 1,000-5,118, E12.5 board 638-5,118.
DEFAULT_N_CELLS = 5000
MAX_SUBMISSION_CELLS = 5118


def load_stage(path: Path, stage: str, genes: list[str] | None = None) -> ad.AnnData:
    """One stage's cells as CSR float32, with obs 'stage' (and 'celltype' when the source has it).

    genes, when given, is the gene axis to put the cells on; it must hold the same genes as the file.
    """
    stage_value(stage)
    source = ad.read_h5ad(path)
    if genes is not None and list(source.var_names) != genes:
        if set(source.var_names) != set(genes):
            raise ValueError(f"{path} does not hold the released genes")
        source = source[:, genes]
    matrix = sparse.csr_matrix(source.X, dtype=np.float32)
    matrix.sum_duplicates()
    matrix.eliminate_zeros()
    values = matrix.data
    if not np.isfinite(values).all() or (values < 0).any():
        raise ValueError(f"{path} holds negative or non-finite values")
    if values.size and values.max() > MAX_LOG_EXPRESSION:
        raise ValueError(f"{path} does not look log1p-normalised (max {values.max():.1f})")
    obs = pd.DataFrame({STAGE_COLUMN_NAME: stage}, index=(f"{stage}:" + source.obs_names.astype(str)))
    if CELLTYPE_COLUMN_NAME in source.obs.columns:
        obs[CELLTYPE_COLUMN_NAME] = source.obs[CELLTYPE_COLUMN_NAME].astype(str).to_numpy()
    return ad.AnnData(X=matrix, obs=obs, var=pd.DataFrame(index=pd.Index(list(source.var_names))))


def concat_stages(stages: list[ad.AnnData]) -> ad.AnnData:
    combined = ad.concat(stages, join="outer", merge="same")
    combined.obs = combined.obs.fillna("") if CELLTYPE_COLUMN_NAME in combined.obs.columns else combined.obs
    combined.X = sparse.csr_matrix(combined.X, dtype=np.float32)
    return combined


def write_cells(cells: ad.AnnData, path: Path | str) -> None:
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    cells.obs = cells.obs.astype(str)
    # Plain strings, so the h5ad files are readable under both pandas 2 and 3.
    with ad.settings.override(allow_write_nullable_strings=False):
        cells.write_h5ad(path)
