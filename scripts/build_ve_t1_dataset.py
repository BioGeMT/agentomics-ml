"""Build a temporal generation dataset for the Virtual Embryo Challenge 2026, Task 1.

The source is the challenge's T1 training data (T1/E8.5_RNA.h5ad and T1/E9.5_RNA.h5ad, downloaded with the
challenge API) and, once requested, the E10.5 validation answers. The dataset has a single train/ split:
    input/controls.h5ad      the earliest stage (E8.5): the observed cells
    input/perturbations.csv  the later stages as targets (E9.5, E10.5), with n_cells each
    labels.h5ad              the measured cells of those targets
The agent splits it by target stage itself (e.g. train E8.5 -> E9.5, validation E8.5 + E9.5 -> E10.5).
The challenge's E12.5 test is forecast by scripts/ve_t1_submission.py, from every observed stage.

Without the E10.5 answers, --mock-e105 adds a synthetic E10.5 (E9.5 shifted once more by the E8.5 -> E9.5
mean change) so the pipeline can be exercised end to end. Its scores mean nothing biologically.

Usage:
    python scripts/build_ve_t1_dataset.py --source datasets/virtual_embryo_t1 \\
        --e105 <E10.5 validation answers .h5ad> --output datasets/ve_t1
"""
from __future__ import annotations

import argparse
import json
from pathlib import Path

import anndata as ad
import numpy as np
import pandas as pd

from agentomics.datasets.conditional_generation_contract import (
    CONTROLS_FILE_NAME,
    N_CELLS_COLUMN_NAME,
    PERTURBATIONS_FILE_NAME,
)
from agentomics.datasets.data_contract import (
    DATASET_DESCRIPTION_FILE_NAME,
    GENERATION_LABELS_FILE_NAME,
    INPUT_DIR_NAME,
    METADATA_FILE_NAME,
    TRAIN_SPLIT,
)
from agentomics.datasets.dataset_preparation import check_dataset
from agentomics.datasets.temporal_generation_contract import STAGE_COLUMN_NAME
from agentomics.datasets.virtual_embryo import DEFAULT_N_CELLS, concat_stages, load_stage, write_cells
from agentomics.utils.task_types import TaskTypes

TRAINING_FILES = {"E8.5": "E8.5_RNA.h5ad", "E9.5": "E9.5_RNA.h5ad"}


def build(source_dir: Path, output_dir: Path, *, e105: Path | None, mock_e105: bool, n_cells: int, seed: int) -> None:
    observed = load_stage(source_dir / TRAINING_FILES["E8.5"], "E8.5")
    genes = list(observed.var_names)
    targets = [load_stage(source_dir / TRAINING_FILES["E9.5"], "E9.5", genes)]
    if e105 is not None:
        targets.append(load_stage(e105, "E10.5", genes))
    elif mock_e105:
        targets.append(_mock_next_stage(observed, targets[0], "E10.5", seed))

    split_dir = output_dir / TRAIN_SPLIT
    write_cells(observed, split_dir / INPUT_DIR_NAME / CONTROLS_FILE_NAME)
    labels = concat_stages(targets)
    write_cells(labels, split_dir / GENERATION_LABELS_FILE_NAME)
    counts = labels.obs[STAGE_COLUMN_NAME].value_counts()
    conditions = pd.DataFrame(
        [(target.obs[STAGE_COLUMN_NAME].iloc[0], min(n_cells, int(counts[target.obs[STAGE_COLUMN_NAME].iloc[0]])))
         for target in targets],
        columns=[STAGE_COLUMN_NAME, N_CELLS_COLUMN_NAME],
    )
    conditions.to_csv(split_dir / INPUT_DIR_NAME / PERTURBATIONS_FILE_NAME, index=False)

    (output_dir / METADATA_FILE_NAME).write_text(
        json.dumps({"task_type": TaskTypes.TEMPORAL_GENERATION, "mock_e105": bool(mock_e105 and e105 is None)}, indent=4)
        + "\n",
        encoding="utf-8",
    )
    (output_dir / DATASET_DESCRIPTION_FILE_NAME).write_text(
        _description(observed, targets, genes, mock=mock_e105 and e105 is None), encoding="utf-8"
    )
    for target in targets:
        print(f"target {target.obs[STAGE_COLUMN_NAME].iloc[0]}: {target.n_obs} cells", flush=True)


def _mock_next_stage(previous: ad.AnnData, current: ad.AnnData, stage: str, seed: int) -> ad.AnnData:
    """current shifted by its mean change from previous (on expressed entries), clipped at zero. Testing only."""
    shift = np.asarray(current.X.mean(axis=0)).ravel() - np.asarray(previous.X.mean(axis=0)).ravel()
    rng = np.random.default_rng(seed)
    rows = np.sort(rng.choice(current.n_obs, size=current.n_obs // 2, replace=False))
    mock = current[rows].copy()
    mock.X.data = np.clip(mock.X.data + shift[mock.X.indices], 0, None).astype(np.float32)
    mock.X.eliminate_zeros()
    mock.obs[STAGE_COLUMN_NAME] = stage
    mock.obs_names = [f"{stage}:mock_{i}" for i in range(mock.n_obs)]
    if "celltype" in mock.obs.columns:
        mock.obs["celltype"] = ""
    return mock


def _description(observed: ad.AnnData, targets: list[ad.AnnData], genes: list[str], mock: bool) -> str:
    def cell_types(cells: ad.AnnData) -> str:
        if "celltype" not in cells.obs.columns or (cells.obs["celltype"] == "").all():
            return "not annotated"
        counts = cells.obs["celltype"].value_counts()
        return ", ".join(f"{name} ({count})" for name, count in counts.items())

    stages = [observed, *targets]
    stage_lines = "\n".join(
        f"- {cells.obs[STAGE_COLUMN_NAME].iloc[0]}: {cells.n_obs} cells; cell types: {cell_types(cells)}"
        for cells in stages
    )
    mock_note = (
        "\n**The E10.5 stage of this dataset is a synthetic mock** (E9.5 shifted once more by the E8.5 -> E9.5 mean "
        "change), built only to exercise the pipeline before the real E10.5 data is available. It is not biological.\n"
        if mock else ""
    )
    return f"""# Virtual Embryo Challenge 2026, Task 1: temporal prediction of the mouse embryo transcriptome
{mock_note}
Real dissociated single-cell RNA (10x Multiome, RNA modality) of mouse embryos, centred on heart development
(cardiomyocyte subtypes, second heart field, endocardium, pericardium, foregut, ...). Each stage comes from
different embryos: cells are not paired across stages.

## Task
Given the cells of earlier developmental stages, generate the cell population of a later, unobserved stage.
The final model forecasts E12.5 from every stage of this dataset; nothing is observed between the last stage
here and E12.5, so there is no bracketing stage to interpolate from. The leaderboard stage is E10.5.

## Stages
{stage_lines}

Consecutive stages share only a few annotated cell types; about two thirds of later cells are of types absent
earlier. Cell-type labels are the source annotation; predictions need none (the challenge types predicted cells
with its own frozen classifier).

## Contents
- {len(genes)} genes (whole transcriptome, gene symbols), in the released order, which predictions must keep.
- Values are log1p-normalised expression (cell totals around 3,500-4,200), not raw counts; about 87% zeros.
- n_cells in perturbations.csv is the number of cells to generate per target stage (the challenge accepts
  1,000-5,118 cells for E10.5 and 638-5,118 for E12.5; its heaviest metrics subsample about 2,000 cells).

## Scoring (Virtual Embryo Challenge, Task 1)
DE gene recovery 25%, change direction 25%, cell-state distribution (MMD) 30%, gene-gene co-variation 20%,
each rescaled so that copy_last (predicting the latest observed stage unchanged) = 50 and a split-half
replicate of the real cells = 100. The organisers report that their neural ODE baseline does not beat a constant
per-cell-type mean shift, and that predicting one average cell scores near nothing on the population metrics.

## Data format
- input/controls.h5ad: observed cells; obs 'stage' (e.g. 'E8.5') and 'celltype'. Sparse CSR float32.
- input/perturbations.csv: the target stages (stage, n_cells) to generate cells for.
- labels.h5ad: the measured cells of every target stage; obs 'stage' and 'celltype'.
"""


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("--source", type=Path, required=True, help="Folder with E8.5_RNA.h5ad and E9.5_RNA.h5ad")
    parser.add_argument("--output", type=Path, required=True, help="Dataset folder to create (must not exist)")
    e105 = parser.add_mutually_exclusive_group()
    e105.add_argument("--e105", type=Path, help="The E10.5 validation answers (.h5ad, log1p, released genes)")
    e105.add_argument("--mock-e105", action="store_true", help="Add a synthetic E10.5, for testing the pipeline only")
    parser.add_argument("--n-cells", type=int, default=DEFAULT_N_CELLS, help="Cells to generate per target stage (at most the real ones)")
    parser.add_argument("--seed", type=int, default=0)
    args = parser.parse_args()

    if args.output.exists():
        parser.error(f"--output already exists: {args.output}")
    build(args.source.expanduser(), args.output, e105=args.e105, mock_e105=args.mock_e105, n_cells=args.n_cells, seed=args.seed)
    summary = check_dataset(args.output)
    print(f"Dataset written to {args.output} and checked: {summary['splits']}")


if __name__ == "__main__":
    main()
