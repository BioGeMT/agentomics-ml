"""Build a conditional generation dataset from the Virtual Cell Challenge 2025 H1 data.

The source is Arc's public VCC 2025 release: CRISPRi Perturb-seq in H1 hESCs, already split by
target gene into train (150 targets), validation (50) and test (100), each file with its own
control cells. Download it with:

    B=https://storage.googleapis.com/arc-institute-virtual-cell-atlas/virtual-cell-challenge/2025
    wget -c -P train $B/train/adata_Training.h5ad
    wget -c -P validation $B/validation/adata_Validation.h5ad
    wget -c -P test $B/test/adata_Test.h5ad

Each split keeps Arc's targets, so the dataset has a single context ('H1'). Cells are put on the
VCC 2026 gene axis, so models output the genes a 2026 submission needs; 2026 genes H1 does not
measure are all zero. Counts are binomially downsampled to the 2026 evaluation depth.

Usage:
    python scripts/build_vcc2025_h1_dataset.py \\
        --source ~/vcc_data/2025 \\
        --genes vcc_2026_controls/gene_names.csv \\
        --output datasets/vcc2025_h1
"""
from __future__ import annotations

import argparse
import json
from pathlib import Path

import anndata as ad
import numpy as np
import pandas as pd
from scipy import sparse

from agentomics.datasets.conditional_generation_contract import (
    CONTEXT_COLUMN_NAME,
    CONTROL_LABEL,
    CONTROLS_FILE_NAME,
    N_CELLS_COLUMN_NAME,
    PERTURBATIONS_FILE_NAME,
    TARGET_GENE_COLUMN_NAME,
    condition_ids,
)
from agentomics.datasets.data_contract import (
    DATASET_DESCRIPTION_FILE_NAME,
    GENERATION_LABELS_FILE_NAME,
    ID_COLUMN_NAME,
    INPUT_DIR_NAME,
    LABEL_COLUMN_NAME,
    LABELS_FILE_NAME,
    METADATA_FILE_NAME,
    TEST_SPLIT_PREFIX,
    TRAIN_SPLIT,
    VALIDATION_SPLIT,
)
from agentomics.datasets.dataset_preparation import check_dataset
from agentomics.utils.task_types import TaskTypes

CONTEXT = "H1"
SOURCE_FILES = {
    TRAIN_SPLIT: "train/adata_Training.h5ad",
    VALIDATION_SPLIT: "validation/adata_Validation.h5ad",
    TEST_SPLIT_PREFIX: "test/adata_Test.h5ad",
}
# Rows read from the backed source at a time, to bound memory.
CHUNK_ROWS = 10_000


def build(
    source_dir: Path,
    genes: list[str],
    output_dir: Path,
    *,
    cells_per_target: int,
    n_controls: int,
    target_median_umi: int,
    seed: int,
) -> None:
    rng = np.random.default_rng(seed)
    output_dir.mkdir(parents=True)
    split_summaries = {}
    for split_name, relative_path in SOURCE_FILES.items():
        source = ad.read_h5ad(source_dir / relative_path, backed="r")
        targets = source.obs[TARGET_GENE_COLUMN_NAME].astype(str).to_numpy()
        control_rows = _sample(np.flatnonzero(targets == CONTROL_LABEL), n_controls, rng)
        perturbed_rows = np.concatenate([
            _sample(np.flatnonzero(targets == target), cells_per_target, rng)
            for target in sorted(set(targets) - {CONTROL_LABEL})
        ])
        rows = np.sort(np.concatenate([control_rows, perturbed_rows]))

        counts = _to_gene_axis(_read_counts(source, rows), source.var_names, genes)
        totals = np.asarray(counts.sum(axis=1)).ravel()
        fraction = min(1.0, target_median_umi / np.median(totals)) if target_median_umi else 1.0
        if fraction < 1.0:
            counts.data = rng.binomial(counts.data.astype(np.int64), fraction).astype(np.float32)
            counts.eliminate_zeros()

        cells = ad.AnnData(
            X=counts,
            obs=pd.DataFrame(
                {CONTEXT_COLUMN_NAME: CONTEXT, TARGET_GENE_COLUMN_NAME: targets[rows]},
                index=source.obs_names[rows].astype(str),
            ),
            var=pd.DataFrame(index=pd.Index(genes)),
        )
        source.file.close()
        is_control = (cells.obs[TARGET_GENE_COLUMN_NAME] == CONTROL_LABEL).to_numpy()
        _write_split(output_dir / split_name, cells[~is_control].copy(), cells[is_control].copy())
        split_summaries[split_name] = (
            cells.obs.loc[~is_control, TARGET_GENE_COLUMN_NAME].nunique(),
            int(is_control.sum()),
            float(np.median(np.asarray(cells.X.sum(axis=1)).ravel())),
        )
        print(f"{split_name}: {split_summaries[split_name][0]} targets, {cells.n_obs} cells, "
              f"downsampling fraction {fraction:.3f}", flush=True)

    (output_dir / METADATA_FILE_NAME).write_text(
        json.dumps({"task_type": TaskTypes.CONDITIONAL_GENERATION}, indent=4) + "\n", encoding="utf-8"
    )
    (output_dir / DATASET_DESCRIPTION_FILE_NAME).write_text(
        _description(split_summaries, genes, cells_per_target), encoding="utf-8"
    )


def _sample(rows: np.ndarray, count: int, rng) -> np.ndarray:
    return rows if len(rows) <= count else rng.choice(rows, size=count, replace=False)


def _read_counts(source, rows: np.ndarray) -> sparse.csr_matrix:
    chunks = [sparse.csr_matrix(source.X[rows[start:start + CHUNK_ROWS]]) for start in range(0, len(rows), CHUNK_ROWS)]
    counts = sparse.vstack(chunks, format="csr", dtype=np.float32)
    if (counts.data < 0).any() or (counts.data != np.round(counts.data)).any():
        raise ValueError("source counts must be non-negative whole numbers")
    return counts


def _to_gene_axis(counts: sparse.csr_matrix, source_genes, genes: list[str]) -> sparse.csr_matrix:
    """Reorders the columns to `genes`; genes missing from the source become all-zero columns."""
    positions = pd.Index(source_genes).get_indexer(genes)
    present = positions >= 0
    selection = sparse.csr_matrix(
        (np.ones(present.sum(), dtype=np.float32), (positions[present], np.flatnonzero(present))),
        shape=(len(source_genes), len(genes)),
    )
    return (counts @ selection).tocsr()


def _write_split(split_dir: Path, perturbed, controls) -> None:
    (split_dir / INPUT_DIR_NAME).mkdir(parents=True)
    # Plain strings, so the h5ad files are readable under both pandas 2 and 3.
    with ad.settings.override(allow_write_nullable_strings=False):
        controls.write_h5ad(split_dir / INPUT_DIR_NAME / CONTROLS_FILE_NAME, compression="gzip")
        perturbed.write_h5ad(split_dir / GENERATION_LABELS_FILE_NAME, compression="gzip")
    n_cells = perturbed.obs[TARGET_GENE_COLUMN_NAME].value_counts().sort_index()
    contexts = [CONTEXT] * len(n_cells)
    pd.DataFrame(
        {CONTEXT_COLUMN_NAME: contexts, TARGET_GENE_COLUMN_NAME: n_cells.index, N_CELLS_COLUMN_NAME: n_cells.to_numpy()}
    ).to_csv(split_dir / INPUT_DIR_NAME / PERTURBATIONS_FILE_NAME, index=False)
    pd.DataFrame(
        {ID_COLUMN_NAME: condition_ids(contexts, n_cells.index), LABEL_COLUMN_NAME: n_cells.to_numpy()}
    ).to_csv(split_dir / LABELS_FILE_NAME, index=False)


def _description(split_summaries, genes, cells_per_target) -> str:
    train_targets, n_controls, median_umi = split_summaries[TRAIN_SPLIT]
    validation_targets = split_summaries[VALIDATION_SPLIT][0]
    return f"""# Virtual Cell Challenge 2025 H1 CRISPRi Perturb-seq (conditional generation)

Arc Institute's Virtual Cell Challenge 2025 data: H1 human embryonic stem cells, CRISPRi knockdowns
profiled with 10x Flex. All cells belong to one context, 'H1'.

## Task
Given the control cells of a context and a target gene, generate the cells that would be measured
after knocking that gene down with CRISPRi. The splits are Arc's 2025 split by target gene: the
validation targets ({validation_targets}) are never seen during training ({train_targets} targets).

## Contents
- Up to {cells_per_target} measured cells per target gene; n_cells in perturbations.csv gives the exact number.
- {n_controls} control (non-targeting) cells per split.
- {len(genes)} genes, in a fixed order: the gene panel of the Virtual Cell Challenge 2026. Genes not measured
  in H1 are all zero.
- Values are raw UMI counts, downsampled to a median of about {median_umi:,.0f} UMIs per cell.

## Data format
- input/controls.h5ad: control cells; obs 'context' and 'target_gene' ('non-targeting').
- input/perturbations.csv: the conditions (context, target_gene, n_cells) to generate cells for.
- labels.csv: one row per condition; its id '<context>|<target_gene>' and its n_cells.
- labels.h5ad: the measured cells of every condition; obs 'context' and 'target_gene'.
"""


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("--source", type=Path, required=True, help="Folder with the VCC 2025 train/, validation/ and test/ downloads")
    parser.add_argument("--genes", type=Path, required=True, help="VCC 2026 gene_names.csv (the output gene axis)")
    parser.add_argument("--output", type=Path, required=True, help="Dataset folder to create (must not exist)")
    parser.add_argument("--cells-per-target", type=int, default=400)
    parser.add_argument("--n-controls", type=int, default=18_400, help="Control cells kept per split (VCC 2026 has 18,400 per context)")
    parser.add_argument("--target-median-umi", type=int, default=20_000, help="Downsample counts to this median UMIs per cell (0 keeps the source depth)")
    parser.add_argument("--seed", type=int, default=0)
    args = parser.parse_args()

    if args.output.exists():
        parser.error(f"--output already exists: {args.output}")
    genes = pd.read_csv(args.genes).iloc[:, 0].astype(str).tolist()
    build(
        args.source.expanduser(),
        genes,
        args.output,
        cells_per_target=args.cells_per_target,
        n_controls=args.n_controls,
        target_median_umi=args.target_median_umi,
        seed=args.seed,
    )
    summary = check_dataset(args.output)
    print(f"Dataset written to {args.output} and checked: {summary['splits']}, test rows: {summary.get('test_rows')}")


if __name__ == "__main__":
    main()
