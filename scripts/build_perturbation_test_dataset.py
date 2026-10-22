"""Build a small conditional generation dataset from CRISPRi Perturb-seq screens, for test runs.

Each input h5ad is one screen in one cell line (context), as distributed by scPerturb
(https://doi.org/10.5281/zenodo.13350497): raw counts, one row per cell, and an obs column naming
the knocked-down gene. The dataset keeps target genes knocked down well in every context, a fixed
number of cells per target and of control cells, and the most expressed genes plus the targets.
Validation contexts are held out entirely, as in the Virtual Cell Challenge.

Usage:
    python scripts/build_perturbation_test_dataset.py \\
        --train K562=ReplogleWeissman2022_K562_essential.h5ad RPE1=ReplogleWeissman2022_rpe1.h5ad \\
        --validation HepG2=NadigOConner2024_hepg2.h5ad \\
        --output datasets/perturb_small
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
    TRAIN_SPLIT,
    VALIDATION_SPLIT,
)
from agentomics.datasets.dataset_preparation import check_dataset
from agentomics.utils.task_types import TaskTypes

# Cells per target sampled when estimating knockdown, before the targets are chosen.
KNOCKDOWN_SAMPLE_CELLS = 20


def build(
    contexts: dict[str, dict[str, Path]],
    output_dir: Path,
    *,
    perturbation_column: str,
    control_label: str,
    n_targets: int,
    cells_per_target: int,
    n_controls: int,
    n_genes: int,
    max_knockdown_ratio: float,
    seed: int,
) -> None:
    rng = np.random.default_rng(seed)
    screens = {
        context: ad.read_h5ad(path, backed="r")
        for split_contexts in contexts.values()
        for context, path in split_contexts.items()
    }
    genes = _common_genes(screens)
    candidates = _common_targets(screens, genes, perturbation_column, control_label, cells_per_target)
    print(f"{len(genes)} common genes, {len(candidates)} targets with >= {cells_per_target} cells in every context")

    ratios, control_means = {}, {}
    for context, screen in screens.items():
        sample = _load_cells(screen, _sample_rows(screen, candidates, KNOCKDOWN_SAMPLE_CELLS, n_controls, perturbation_column, control_label, rng), genes)
        ratios[context], control_means[context] = _knockdown_ratios(sample, candidates, perturbation_column, control_label)
    knocked_down = [target for target in candidates if all(ratios[context][target] <= max_knockdown_ratio for context in screens)]
    print(f"{len(knocked_down)} targets knocked down to <= {max_knockdown_ratio:.0%} of control expression in every context")
    if len(knocked_down) < n_targets:
        raise ValueError(f"only {len(knocked_down)} targets qualify, fewer than --n-targets {n_targets}")
    targets = sorted(rng.choice(knocked_down, size=n_targets, replace=False))

    pooled_means = pd.concat(control_means.values(), axis=1).mean(axis=1)
    top_genes = set(pooled_means.nlargest(n_genes).index) | set(targets)
    kept_genes = [gene for gene in genes if gene in top_genes]
    print(f"Keeping {len(targets)} targets and {len(kept_genes)} genes")

    output_dir.mkdir(parents=True)
    for split_name, split_contexts in contexts.items():
        perturbed_parts, control_parts = [], []
        for context in split_contexts:
            screen = screens[context]
            cells = _load_cells(
                screen,
                _sample_rows(screen, targets, cells_per_target, n_controls, perturbation_column, control_label, rng),
                kept_genes,
            )
            cells.obs = pd.DataFrame(
                {
                    CONTEXT_COLUMN_NAME: context,
                    TARGET_GENE_COLUMN_NAME: cells.obs[perturbation_column]
                    .astype(str)
                    .replace(control_label, CONTROL_LABEL)
                    .to_numpy(),
                },
                index=[f"{context}_{name}" for name in cells.obs_names],
            )
            is_control = (cells.obs[TARGET_GENE_COLUMN_NAME] == CONTROL_LABEL).to_numpy()
            control_parts.append(cells[is_control])
            perturbed_parts.append(cells[~is_control])
        _write_split(
            output_dir / split_name,
            ad.concat(perturbed_parts),
            ad.concat(control_parts),
            [(context, target) for context in split_contexts for target in targets],
            cells_per_target,
        )

    (output_dir / METADATA_FILE_NAME).write_text(
        json.dumps({"task_type": TaskTypes.CONDITIONAL_GENERATION}, indent=4) + "\n", encoding="utf-8"
    )
    (output_dir / DATASET_DESCRIPTION_FILE_NAME).write_text(
        _description(contexts, targets, kept_genes, cells_per_target, n_controls), encoding="utf-8"
    )


def _common_genes(screens: dict[str, ad.AnnData]) -> list[str]:
    first, *others = screens.values()
    shared = set(first.var_names).intersection(*(set(screen.var_names) for screen in others))
    return [gene for gene in first.var_names if gene in shared]


def _common_targets(screens, genes, perturbation_column, control_label, min_cells) -> list[str]:
    gene_set = set(genes)
    common = None
    for screen in screens.values():
        counts = screen.obs[perturbation_column].astype(str).value_counts()
        eligible = {
            target for target, count in counts.items()
            if count >= min_cells and target != control_label and target in gene_set
        }
        common = eligible if common is None else common & eligible
    return sorted(common)


def _sample_rows(screen, targets, cells_per_target, n_controls, perturbation_column, control_label, rng) -> np.ndarray:
    labels = screen.obs[perturbation_column].astype(str).to_numpy()
    rows = []
    for label, count in [(control_label, n_controls), *((target, cells_per_target) for target in targets)]:
        label_rows = np.flatnonzero(labels == label)
        if len(label_rows) < count:
            raise ValueError(f"only {len(label_rows)} '{label}' cells, need {count}")
        rows.append(rng.choice(label_rows, size=count, replace=False))
    return np.sort(np.concatenate(rows))


def _load_cells(screen, rows: np.ndarray, genes: list[str]) -> ad.AnnData:
    cells = screen[rows].to_memory()[:, genes].copy()
    cells.X = sparse.csr_matrix(cells.X, dtype=np.float32)
    return cells


def _knockdown_ratios(cells, targets, perturbation_column, control_label) -> tuple[dict[str, float], pd.Series]:
    """Mean normalized expression of each target in its own knockdown cells, relative to the controls."""
    totals = np.asarray(cells.X.sum(axis=1)).ravel()
    normalized = sparse.diags(1e4 / np.maximum(totals, 1)) @ cells.X
    labels = cells.obs[perturbation_column].astype(str).to_numpy()
    control_means = pd.Series(np.asarray(normalized[labels == control_label].mean(axis=0)).ravel(), index=cells.var_names)
    gene_index = {gene: i for i, gene in enumerate(cells.var_names)}
    ratios = {}
    for target in targets:
        knocked = normalized[labels == target][:, gene_index[target]].mean()
        ratios[target] = knocked / control_means[target] if control_means[target] > 0 else np.inf
    return ratios, control_means


def _write_split(split_dir: Path, perturbed, controls, conditions, cells_per_target: int) -> None:
    (split_dir / INPUT_DIR_NAME).mkdir(parents=True)
    # Plain strings, so the h5ad files are readable under both pandas 2 and 3.
    with ad.settings.override(allow_write_nullable_strings=False):
        controls.write_h5ad(split_dir / INPUT_DIR_NAME / CONTROLS_FILE_NAME, compression="gzip")
        perturbed.write_h5ad(split_dir / GENERATION_LABELS_FILE_NAME, compression="gzip")
    contexts, targets = zip(*conditions)
    pd.DataFrame(
        {CONTEXT_COLUMN_NAME: contexts, TARGET_GENE_COLUMN_NAME: targets, N_CELLS_COLUMN_NAME: cells_per_target}
    ).to_csv(split_dir / INPUT_DIR_NAME / PERTURBATIONS_FILE_NAME, index=False)
    pd.DataFrame(
        {ID_COLUMN_NAME: condition_ids(contexts, targets), LABEL_COLUMN_NAME: cells_per_target}
    ).to_csv(split_dir / LABELS_FILE_NAME, index=False)


def _description(contexts, targets, genes, cells_per_target, n_controls) -> str:
    train = ", ".join(contexts[TRAIN_SPLIT])
    validation = ", ".join(contexts[VALIDATION_SPLIT])
    return f"""# Small CRISPRi Perturb-seq dataset (conditional generation)

Cells from CRISPRi Perturb-seq screens of essential genes (Replogle et al. 2022; Nadig et al. 2025),
harmonized by scPerturb. Each cell line is one context.

## Task
Given the control cells of a context and a target gene, generate the cells that would be measured
after knocking that gene down with CRISPRi. Validation contexts ({validation}) are never seen
perturbed during training ({train}), as in the Virtual Cell Challenge.

## Contents
- {len(targets)} target genes, the same in every context, each with {cells_per_target} measured cells per context.
- {n_controls} control (non-targeting) cells per context.
- {len(genes)} genes: the most expressed ones plus every target gene. Values are raw UMI counts.

## Data format
- input/controls.h5ad: control cells; obs 'context' and 'target_gene' ('non-targeting').
- input/perturbations.csv: the conditions (context, target_gene, n_cells) to generate cells for.
- labels.csv: one row per condition; its id '<context>|<target_gene>' and its n_cells.
- labels.h5ad: the measured cells of every condition; obs 'context' and 'target_gene'.
"""


def _parse_contexts(values: list[str]) -> dict[str, Path]:
    contexts = {}
    for value in values:
        name, _, path = value.partition("=")
        if not name or not path:
            raise argparse.ArgumentTypeError(f"expected CONTEXT=PATH, got '{value}'")
        contexts[name] = Path(path)
    return contexts


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("--train", nargs="+", required=True, metavar="CONTEXT=PATH", help="Training screens")
    parser.add_argument("--validation", nargs="+", required=True, metavar="CONTEXT=PATH", help="Held-out screens")
    parser.add_argument("--output", type=Path, required=True, help="Dataset folder to create (must not exist)")
    parser.add_argument("--perturbation-column", default="perturbation", help="obs column with the knocked-down gene")
    parser.add_argument("--control-label", default="control", help="Label of control cells in that column")
    parser.add_argument("--n-targets", type=int, default=40)
    parser.add_argument("--cells-per-target", type=int, default=50)
    parser.add_argument("--n-controls", type=int, default=1000)
    parser.add_argument("--n-genes", type=int, default=2000, help="Most expressed genes kept (targets are always kept)")
    parser.add_argument("--max-knockdown-ratio", type=float, default=0.5,
                        help="Keep targets whose expression in knockdown cells is at most this fraction of controls")
    parser.add_argument("--seed", type=int, default=0)
    args = parser.parse_args()

    if args.output.exists():
        parser.error(f"--output already exists: {args.output}")
    contexts = {TRAIN_SPLIT: _parse_contexts(args.train), VALIDATION_SPLIT: _parse_contexts(args.validation)}
    build(
        contexts,
        args.output,
        perturbation_column=args.perturbation_column,
        control_label=args.control_label,
        n_targets=args.n_targets,
        cells_per_target=args.cells_per_target,
        n_controls=args.n_controls,
        n_genes=args.n_genes,
        max_knockdown_ratio=args.max_knockdown_ratio,
        seed=args.seed,
    )
    summary = check_dataset(args.output)
    print(f"Dataset written to {args.output} and checked: {summary['splits']}")


if __name__ == "__main__":
    main()
