"""Copy a conditional generation dataset, keeping at most --max-cells cells per perturbation in its validation split.

cell-eval2's generic baseline resamples one control cell per measured cell of a context. scipy
counts the stored values of that resample in int32 (scipy issue #12948), so a context with more
than about 2^31 of them cannot be scored: VCC2025 H1 has 376,518 perturbed cells of about 8,400
stored values each. Capping every validation perturbation at 400 cells, the VCC's own number of
cells per perturbation, keeps each context well below that.

Only the validation split's labels.h5ad and input/perturbations.csv change. Each perturbation with
more than --max-cells cells keeps a random --max-cells of them (seeded, so reruns give the same
cells, in their original order). Everything else, including the train split and the validation
controls, is copied unchanged, and dataset_description.md gets a note on the cap. The source is
never modified.

Usage:
    python scripts/cap_validation_cells.py \\
        --source /<path_to_folder>/vcc_agentomics \\
        --output datasets/vcc_agentomics_cap400
"""
from __future__ import annotations

import argparse
import shutil
from pathlib import Path

import anndata as ad
import numpy as np
import pandas as pd

from agentomics.datasets.conditional_generation_contract import (
    CONTEXT_COLUMN_NAME,
    N_CELLS_COLUMN_NAME,
    PERTURBATIONS_FILE_NAME,
    TARGET_GENE_COLUMN_NAME,
    validate_conditional_generation_split,
)
from agentomics.datasets.data_contract import (
    DATASET_DESCRIPTION_FILE_NAME,
    GENERATION_LABELS_FILE_NAME,
    INPUT_DIR_NAME,
    VALIDATION_SPLIT,
)

CONDITION_COLUMNS = [CONTEXT_COLUMN_NAME, TARGET_GENE_COLUMN_NAME]


def capped_rows(obs: pd.DataFrame, max_cells: int, seed: int) -> np.ndarray:
    """Positions of the cells to keep, in their original order: at most max_cells per condition."""
    rng = np.random.default_rng(seed)
    conditions = obs[CONDITION_COLUMNS].astype(str)
    keep = []
    for _, rows in sorted(conditions.groupby(CONDITION_COLUMNS, sort=False).indices.items()):
        keep.append(rng.choice(rows, max_cells, replace=False) if rows.size > max_cells else rows)
    return np.sort(np.concatenate(keep))


def write_capped_split(source_split: Path, output_split: Path, max_cells: int, seed: int) -> pd.DataFrame:
    labels = ad.read_h5ad(source_split / GENERATION_LABELS_FILE_NAME)
    keep = capped_rows(labels.obs, max_cells, seed)
    capped = labels[keep].copy()
    del labels
    (output_split / INPUT_DIR_NAME).mkdir(parents=True)
    # Plain strings, so the h5ad is readable under both pandas 2 and 3.
    with ad.settings.override(allow_write_nullable_strings=False):
        capped.write_h5ad(output_split / GENERATION_LABELS_FILE_NAME, compression="lzf")

    counts = capped.obs[CONDITION_COLUMNS].astype(str).value_counts()
    perturbations = pd.read_csv(source_split / INPUT_DIR_NAME / PERTURBATIONS_FILE_NAME)
    before = perturbations[N_CELLS_COLUMN_NAME].copy()
    perturbations[N_CELLS_COLUMN_NAME] = counts.reindex(
        pd.MultiIndex.from_frame(perturbations[CONDITION_COLUMNS].astype(str))
    ).to_numpy()
    perturbations.to_csv(output_split / INPUT_DIR_NAME / PERTURBATIONS_FILE_NAME, index=False)
    return perturbations.assign(n_cells_before=before)


def copy_unchanged(source: Path, output: Path, changed: set[Path]) -> None:
    """Copy every file except the changed ones, with default permissions so the copy stays writable."""
    for path in sorted(source.rglob("*")):
        relative = path.relative_to(source)
        if path.is_symlink():
            raise ValueError(f"datasets may not contain symlinks: {path}")
        if path.is_dir():
            (output / relative).mkdir(parents=True, exist_ok=True)
        elif relative not in changed:
            print(f"Copying {relative} ({path.stat().st_size / 1e9:.1f} GB)", flush=True)
            shutil.copyfile(path, output / relative)


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("--source", type=Path, required=True, help="Conditional generation dataset to copy")
    parser.add_argument("--output", type=Path, required=True, help="Dataset folder to create (must not exist)")
    parser.add_argument("--max-cells", type=int, default=400, help="Cells kept per validation perturbation")
    parser.add_argument("--seed", type=int, default=0, help="Seed of the cell sampling")
    args = parser.parse_args()

    if args.output.exists():
        parser.error(f"--output already exists: {args.output}")
    source_split = args.source / VALIDATION_SPLIT
    output_split = args.output / VALIDATION_SPLIT

    summary = write_capped_split(source_split, output_split, args.max_cells, args.seed)
    by_context = summary.groupby(CONTEXT_COLUMN_NAME)[[f"{N_CELLS_COLUMN_NAME}_before", N_CELLS_COLUMN_NAME]].sum()
    print(by_context.to_string(), flush=True)

    changed = {
        Path(VALIDATION_SPLIT) / GENERATION_LABELS_FILE_NAME,
        Path(VALIDATION_SPLIT) / INPUT_DIR_NAME / PERTURBATIONS_FILE_NAME,
        Path(DATASET_DESCRIPTION_FILE_NAME),
    }
    copy_unchanged(args.source, args.output, changed)
    description = (args.source / DATASET_DESCRIPTION_FILE_NAME).read_text(encoding="utf-8")
    (args.output / DATASET_DESCRIPTION_FILE_NAME).write_text(
        description.rstrip()
        + f"\n\nValidation perturbations keep at most {args.max_cells} randomly sampled cells each (seed {args.seed}),"
        f" the VCC's number of cells per perturbation; n_cells in validation/input/{PERTURBATIONS_FILE_NAME} gives"
        " the cells kept. The train split and all control cells are complete.\n",
        encoding="utf-8",
    )

    validate_conditional_generation_split(output_split)
    print(f"Wrote {args.output}: validation cells {summary[f'{N_CELLS_COLUMN_NAME}_before'].sum()} -> {summary[N_CELLS_COLUMN_NAME].sum()}")


if __name__ == "__main__":
    main()
