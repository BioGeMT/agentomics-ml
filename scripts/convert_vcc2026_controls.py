"""Convert the VCC 2026 control download into a conditional generation input folder.

The download holds only control cells, so the result has no labels and cannot be a train or
validation split. It is the input to predict the challenge perturbations from:

    <output>/input/controls.h5ad      control cells of every context
    <output>/input/perturbations.csv  every panel gene in every context, with the panel's cells per perturbation

Usage:
    python scripts/convert_vcc2026_controls.py --source vcc_2026_controls --output <folder>
    agentomics-inference --agent-dir <run> --input <folder> --output predictions.h5ad
"""
from __future__ import annotations

import argparse
import itertools
import json
from pathlib import Path

import anndata as ad
import pandas as pd

from agentomics.datasets.conditional_generation_contract import (
    CONTEXT_COLUMN_NAME,
    CONTROL_LABEL,
    CONTROLS_FILE_NAME,
    N_CELLS_COLUMN_NAME,
    PERTURBATIONS_FILE_NAME,
    TARGET_GENE_COLUMN_NAME,
    read_perturbations,
)
from agentomics.datasets.data_contract import INPUT_DIR_NAME


def convert(source_dir: Path, output_dir: Path) -> None:
    manifest = json.loads((source_dir / "manifest.json").read_text(encoding="utf-8"))
    expected_conventions = {
        "context_col": CONTEXT_COLUMN_NAME,
        "pert_col": TARGET_GENE_COLUMN_NAME,
        "control_label": CONTROL_LABEL,
    }
    for key, expected in expected_conventions.items():
        if manifest[key] != expected:
            raise ValueError(f"manifest {key} is '{manifest[key]}', expected '{expected}'")

    genes = pd.read_csv(source_dir / "gene_names.csv")["gene_name"].tolist()
    targets = pd.read_csv(source_dir / "pert_counts.csv")[TARGET_GENE_COLUMN_NAME].tolist()
    if len(targets) != manifest["n_constructs"]:
        raise ValueError(f"pert_counts.csv lists {len(targets)} genes, manifest says {manifest['n_constructs']}")

    controls = []
    for context in manifest["contexts"]:
        context_controls = ad.read_h5ad(source_dir / f"context_{context}.h5ad")
        if list(context_controls.var_names) != genes:
            raise ValueError(f"context_{context}.h5ad genes differ from gene_names.csv")
        if set(context_controls.obs[CONTEXT_COLUMN_NAME]) != {context}:
            raise ValueError(f"context_{context}.h5ad holds cells of other contexts")
        controls.append(context_controls)

    input_dir = output_dir / INPUT_DIR_NAME
    input_dir.mkdir(parents=True)
    # Plain strings, so the h5ad is readable under both pandas 2 and 3.
    with ad.settings.override(allow_write_nullable_strings=False):
        ad.concat(controls).write_h5ad(input_dir / CONTROLS_FILE_NAME)

    pd.DataFrame(
        [
            (context, target, manifest["cells_per_pert"])
            for context, target in itertools.product(manifest["contexts"], targets)
        ],
        columns=[CONTEXT_COLUMN_NAME, TARGET_GENE_COLUMN_NAME, N_CELLS_COLUMN_NAME],
    ).to_csv(input_dir / PERTURBATIONS_FILE_NAME, index=False)
    read_perturbations(input_dir / PERTURBATIONS_FILE_NAME)


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("--source", type=Path, required=True, help="Folder of the VCC 2026 control download")
    parser.add_argument("--output", type=Path, required=True, help="Folder to create (must not exist)")
    args = parser.parse_args()

    if args.output.exists():
        parser.error(f"--output already exists: {args.output}")
    convert(args.source, args.output)
    print(f"Wrote {args.output / INPUT_DIR_NAME}")


if __name__ == "__main__":
    main()
