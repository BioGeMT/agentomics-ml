"""Generate the VCC 2026 final-phase prediction with a trained Agentomics model and package it for submission.

Steps, each in its own folder under --work-dir:
    1. input/       the model's input: the D/E/F controls and the official perturbations, with
                    perturbations.csv listing every panel gene in every context with the panel's cells per perturbation
    2. generated.h5ad   agentomics-inference on that input (skipped with --predictions)
    3. pred.h5ad    the generated cells in the challenge's format: contexts D/E/F, sparse raw counts
    4. pred.vcc     `vcc prep` of pred.h5ad (only validated with --dry-run)
Submitting is left to you: `vcc submit <work-dir>/pred.vcc -m "..." --wait`. Only the last final entry counts.

The controls come from the dataset builder's harmonised files, which carry the obs columns the model was trained
with. They are checked against the official download (same cells, genes and counts) before anything is built.
Contexts are named like the dataset's, <dataset>__<cell_line>__<condition> (e.g. VCC2026_inputs__D__none).

Usage:
    python scripts/vcc2026_submission.py \\
        --vcc-dir <unzipped controls-final download> \\
        --controls-dir <folder with VCC2026_inputs__{D,E,F}.h5ad> \\
        --dataset-dir <agentomics dataset the model was trained on> \\
        --agent-dir outputs/<run> --work-dir <empty or new folder> [--dev]
"""
from __future__ import annotations

import argparse
import itertools
import json
import os
import shutil
import subprocess
import sys
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
    read_perturbations,
    validate_generated_cells,
)
from agentomics.datasets.data_contract import INPUT_DIR_NAME, VALIDATION_SPLIT

GENERATED_FILE_NAME = "generated.h5ad"
PREDICTION_FILE_NAME = "pred.h5ad"
PACKAGE_FILE_NAME = "pred.vcc"
DEFAULT_VCC = Path.home() / "miniconda3/envs/vcc/bin/vcc"


def load_official(vcc_dir: Path) -> tuple[dict, list[str], list[str]]:
    manifest = json.loads((vcc_dir / "manifest.json").read_text(encoding="utf-8"))
    expected_conventions = {
        "context_col": CONTEXT_COLUMN_NAME,
        "pert_col": TARGET_GENE_COLUMN_NAME,
        "control_label": CONTROL_LABEL,
    }
    for key, expected in expected_conventions.items():
        if manifest[key] != expected:
            raise ValueError(f"manifest {key} is '{manifest[key]}', expected '{expected}'")
    genes = pd.read_csv(vcc_dir / "gene_names.csv")["gene_name"].tolist()
    targets = pd.read_csv(vcc_dir / "pert_counts.csv")[TARGET_GENE_COLUMN_NAME].tolist()
    if len(genes) != manifest["n_genes"]:
        raise ValueError(f"gene_names.csv lists {len(genes)} genes, manifest says {manifest['n_genes']}")
    if len(targets) != manifest["n_constructs"]:
        raise ValueError(f"pert_counts.csv lists {len(targets)} genes, manifest says {manifest['n_constructs']}")
    return manifest, genes, targets


def harmonised_controls(vcc_dir: Path, controls_dir: Path, context: str, genes: list[str]) -> ad.AnnData:
    """The builder's harmonised controls of one context, after checking they are the official ones."""
    official = ad.read_h5ad(vcc_dir / f"context_{context}.h5ad")
    controls = ad.read_h5ad(controls_dir / f"VCC2026_inputs__{context}.h5ad")
    if list(official.var_names) != genes or list(controls.var_names) != genes:
        raise ValueError(f"context {context}: control genes differ from gene_names.csv")
    if set(official.obs[CONTEXT_COLUMN_NAME]) != {context} or set(controls.obs["cell_line"]) != {context}:
        raise ValueError(f"context {context}: control files hold cells of other contexts")
    if (controls.obs[TARGET_GENE_COLUMN_NAME] != CONTROL_LABEL).any():
        raise ValueError(f"context {context}: harmonised controls include non-control cells")
    source_ids = pd.Index(controls.obs["source_cell_id"].astype(str))
    if set(source_ids) != set(official.obs_names) or not source_ids.is_unique:
        raise ValueError(f"context {context}: harmonised and official controls are not the same cells")
    official_counts = sparse.csr_matrix(official.X)[official.obs_names.get_indexer(source_ids)]
    if (sparse.csr_matrix(controls.X) != official_counts).nnz:
        raise ValueError(f"context {context}: harmonised and official control counts differ")
    return controls


def build_input(vcc_dir: Path, controls_dir: Path, dataset_dir: Path, input_dir: Path) -> dict[str, str]:
    """Writes the model's input folder and returns each context's name in it, by challenge context."""
    manifest, genes, targets = load_official(vcc_dir)
    # The model was trained on these obs columns; columns the challenge controls lack are left missing.
    dataset_columns = ad.read_h5ad(
        dataset_dir / VALIDATION_SPLIT / INPUT_DIR_NAME / CONTROLS_FILE_NAME, backed="r"
    ).obs.columns

    names, controls = {}, []
    for context in manifest["contexts"]:
        context_controls = harmonised_controls(vcc_dir, controls_dir, context, genes)
        obs = context_controls.obs
        (name,) = set(obs["dataset"].astype(str) + "__" + obs["cell_line"].astype(str) + "__" + obs["condition"].astype(str))
        names[context] = name
        context_controls.obs = obs.assign(**{CONTEXT_COLUMN_NAME: name}).reindex(columns=dataset_columns)
        context_controls.obs_names = f"{name.rsplit('__', 1)[0]}:" + context_controls.obs_names
        controls.append(context_controls)

    input_dir.mkdir(parents=True)
    # Plain strings, so the h5ad is readable under both pandas 2 and 3.
    with ad.settings.override(allow_write_nullable_strings=False):
        ad.concat(controls, merge="same", uns_merge="same").write_h5ad(input_dir / CONTROLS_FILE_NAME)
    pd.DataFrame(
        [
            (names[context], target, manifest["cells_per_pert"])
            for context, target in itertools.product(manifest["contexts"], targets)
        ],
        columns=[CONTEXT_COLUMN_NAME, TARGET_GENE_COLUMN_NAME, N_CELLS_COLUMN_NAME],
    ).to_csv(input_dir / PERTURBATIONS_FILE_NAME, index=False)
    read_perturbations(input_dir / PERTURBATIONS_FILE_NAME)
    return names


def run_inference(agent_dir: Path, split_dir: Path, output_path: Path, iteration_dir: str | None, dev: bool) -> None:
    command = [
        "agentomics-inference",
        "--agent-dir", str(agent_dir),
        "--input", str(split_dir),
        "--output", str(output_path),
    ]
    if iteration_dir:
        command += ["--iteration-dir", iteration_dir]
    if dev:
        command.append("--dev")
    print("Running", " ".join(command), flush=True)
    subprocess.run(command, check=True)


def write_prediction(generated_path: Path, names: dict[str, str], genes: list[str], output_path: Path) -> None:
    """The generated cells with only the challenge's obs columns, contexts D/E/F and sparse float32 counts."""
    generated = ad.read_h5ad(generated_path)
    if list(generated.var_names) != genes:
        raise ValueError("generated cells' genes differ from gene_names.csv")
    contexts = generated.obs[CONTEXT_COLUMN_NAME].astype(str).map({name: context for context, name in names.items()})
    counts = sparse.csr_matrix(generated.X, dtype=np.float32)
    counts.eliminate_zeros()
    prediction = ad.AnnData(
        X=counts,
        obs=pd.DataFrame(
            {
                TARGET_GENE_COLUMN_NAME: generated.obs[TARGET_GENE_COLUMN_NAME].astype(str).to_numpy(),
                CONTEXT_COLUMN_NAME: contexts.to_numpy(),
            },
            index=[f"cell_{i}" for i in range(generated.n_obs)],
        ),
        var=pd.DataFrame(index=genes),
    )
    with ad.settings.override(allow_write_nullable_strings=False):
        prediction.write_h5ad(output_path)
    print(f"Wrote {output_path}: {prediction.n_obs} cells, {counts.nnz / np.prod(counts.shape):.1%} nonzero")


def run_prep(vcc: Path, vcc_dir: Path, prediction_path: Path, package_path: Path, dry_run: bool) -> None:
    command = [
        str(vcc), "prep", str(prediction_path),
        "-g", str(vcc_dir / "gene_names.csv"),
        "--perts", str(vcc_dir / "pert_counts.csv"),
        "-o", str(package_path),
    ]
    if dry_run:
        command.append("--dry-run")
    print("Running", " ".join(command), flush=True)
    subprocess.run(command, check=True)


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("--vcc-dir", type=Path, required=True, help="Unzipped `vcc datasets download controls-final`")
    parser.add_argument("--controls-dir", type=Path, required=True, help="Folder with VCC2026_inputs__{D,E,F}.h5ad")
    parser.add_argument("--dataset-dir", type=Path, required=True, help="Agentomics dataset the model was trained on")
    parser.add_argument("--work-dir", type=Path, required=True, help="Folder for every output (new or empty)")
    model = parser.add_mutually_exclusive_group(required=True)
    model.add_argument("--agent-dir", type=Path, help="Agentomics run whose model generates the cells")
    model.add_argument("--predictions", type=Path, help="Already generated cells for <work-dir>/input, instead of running inference")
    parser.add_argument("--iteration-dir", help="Iteration of --agent-dir to use (default: the best iteration)")
    parser.add_argument("--dev", action="store_true", help="Run inference with the agentomics:dev image")
    parser.add_argument("--vcc", type=Path, default=DEFAULT_VCC, help=f"vcc executable (default: {DEFAULT_VCC})")
    parser.add_argument("--dry-run", action="store_true", help="Only validate pred.h5ad with vcc prep, write no pred.vcc")
    args = parser.parse_args()

    if not args.vcc.is_file():
        parser.error(f"vcc executable not found: {args.vcc}")
    if args.work_dir.exists() and any(args.work_dir.iterdir()):
        parser.error(f"--work-dir must be new or empty: {args.work_dir}")
    if args.predictions is None and shutil.which("agentomics-inference") is None:
        parser.error("agentomics-inference is not on PATH")

    _, genes, _ = load_official(args.vcc_dir)
    split_dir = args.work_dir / "split"
    input_dir = split_dir / INPUT_DIR_NAME
    names = build_input(args.vcc_dir, args.controls_dir, args.dataset_dir, input_dir)
    print(f"Wrote {input_dir}: contexts {names}", flush=True)

    generated_path = args.work_dir / GENERATED_FILE_NAME
    if args.predictions is None:
        run_inference(args.agent_dir.resolve(), split_dir.resolve(), generated_path.resolve(), args.iteration_dir, args.dev)
    else:
        os.symlink(args.predictions.resolve(), generated_path)
    validate_generated_cells(generated_path, input_dir)
    print("Generated cells match the input", flush=True)

    prediction_path = args.work_dir / PREDICTION_FILE_NAME
    write_prediction(generated_path, names, genes, prediction_path)
    run_prep(args.vcc, args.vcc_dir, prediction_path, args.work_dir / PACKAGE_FILE_NAME, args.dry_run)
    if not args.dry_run:
        print(f'Submit with: {args.vcc} submit {args.work_dir / PACKAGE_FILE_NAME} -m "<message>" --wait')


if __name__ == "__main__":
    sys.exit(main())
