"""Generate a Virtual Embryo Challenge Task 1 prediction with a trained Agentomics model.

Steps, each in its own folder under --work-dir:
    1. split/input/     the model's input: every observed stage (E8.5, E9.5 and, with --e105, E10.5) in
                        controls.h5ad, and the target stage with its n_cells in perturbations.csv
    2. generated.h5ad   agentomics-inference on that input (skipped with --predictions)
    3. pred.h5ad        the submission: .X log1p expression (CSR float32) over the released genes, in order
    4. score.json       the starter kit's local check, with --score-script (score_h5ad.py)
Uploading pred.h5ad is left to you (virtualembryo.ai/challenge/account). E10.5 is the leaderboard (validation)
stage, E12.5 the final test.

Usage:
    python scripts/ve_t1_submission.py --source datasets/virtual_embryo_t1 --e105 <E10.5 answers .h5ad> \\
        --target-stage E12.5 --agent-dir outputs/<run> --work-dir <new folder> [--dev]
"""
from __future__ import annotations

import argparse
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
    CONTROLS_FILE_NAME,
    N_CELLS_COLUMN_NAME,
    PERTURBATIONS_FILE_NAME,
)
from agentomics.datasets.data_contract import INPUT_DIR_NAME, validate_generated_cells
from agentomics.datasets.temporal_generation_contract import STAGE_COLUMN_NAME, read_conditions, stage_value
from agentomics.datasets.virtual_embryo import (
    DEFAULT_N_CELLS,
    MAX_SUBMISSION_CELLS,
    concat_stages,
    load_stage,
    write_cells,
)
from agentomics.utils.task_types import TaskTypes

TRAINING_FILES = {"E8.5": "E8.5_RNA.h5ad", "E9.5": "E9.5_RNA.h5ad"}
MIN_SUBMISSION_CELLS = {"E10.5": 1000, "E12.5": 638}
GENERATED_FILE_NAME = "generated.h5ad"
PREDICTION_FILE_NAME = "pred.h5ad"


def build_input(source_dir: Path, e105: Path | None, target_stage: str, n_cells: int, input_dir: Path) -> list[str]:
    """Writes the model's input folder and returns the released gene order."""
    stages = [load_stage(source_dir / TRAINING_FILES["E8.5"], "E8.5")]
    genes = list(stages[0].var_names)
    stages.append(load_stage(source_dir / TRAINING_FILES["E9.5"], "E9.5", genes))
    if e105 is not None:
        stages.append(load_stage(e105, "E10.5", genes))
    latest = max(stage.obs[STAGE_COLUMN_NAME].iloc[0] for stage in stages)
    if stage_value(target_stage) <= stage_value(latest):
        raise ValueError(f"target stage {target_stage} must be later than the latest observed stage {latest}")

    write_cells(concat_stages(stages), input_dir / CONTROLS_FILE_NAME)
    pd.DataFrame({STAGE_COLUMN_NAME: [target_stage], N_CELLS_COLUMN_NAME: [n_cells]}).to_csv(
        input_dir / PERTURBATIONS_FILE_NAME, index=False
    )
    read_conditions(input_dir / PERTURBATIONS_FILE_NAME)
    return genes


def run_inference(agent_dir: Path, split_dir: Path, output_path: Path, iteration_dir: str | None, dev: bool) -> None:
    command = ["agentomics-inference", "--agent-dir", str(agent_dir), "--input", str(split_dir), "--output", str(output_path)]
    if iteration_dir:
        command += ["--iteration-dir", iteration_dir]
    if dev:
        command.append("--dev")
    print("Running", " ".join(command), flush=True)
    subprocess.run(command, check=True)


def write_prediction(generated_path: Path, genes: list[str], output_path: Path) -> None:
    generated = ad.read_h5ad(generated_path)
    if list(generated.var_names) != genes:
        raise ValueError("generated cells' genes differ from the released gene order")
    matrix = sparse.csr_matrix(generated.X, dtype=np.float32)
    matrix.eliminate_zeros()
    prediction = ad.AnnData(
        X=matrix,
        obs=pd.DataFrame(index=[f"cell_{i}" for i in range(generated.n_obs)]),
        var=pd.DataFrame(index=pd.Index(genes)),
    )
    with ad.settings.override(allow_write_nullable_strings=False):
        prediction.write_h5ad(output_path)
    print(f"Wrote {output_path}: {prediction.n_obs} cells, {matrix.nnz / np.prod(matrix.shape):.1%} nonzero, "
          f"max {matrix.data.max() if matrix.nnz else 0:.2f}")


def run_score_script(score_script: Path, prediction_path: Path, output_path: Path) -> None:
    command = [sys.executable, str(score_script), "--task", "T1", "--input", str(prediction_path)]
    print("Running", " ".join(command), flush=True)
    result = subprocess.run(command, check=True, capture_output=True, text=True, cwd=score_script.parent)
    output_path.write_text(result.stdout, encoding="utf-8")
    print(result.stdout)


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("--source", type=Path, required=True, help="Folder with E8.5_RNA.h5ad and E9.5_RNA.h5ad")
    parser.add_argument("--e105", type=Path, help="The E10.5 validation answers, given to the model as an observed stage")
    parser.add_argument("--target-stage", required=True, choices=sorted(MIN_SUBMISSION_CELLS), help="Stage to forecast")
    parser.add_argument("--n-cells", type=int, default=DEFAULT_N_CELLS)
    parser.add_argument("--work-dir", type=Path, required=True, help="Folder for every output (new or empty)")
    model = parser.add_mutually_exclusive_group(required=True)
    model.add_argument("--agent-dir", type=Path, help="Agentomics run whose model generates the cells")
    model.add_argument("--predictions", type=Path, help="Already generated cells for <work-dir>/split/input")
    parser.add_argument("--iteration-dir", help="Iteration of --agent-dir to use (default: the best iteration)")
    parser.add_argument("--dev", action="store_true", help="Run inference with the agentomics:dev image")
    parser.add_argument("--score-script", type=Path, help="The starter kit's score_h5ad.py, for a local check")
    args = parser.parse_args()

    if not MIN_SUBMISSION_CELLS[args.target_stage] <= args.n_cells <= MAX_SUBMISSION_CELLS:
        parser.error(f"--n-cells must be within {MIN_SUBMISSION_CELLS[args.target_stage]}-{MAX_SUBMISSION_CELLS} for {args.target_stage}")
    if args.target_stage == "E10.5" and args.e105 is not None:
        parser.error("--e105 cannot be an observed stage when forecasting E10.5")
    if args.work_dir.exists() and any(args.work_dir.iterdir()):
        parser.error(f"--work-dir must be new or empty: {args.work_dir}")
    if args.predictions is None and shutil.which("agentomics-inference") is None:
        parser.error("agentomics-inference is not on PATH")

    split_dir = args.work_dir / "split"
    input_dir = split_dir / INPUT_DIR_NAME
    genes = build_input(args.source.expanduser(), args.e105, args.target_stage, args.n_cells, input_dir)
    print(f"Wrote {input_dir}: forecasting {args.target_stage}", flush=True)

    generated_path = args.work_dir / GENERATED_FILE_NAME
    if args.predictions is None:
        run_inference(args.agent_dir.resolve(), split_dir.resolve(), generated_path.resolve(), args.iteration_dir, args.dev)
    else:
        os.symlink(args.predictions.resolve(), generated_path)
    validate_generated_cells(generated_path, input_dir, TaskTypes.TEMPORAL_GENERATION)
    print("Generated cells match the input", flush=True)

    prediction_path = args.work_dir / PREDICTION_FILE_NAME
    write_prediction(generated_path, genes, prediction_path)
    if args.score_script:
        run_score_script(args.score_script.resolve(), prediction_path.resolve(), args.work_dir / "score.json")
    print(json.dumps({"submission": str(prediction_path), "target_stage": args.target_stage}))


if __name__ == "__main__":
    main()
