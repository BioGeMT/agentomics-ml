"""Virtual Cell Challenge 2026 scoring through cell-eval2 (Arc's official scorer).

Both inputs are h5ad files of raw counts with obs `context` and `target_gene`:
- real: measured cells, including the control cells (`target_gene == "non-targeting"`)
- pred: generated cells for the same perturbations and genes, without control cells

Each context is scored independently against its own control cells and the values are
averaged over contexts. Every value is scaled so that 0 is the context-mean baseline and 1 is
an experimental replicate. The baseline and replicate (the "real bundle") depend only on the
real data, so they are cached on disk; the full score is cached in memory per file version.

Runs on GPU when cell-eval2's GPU extras (`cell-eval2[gpu,gpudge]`) and a CUDA device are
usable, and on CPU otherwise.
"""
from __future__ import annotations

import argparse
import functools
import hashlib
import subprocess
import sys
import tempfile
from pathlib import Path

import pandas as pd

CONTEXT_COLUMN = "context"
PERTURBATION_COLUMN = "target_gene"
CONTROL_LABEL = "non-targeting"
CELL_EVAL2_FLAGS = ("--preset", "vcc2026", "--pert-col", PERTURBATION_COLUMN, "--control", CONTROL_LABEL)
BUNDLE_CACHE_DIR = Path(tempfile.gettempdir()) / "agentomics_cell_eval2"

_score_cache: dict[tuple, dict[str, float]] = {}


def score_vcc2026(real_path, pred_path) -> dict[str, float]:
    """Return {cell-eval2 metric: replicate-scaled value averaged over contexts}, plus avg_score."""
    key = (_file_version(real_path), _file_version(pred_path))
    if key not in _score_cache:
        _score_cache[key] = _score(Path(real_path), Path(pred_path))
    return _score_cache[key]

def _file_version(path) -> tuple:
    path = Path(path).resolve()
    stat = path.stat()
    return (str(path), stat.st_size, stat.st_mtime_ns)

def _score(real_path: Path, pred_path: Path) -> dict[str, float]:
    import anndata as ad

    real = ad.read_h5ad(real_path, backed="r")
    pred = ad.read_h5ad(pred_path)
    _validate_pair(real, pred)

    per_context = {}
    for context in sorted(real.obs[CONTEXT_COLUMN].unique()):
        real_context = real[(real.obs[CONTEXT_COLUMN] == context).to_numpy()].to_memory()
        controls = real_context[(real_context.obs[PERTURBATION_COLUMN] == CONTROL_LABEL).to_numpy()]
        pred_context = ad.concat(
            [pred[(pred.obs[CONTEXT_COLUMN] == context).to_numpy()], controls],
            keys=["pred", "control"],
            index_unique="-",
        )
        per_context[context] = _score_context(real_context, pred_context)
        print(f"cell-eval2 vcc2026 [context {context}]: {per_context[context]}")

    scores = pd.DataFrame(per_context).mean(axis=1)
    return {metric: float(value) for metric, value in scores.items()}

def _validate_pair(real, pred) -> None:
    for name, adata in (("real", real), ("pred", pred)):
        missing = {CONTEXT_COLUMN, PERTURBATION_COLUMN} - set(adata.obs.columns)
        if missing:
            raise ValueError(f"{name} h5ad is missing obs columns: {sorted(missing)}")
    if list(pred.var_names) != list(real.var_names):
        raise ValueError("pred h5ad genes (var_names) must match the real h5ad genes, in the same order")
    if (pred.obs[PERTURBATION_COLUMN] == CONTROL_LABEL).any():
        raise ValueError(
            f"pred h5ad must not contain control cells ('{CONTROL_LABEL}'); "
            "the real control cells are used as the reference"
        )
    real_contexts = set(real.obs[CONTEXT_COLUMN].unique())
    pred_contexts = set(pred.obs[CONTEXT_COLUMN].unique())
    if real_contexts != pred_contexts:
        raise ValueError(f"pred contexts {sorted(pred_contexts)} differ from real contexts {sorted(real_contexts)}")

def _score_context(real, pred) -> dict[str, float]:
    import anndata as ad
    from scipy import sparse

    # cell-eval2 refuses sparse matrices that are not canonical (sorted, no duplicate entries).
    for adata in (real, pred):
        if sparse.issparse(adata.X):
            adata.X.sum_duplicates()

    with tempfile.TemporaryDirectory() as temporary_directory:
        work_dir = Path(temporary_directory)
        real_path = work_dir / "real.h5ad"
        pred_path = work_dir / "pred.h5ad"
        # Under pandas 2, anndata refuses to write nullable-string obs columns by default. Write
        # plain strings instead, so cell-eval2's own h5ad writes downstream don't hit the same error.
        with ad.settings.override(allow_write_nullable_strings=False):
            real.write_h5ad(real_path)
            pred.write_h5ad(pred_path)

        bundle_dir = _get_real_bundle(real, real_path)
        run_dir = work_dir / "run"
        scored_path = work_dir / "scored.csv"
        _run_cell_eval2("run", "-ap", pred_path, "-ar", real_path, "-o", run_dir, *_cell_eval2_flags())
        _run_cell_eval2(
            "score",
            "--user-agg", run_dir / "agg_results.csv",
            "--real-bundle", bundle_dir,
            "-o", scored_path,
        )
        scored = pd.read_csv(scored_path)
    return dict(zip(scored["metric"], scored["from_replicate"].astype(float)))

def _get_real_bundle(real, real_path: Path) -> Path:
    """Build (or reuse) the baseline and replicate-anchor bundle for this real data."""
    bundle_dir = BUNDLE_CACHE_DIR / _content_digest(real)
    if bundle_dir.is_dir():
        return bundle_dir

    BUNDLE_CACHE_DIR.mkdir(parents=True, exist_ok=True)
    with tempfile.TemporaryDirectory(dir=BUNDLE_CACHE_DIR) as temporary_directory:
        work_dir = Path(temporary_directory)
        baseline_pred_path = work_dir / "baseline_pred.h5ad"
        _run_cell_eval2(
            "baseline", "-ar", real_path, "-o", work_dir / "baseline",
            "--save-pred", baseline_pred_path, *_cell_eval2_flags(),
        )
        _run_cell_eval2(
            "prep-real-bundle", "--real", real_path, "--baseline", baseline_pred_path,
            "-o", work_dir / "bundle", *_cell_eval2_flags(),
        )
        # Rename only once complete, so an interrupted build is never reused.
        (work_dir / "bundle").rename(bundle_dir)
    return bundle_dir

def _content_digest(adata) -> str:
    from scipy import sparse

    matrix = sparse.csr_matrix(adata.X)
    digest = hashlib.sha256()
    digest.update(" ".join(_cell_eval2_flags()).encode())
    digest.update(str(matrix.shape).encode())
    for array in (matrix.data, matrix.indices, matrix.indptr):
        digest.update(array.tobytes())
    for values in (adata.var_names, adata.obs[CONTEXT_COLUMN], adata.obs[PERTURBATION_COLUMN]):
        digest.update("\n".join(map(str, values)).encode())
    return digest.hexdigest()

def _run_cell_eval2(*arguments) -> None:
    executable = Path(sys.executable).parent / "cell-eval2"
    command = [executable, *map(str, arguments)]
    result = subprocess.run(command, capture_output=True, text=True)
    if result.returncode != 0:
        raise RuntimeError(
            f"cell-eval2 {arguments[0]} failed with exit code {result.returncode}:\n{result.stderr[-3000:]}"
        )

@functools.cache
def _cell_eval2_flags() -> tuple[str, ...]:
    """Competition flags plus the compute engines, chosen once per process.

    With a visible CUDA device, scoring runs on the GPU engines (gpudge for DE, CUDA kernels) and
    fails if they cannot run, instead of silently switching engine. Without one (including
    --cpu-only, which hides the GPU), it runs on CPU. The engines are part of the real-bundle cache
    key, because DE numbers differ between engines and a bundle must be scored with the engine
    that built it.
    """
    from cell_eval2.de_compute import _available, _cuda_device_present
    from cell_eval2.gpu import resolve_device

    if _cuda_device_present():
        if not _available("gpudge") or resolve_device("auto") != "cuda":
            raise RuntimeError(
                "A CUDA device is visible but cell-eval2's GPU engines cannot use it "
                "(torch.cuda or cupy unavailable). The runtime env needs CUDA 12.4+ drivers; "
                "check `nvidia-smi` and `python -c 'import torch; print(torch.cuda.is_available())'`."
            )
        backend, device = "gpudge", "cuda"
    else:
        backend, device = "scanpy", "cpu"
    print(f"cell-eval2 engines: de.backend={backend}, device={device}")
    return (*CELL_EVAL2_FLAGS, "--set", f"de.backend={backend}", "--set", f"device={device}")

def main() -> None:
    from agentomics.utils.metrics import get_conditional_generation_metrics_functions

    parser = argparse.ArgumentParser(description="Score generated cells with the VCC 2026 metrics (cell-eval2).")
    parser.add_argument("--real", required=True, help="h5ad of measured cells, including control cells")
    parser.add_argument("--pred", required=True, help="h5ad of generated cells, without control cells")
    args = parser.parse_args()

    for name, metric in get_conditional_generation_metrics_functions().items():
        print(f"{name}: {metric(args.real, args.pred)}")

if __name__ == "__main__":
    main()
