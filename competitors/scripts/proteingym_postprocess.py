import argparse
import hashlib
import subprocess
import sys
from pathlib import Path

PROJECT_ROOT = Path(__file__).resolve().parents[2]
SRC_DIR = PROJECT_ROOT / "src"
if str(SRC_DIR) not in sys.path:
    sys.path.insert(0, str(SRC_DIR))

from utils.biomlbench_proteingym_cv import generate_proteingym_submission_with_cv


def _get_submission_dir(artifact_root):
    submission_csv_paths = sorted(artifact_root.glob("**/submission/submission.csv"))
    assert len(submission_csv_paths) == 1, (
        "Expected exactly one submission.csv under run artifacts. "
        f"Found {len(submission_csv_paths)} at: {[str(p) for p in submission_csv_paths]}"
    )
    return submission_csv_paths[0].parent


def _ensure_conda_env(env_path, env_yaml_path):
    if env_path.exists():
        return
    env_path.parent.mkdir(parents=True, exist_ok=True)
    result = subprocess.run(
        ["conda", "env", "create", "-p", str(env_path), "-f", str(env_yaml_path)],
        capture_output=True,
        text=True,
        check=False,
    )
    if result.returncode != 0:
        raise RuntimeError(
            f"Failed to create conda env at {env_path} from {env_yaml_path}\n"
            f"STDOUT:\n{result.stdout}\n"
            f"STDERR:\n{result.stderr}"
        )


def _get_short_env_path(submission_dir, env_name):
    submission_dir = Path(submission_dir).resolve()
    submission_hash = hashlib.sha1(str(submission_dir).encode("utf-8")).hexdigest()[:10]
    return Path("/tmp/proteingym_cv_envs") / f"{env_name}_{submission_hash}"


def postprocess_proteingym_submission(artifact_root, task_id, data_dir, env_name="proteingym_cv_env"):
    artifact_root = Path(artifact_root)
    data_dir = Path(data_dir)
    assert artifact_root.is_dir(), f"Artifact root not found: {artifact_root}"
    assert task_id.startswith("proteingym-dms/"), f"Not a ProteinGym task id: {task_id}"
    assert data_dir.is_dir(), f"BioMLBench data dir not found: {data_dir}"

    submission_dir = _get_submission_dir(artifact_root)
    train_script_path = submission_dir / "train.py"
    inference_script_path = submission_dir / "inference.py"
    env_yaml_path = submission_dir / "environment.yaml"
    submission_path = submission_dir / "submission.csv"
    submission_extended_path = submission_dir / "submission_extended.csv"

    assert train_script_path.is_file(), f"Missing train.py: {train_script_path}"
    assert inference_script_path.is_file(), f"Missing inference.py: {inference_script_path}"
    assert env_yaml_path.is_file(), f"Missing environment.yaml: {env_yaml_path}"

    data_csv_path = data_dir / task_id / "prepared" / "public" / "data.csv"
    assert data_csv_path.is_file(), f"Missing ProteinGym data.csv: {data_csv_path}"

    env_path = _get_short_env_path(submission_dir, env_name)
    _ensure_conda_env(env_path, env_yaml_path)

    return generate_proteingym_submission_with_cv(
        env_path=env_path,
        train_script_path=train_script_path,
        inference_script_path=inference_script_path,
        data_csv_path=data_csv_path,
        submission_path=submission_path,
        submission_extended_path=submission_extended_path,
        source_label_col="fitness_score",
        script_label_col="fitness_score",
    )


def parse_args():
    parser = argparse.ArgumentParser(description="Post-process ProteinGym submissions with fold-aware CV retraining")
    parser.add_argument("--artifact-root", required=True, help="Path to copied run_artifacts directory")
    parser.add_argument("--task-id", required=True, help="BioMLBench task id, e.g. proteingym-dms/...")
    parser.add_argument("--data-dir", required=True, help="BioMLBench prepared data directory")
    parser.add_argument("--env-name", default="proteingym_cv_env", help="Name suffix for local conda env path")
    return parser.parse_args()


def main():
    args = parse_args()
    result = postprocess_proteingym_submission(
        artifact_root=args.artifact_root,
        task_id=args.task_id,
        data_dir=args.data_dir,
        env_name=args.env_name,
    )
    print(result)


if __name__ == "__main__":
    main()
