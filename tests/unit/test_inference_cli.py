from pathlib import Path
import re

import pytest

from agentomics.cli.inference import build_parser, run_inference_in_docker


@pytest.mark.parametrize(
    ("path_kind", "expected_error"),
    [("missing", FileNotFoundError), ("file", NotADirectoryError)],
)
def test_inference_rejects_invalid_artifact_directories_before_execution(
    tmp_path: Path, path_kind: str, expected_error: type[Exception],
):
    artifacts = tmp_path / "artifacts"
    if path_kind == "file":
        artifacts.write_text("not a directory", encoding="utf-8")
    arguments = build_parser().parse_args([
        "--agent-dir", str(tmp_path), "--input", str(tmp_path),
        "--output", str(tmp_path / "predictions.csv"),
        "--artifacts-dir", str(artifacts),
    ])

    with pytest.raises(expected_error, match=re.escape(str(artifacts))):
        run_inference_in_docker(arguments)


def test_inference_requires_one_iteration_when_artifacts_are_supplied(tmp_path: Path):
    artifacts = tmp_path / "artifacts"
    artifacts.mkdir()
    arguments = build_parser().parse_args([
        "--agent-dir", str(tmp_path), "--input", str(tmp_path),
        "--output", str(tmp_path / "predictions.csv"),
        "--artifacts-dir", str(artifacts), "--all-iterations",
    ])

    with pytest.raises(ValueError, match="--artifacts-dir cannot be used with --all-iterations"):
        run_inference_in_docker(arguments)
