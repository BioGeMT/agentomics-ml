import json

import pytest

pytestmark = pytest.mark.parametrize(
    "_completed_run",
    ["scripted-dataset-isolation"],
    indirect=True,
)


def _load_probe_evidence(workspace):
    evidence_path = (
        workspace
        / "run"
        / "iteration_0"
        / "data_exploration"
        / "dataset_access_check.json"
    )
    return json.loads(evidence_path.read_text(encoding="utf-8"))


def test_agent_authored_code_can_read_public_source_dataset_files(completed_run):
    evidence = _load_probe_evidence(completed_run)

    assert evidence["public_files"] == {
        "dataset_description.md": (
            "Synthetic promoter-sequence classification dataset.\n"
        ),
        "metadata.json": '{"task_type": "classification"}',
        "supplementary/paper.txt": "Supporting material.\n",
        "train/input/data.csv": (
            "id,sequence\ntrain-0,ACGTACGT\ntrain-1,TGCATGCA\n"
        ),
        "train/labels.csv": "id,label\ntrain-0,positive\ntrain-1,negative\n",
        "validation/input/data.csv": (
            "id,sequence\nvalidation-0,ACGTACGT\nvalidation-1,TGCATGCA\n"
        ),
        "validation/labels.csv": (
            "id,label\nvalidation-0,positive\nvalidation-1,negative\n"
        ),
    }


def test_agent_authored_code_cannot_change_public_source_dataset_files(completed_run):
    evidence = _load_probe_evidence(completed_run)

    assert evidence["public_source_write_succeeded"] is False
    assert evidence["write_target_content_after_attempt"] == "Supporting material.\n"


def test_held_out_data_is_unavailable_in_agent_execution_environment(completed_run):
    evidence = _load_probe_evidence(completed_run)

    assert evidence["test_paths"] == []
    assert evidence["test_sequence_matches"] == []
