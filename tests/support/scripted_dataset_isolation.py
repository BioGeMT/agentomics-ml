"""Controller-side scripted responses for the dataset-isolation scenario."""

from pathlib import Path
from shlex import quote
from typing import Any

from pydantic_ai.messages import ModelMessage

from agentomics.utils.config import Config
from tests.dataset_helpers import TEST_SEQUENCE_PREFIX
from tests.support.scripted_workflow import build_default_step_response


def build_dataset_isolation_step_response(
    output_fields: dict[str, Any],
    messages: list[ModelMessage],
    config: Config,
) -> tuple[list[tuple[str, dict[str, Any], str]], dict[str, Any]]:
    """Build model responses that exercise dataset access during exploration."""
    if "data_description" not in output_fields:
        return build_default_step_response(output_fields, messages, config)

    step = config.current_step_dir
    script = step / "dataset_access_check.py"
    evidence = step / "dataset_access_check.json"
    source_dataset = Path("/datasets") / config.dataset
    script_contents = Path(__file__).with_name("dataset_access_check.py").read_text()
    arguments = (
        f"--dataset-source {quote(str(source_dataset))} "
        f"--evidence {quote(str(evidence))} "
        f"--test-sequence-prefix {quote(TEST_SEQUENCE_PREFIX)}"
    )
    tool_calls = [
        (
            "write_python",
            {"file_path": str(script), "code": script_contents},
            f"Code syntax OK, written to {script}",
        ),
        (
            "run_python",
            {"python_file_path": str(script), "args": arguments},
            "Dataset access check completed",
        ),
    ]
    return tool_calls, {
        "data_description": (
            "Two short synthetic nucleotide sequences with supplied validation."
        ),
        "feature_analysis": (
            "The first nucleotide distinguishes the two training classes."
        ),
        "domain_insights": (
            "This synthetic dataset demonstrates a minimal classification workflow."
        ),
        "id_to_sample_info": (
            "The id column of input/data.csv joins to the id column of labels.csv."
        ),
        "supplementary_insights": None,
    }
