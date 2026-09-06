"""Controller-side scripted responses for the failed-workflow scenario."""

from pathlib import Path
from typing import Any

from pydantic_ai.messages import ModelMessage

from agentomics.utils.config import Config
from tests.support.scripted_workflow import build_default_step_response


def build_workflow_failure_step_response(
    output_fields: dict[str, Any],
    messages: list[ModelMessage],
    config: Config,
) -> tuple[list[tuple[str, dict[str, Any], str]], dict[str, Any]]:
    """Build an inference script that fails only during validation evaluation."""
    if "inference_summary" not in output_fields:
        return build_default_step_response(output_fields, messages, config)

    step = config.current_step_dir
    inference_script = step / "inference.py"
    classifier_script = step / "classifier_inference.py"
    wrapper_contents = Path(__file__).with_name(
        "scripted_validation_failure_inference.py"
    ).read_text()
    classifier_contents = Path(__file__).with_name(
        "scripted_classifier_inference.py"
    ).read_text()
    tool_calls = [
        (
            "write_python",
            {"file_path": str(classifier_script), "code": classifier_contents},
            f"Code syntax OK, written to {classifier_script}",
        ),
        (
            "write_python",
            {"file_path": str(inference_script), "code": wrapper_contents},
            f"Code syntax OK, written to {inference_script}",
        ),
    ]
    return tool_calls, {
        "path_to_inference_file": str(inference_script),
        "inference_summary": (
            "Delegate normal predictions and fail during validation evaluation."
        ),
        "unresolved_issues": None,
    }
