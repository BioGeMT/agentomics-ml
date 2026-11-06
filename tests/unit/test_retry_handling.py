import pytest
from pydantic_ai import ModelRequestNode
from pydantic_ai.messages import ModelRequest, RetryFeedbackPart, RetryInfo, ToolReturnPart

from agentomics.utils.printing_utils import pretty_print_node


@pytest.mark.parametrize(
    "retry_part",
    [
        ToolReturnPart(tool_name="final_result", content="value must be 2", outcome="retried", retry=RetryInfo(cause="model_retry")),
        RetryFeedbackPart(content="value must be 2", retry=RetryInfo(cause="model_retry")),
    ],
    ids=["output_tool", "text_output"],
)
@pytest.mark.parametrize(
    ("verbosity", "expected_lines"),
    [
        ("summary", ["agent output validation failed; retrying"]),
        ("full", ["Output Validation Failed, Retry info:", "value must be 2"]),
    ],
)
def test_rejected_output_is_printed_as_retry(monkeypatch, capsys, retry_part, verbosity, expected_lines):
    monkeypatch.setenv("AGENTOMICS_VERBOSITY", verbosity)

    pretty_print_node(ModelRequestNode(ModelRequest(parts=[retry_part])))

    printed = capsys.readouterr().out
    for line in expected_lines:
        assert line in printed
