from pathlib import Path
from shlex import quote
from typing import Any

from pydantic_ai import models
from pydantic_ai.messages import (
    ModelMessage,
    ModelResponse,
    RetryPromptPart,
    ToolCallPart,
    ToolReturnPart,
    UserPromptPart,
)
from pydantic_ai.models.function import AgentInfo, FunctionModel

from agentomics.utils.config import Config
from agentomics.utils.providers.provider import Provider


class ScriptedProvider(Provider):
    def __init__(self):
        # conftest.py blocks real model requests in unit/integration tests.
        # End-to-end tests launch a separate Docker worker, so guard it here too.
        models.ALLOW_MODEL_REQUESTS = False
        super().__init__("scripted", base_url="")

    def create_model(self, model_name: str, config: Config) -> FunctionModel:
        build_step_response = build_default_step_response
        if model_name == "scripted-dataset-isolation":
            from tests.support.scripted_dataset_isolation import (
                build_dataset_isolation_step_response,
            )

            build_step_response = build_dataset_isolation_step_response
        elif model_name == "scripted-workflow-failure":
            from tests.support.scripted_workflow_failure import (
                build_workflow_failure_step_response,
            )

            build_step_response = build_workflow_failure_step_response

        # FunctionModel invokes this callback for each model request. The selected
        # builder supplies that workflow step's tool calls and structured output.
        def respond(messages: list[ModelMessage], info: AgentInfo) -> ModelResponse:
            assert models.ALLOW_MODEL_REQUESTS is False
            output_tool = info.output_tools[0]
            output_fields = output_tool.parameters_json_schema["properties"]
            tool_calls, step_output = build_step_response(
                output_fields, messages, config
            )
            tool_results = _current_step_tool_results(messages)
            assert len(tool_results) <= len(tool_calls), "Unexpected scripted tool interaction"
            for result, (tool_name, _, expected_output) in zip(tool_results, tool_calls):
                content = result.model_response_str()
                assert expected_output in content.splitlines(), (
                    f"Scripted {tool_name} did not report {expected_output!r}:\n{content}"
                )
            if len(tool_results) < len(tool_calls):
                tool_name, arguments, _ = tool_calls[len(tool_results)]
                return ModelResponse(parts=[ToolCallPart(tool_name, arguments)])
            return ModelResponse(parts=[ToolCallPart(output_tool.name, step_output)])

        return FunctionModel(respond, model_name=model_name)


def _current_step_tool_results(messages: list[ModelMessage]) -> list[ToolReturnPart]:
    """Read results since the latest user prompt, excluding previous steps' history."""
    parts = [part for message in messages for part in message.parts]
    current_start = max(
        index for index, part in enumerate(parts) if isinstance(part, UserPromptPart)
    )
    tool_results = []
    for part in parts[current_start + 1:]:
        if isinstance(part, RetryPromptPart):
            raise AssertionError(f"Scripted response rejected: {part.content}")
        if isinstance(part, ToolReturnPart):
            tool_results.append(part)
    return tool_results


def _previous_step_output(messages: list[ModelMessage], field: str) -> dict[str, Any]:
    """Read the structured history visible to the model, not runtime files."""
    for message in reversed(messages):
        for part in message.parts:
            if isinstance(part, ToolCallPart):
                arguments = part.args_as_dict()
                if field in arguments:
                    return arguments
    raise AssertionError(f"Missing prior step output containing {field}")


def build_default_step_response(
    output_fields: dict[str, Any], messages: list[ModelMessage], config: Config,
) -> tuple[list[tuple[str, dict[str, Any], str]], dict[str, Any]]:
    """Return tool calls (name, arguments, expected output line) and the step output."""
    step = config.current_step_dir
    dataset = config.dataset_dir
    # Distinct fields in the requested output schema identify each workflow step.
    if "data_exploration_instructions" in output_fields:
        return [], {
            "data_exploration_instructions": "Inspect the supplied sequence table.",
            "data_split_instructions": "Keep supplied validation and use all tiny training data as mini_train.",
            "data_representation_instructions": "Use the first nucleotide as a categorical feature.",
            "model_architecture_instructions": "Learn the majority class for each first nucleotide.",
            "model_training_instructions": "Fit from training inputs and labels and save model.json.",
            "model_inference_instructions": "Load model.json and predict from input sequences only.",
            "prediction_exploration_instructions": "Generate validation predictions using the trained model.",
            "other_instructions": None,
        }
    if "data_description" in output_fields:
        command = (
            f"head {quote(str(dataset / 'train/input/data.csv'))} && "
            "echo 'Dataset inspected'"
        )
        return [("bash", {"command": command}, "Dataset inspected")], {
            "data_description": "Two short synthetic nucleotide sequences with supplied validation.",
            "feature_analysis": "The first nucleotide distinguishes the two training classes.",
            "domain_insights": "This synthetic dataset demonstrates a minimal classification workflow.",
            "id_to_sample_info": "The id column of input/data.csv joins to the id column of labels.csv.",
            "supplementary_insights": None,
        }
    if "mini_train_path" in output_fields:
        mini = step / "mini_train"
        command = (
            f"mkdir -p {quote(str(mini / 'input'))} && "
            f"ln -s {quote(str(dataset / 'train/input/data.csv'))} {quote(str(mini / 'input/data.csv'))} && "
            f"cp {quote(str(dataset / 'train/labels.csv'))} {quote(str(mini / 'labels.csv'))} && "
            "echo 'Mini-training split created'"
        )
        return [("bash", {"command": command}, "Mini-training split created")], {
            "train_path": str(dataset / "train"), "val_path": str(dataset / "validation"),
            "mini_train_path": str(mini),
            "splitting_strategy": "Keep supplied validation; all tiny training samples form mini_train.",
        }
    if "representation" in output_fields:
        return [], {"representation": "First nucleotide, represented as a categorical feature."}
    if "architecture" in output_fields:
        return [], {
            "architecture": "First-nucleotide majority-class classifier.",
            "hyperparameters": "No tunable hyperparameters.",
        }
    if "training_summary" in output_fields:
        split = _previous_step_output(messages, "mini_train_path")
        script = step / "train.py"
        artifacts = step / "training_artifacts"
        script_contents = Path(__file__).with_name("scripted_classifier_train.py").read_text()
        arguments = (
            f"--train-data {quote(split['train_path'])} "
            f"--validation-data {quote(split['val_path'])} "
            f"--artifacts-dir {quote(str(artifacts))}"
        )
        tool_calls = [
            (
                "write_python", {"file_path": str(script), "code": script_contents},
                f"Code syntax OK, written to {script}",
            ),
            (
                "run_python", {"python_file_path": str(script), "args": arguments},
                "Training completed",
            ),
        ]
        return tool_calls, {
            "path_to_train_file": str(script),
            "path_to_model_file": str(artifacts / "model.json"),
            "path_to_artifacts_dir": str(artifacts),
            "training_summary": "Learned the majority class per first nucleotide from training labels.",
            "unresolved_issues": None,
        }
    if "inference_summary" in output_fields:
        script = step / "inference.py"
        script_contents = Path(__file__).with_name("scripted_classifier_inference.py").read_text()
        tool_calls = [
            (
                "write_python", {"file_path": str(script), "code": script_contents},
                f"Code syntax OK, written to {script}",
            ),
        ]
        return tool_calls, {
            "path_to_inference_file": str(script),
            "inference_summary": "Load trained first-nucleotide classes and preserve input IDs.",
            "unresolved_issues": None,
        }
    if "statistics" in output_fields:
        split = _previous_step_output(messages, "mini_train_path")
        training = _previous_step_output(messages, "training_summary")
        inference = _previous_step_output(messages, "inference_summary")
        arguments = (
            f"--input {quote(str(Path(split['val_path']) / 'input'))} "
            f"--output {quote(str(step / 'predictions.csv'))} "
            f"--artifacts-dir {quote(training['path_to_artifacts_dir'])}"
        )
        tool_calls = [
            (
                "run_python", {
                    "python_file_path": inference["path_to_inference_file"],
                    "args": arguments,
                },
                "Predictions generated using trained artifacts",
            ),
        ]
        return tool_calls, {
            "statistics": "Generated predictions for the supplied validation inputs.",
            "insights": "The tiny synthetic dataset is not evidence of real-world generalization.",
        }
    raise AssertionError(f"No scripted response for output fields: {sorted(output_fields)}")

