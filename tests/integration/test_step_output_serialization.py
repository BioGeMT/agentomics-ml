import asyncio
import json

from agentomics.agents.steps.data_exploration import (
    DataExplorationOutput,
    DataExplorationStep,
)
from agentomics.agents.steps.data_representation import (
    DataRepresentationOutput,
    DataRepresentationStep,
)
from agentomics.runtime.step_outputs import load_step_output, load_step_outputs
from agentomics.utils.config import Config
from tests.helpers import create_model_calling_output_tool, create_runtime_step


def test_typed_step_output_survives_step_lifecycle_roundtrip(
    initialized_run_config,
    prepared_iteration,
):
    representation_output = DataRepresentationOutput(
        representation="One-hot encoded nucleotide sequences.",
    )
    representation_step = create_runtime_step(
        DataRepresentationStep,
        initialized_run_config,
        model=create_model_calling_output_tool(representation_output),
    )

    asyncio.run(representation_step.run())

    loaded = load_step_output(
        initialized_run_config,
        representation_step.step_id,
        initialized_run_config.current_iteration_dir,
    )
    assert loaded == representation_output


def test_step_outputs_load_in_configured_sequence_order(
    initialized_run_config,
    prepared_iteration,
):
    exploration_output = DataExplorationOutput(
        data_description="Two-row sequence dataset.",
        feature_analysis="Sequence length is fixed.",
        domain_insights="Sequence order may matter.",
        id_to_sample_info="IDs identify rows in data.csv.",
    )
    representation_output = DataRepresentationOutput(
        representation="One-hot encoded nucleotide sequences.",
    )
    representation_step = create_runtime_step(
        DataRepresentationStep,
        initialized_run_config,
        model=create_model_calling_output_tool(representation_output),
    )
    exploration_step = create_runtime_step(
        DataExplorationStep,
        initialized_run_config,
        model=create_model_calling_output_tool(exploration_output),
    )

    asyncio.run(representation_step.run())
    asyncio.run(exploration_step.run())

    outputs = load_step_outputs(initialized_run_config)
    assert outputs == [exploration_output, representation_output]


def test_exploration_output_no_longer_carries_supplementary_insights(
    initialized_run_config,
    prepared_iteration,
):
    output = DataExplorationOutput(
        data_description="Two-row sequence dataset.",
        feature_analysis="Sequence length is fixed.",
        domain_insights="Sequence order may matter.",
        id_to_sample_info="IDs identify rows in data.csv.",
    )
    step = create_runtime_step(
        DataExplorationStep,
        initialized_run_config,
        model=create_model_calling_output_tool(output),
    )

    asyncio.run(step.run())

    assert "supplementary_insights" not in DataExplorationOutput.model_json_schema()["properties"]
    saved = json.loads(
        (initialized_run_config.current_iteration_dir / step.step_id / Config.STEP_OUTPUT_FILENAME).read_text()
    )
    assert "supplementary_insights" not in saved["payload"]


def test_old_exploration_output_with_supplementary_insights_still_loads(
    initialized_run_config,
    prepared_iteration,
):
    step_dir = initialized_run_config.current_iteration_dir / DataExplorationStep.step_id
    step_dir.mkdir(parents=True, exist_ok=True)
    (step_dir / Config.STEP_OUTPUT_FILENAME).write_text(
        json.dumps(
            {
                "step_id": DataExplorationStep.step_id,
                "model_type": "DataExplorationOutput",
                "payload": {
                    "data_description": "Old description.",
                    "feature_analysis": "Old analysis.",
                    "domain_insights": "Old insights.",
                    "id_to_sample_info": "Old ids.",
                    "supplementary_insights": "Old summary of supplementary files.",
                    "files_created": [],
                },
            }
        ),
        encoding="utf-8",
    )

    loaded = load_step_output(
        initialized_run_config,
        DataExplorationStep.step_id,
        initialized_run_config.current_iteration_dir,
    )

    assert isinstance(loaded, DataExplorationOutput)
    assert loaded.data_description == "Old description."
