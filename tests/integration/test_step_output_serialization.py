import asyncio

from agentomics.agents.steps.data_exploration import (
    DataExplorationOutput,
    DataExplorationStep,
)
from agentomics.agents.steps.data_representation import (
    DataRepresentationOutput,
    DataRepresentationStep,
)
from agentomics.runtime.step_outputs import load_step_output, load_step_outputs
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
