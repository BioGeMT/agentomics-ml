from unittest import mock

from agentomics.agents.steps.base import AgenticStepOutput
from agentomics.agents.steps.iteration_plan import IterationPlanStep


def build_insights_info(iteration_to_outputs):
    step = IterationPlanStep.__new__(IterationPlanStep)
    step.config = mock.Mock()
    step.config.step_sequence = ["data_exploration"]
    step.config.iteration_dir = lambda iteration: iteration
    with mock.patch(
        "agentomics.agents.steps.iteration_plan.get_archived_iterations",
        return_value=sorted(iteration_to_outputs),
    ), mock.patch(
        "agentomics.agents.steps.iteration_plan.load_step_outputs",
        side_effect=lambda config, iteration_dir, step_sequence: iteration_to_outputs[iteration_dir],
    ):
        return step._build_insights_info()


def test_insights_are_deduplicated_and_tagged_with_iteration():
    block = build_insights_info(
        {
            1: [AgenticStepOutput(insights=["Files are gzipped", "IDs need padding"])],
            2: [AgenticStepOutput(insights=["files are  GZIPPED", "labels are strings"])],
        }
    )

    assert block.lower().count("gzipped") == 1
    assert "[from iteration 1] Files are gzipped" in block
    assert "[from iteration 1] IDs need padding" in block
    assert "[from iteration 2] labels are strings" in block


def test_raw_dict_outputs_are_supported():
    block = build_insights_info({1: [{"insights": ["raw dict insight"]}]})

    assert "[from iteration 1] raw dict insight" in block


def test_outputs_without_insights_produce_empty_block():
    assert build_insights_info({1: [AgenticStepOutput()], 2: [{"other": "field"}]}) == ""


def test_insights_are_capped_to_most_recent():
    many = {
        iteration: [AgenticStepOutput(insights=[f"insight {iteration}"])]
        for iteration in range(1, IterationPlanStep.MAX_ACCUMULATED_INSIGHTS + 11)
    }

    block = build_insights_info(many)

    assert block.count("- [from iteration") == IterationPlanStep.MAX_ACCUMULATED_INSIGHTS
    assert "insight 1\n" not in block
    assert f"insight {IterationPlanStep.MAX_ACCUMULATED_INSIGHTS + 10}" in block
