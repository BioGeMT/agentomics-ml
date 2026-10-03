import json

import pytest

from agentomics.runtime.step_outputs import load_step_output, save_step_output
from agentomics.utils.config import Config


def test_save_step_output_creates_missing_step_directory(config_factory):
    config = config_factory()

    save_step_output(config, "data_exploration", {"summary": "complete"})

    assert config.current_step_dir.is_dir()
    assert (config.current_step_dir / Config.STEP_OUTPUT_FILENAME).is_file()


def test_save_step_output_rejects_overwriting_current_output(config_factory):
    config = config_factory()
    save_step_output(config, "data_exploration", {"summary": "first"})

    with pytest.raises(FileExistsError):
        save_step_output(config, "data_exploration", {"summary": "second"})


def test_step_output_insights_survive_save_and_load(config_factory):
    from agentomics.agents.steps.data_exploration import DataExplorationOutput

    config = config_factory()
    output = DataExplorationOutput(
        data_description="d",
        feature_analysis="f",
        domain_insights="o",
        id_to_sample_info="rows",
        insights=["ids in labels.csv need zero-padding"],
    )

    save_step_output(config, "data_exploration", output)
    config.current_step_dir.rename(config.current_iteration_dir / "data_exploration")
    loaded = load_step_output(config, "data_exploration", config.current_iteration_dir)

    assert loaded.insights == ["ids in labels.csv need zero-padding"]


def test_load_unknown_step_output_returns_plain_payload(config_factory):
    config = config_factory()
    step_id = "custom_step"
    step_dir = config.current_iteration_dir / step_id
    step_dir.mkdir(parents=True)
    (step_dir / Config.STEP_OUTPUT_FILENAME).write_text(
        json.dumps(
            {
                "step_id": step_id,
                "model_type": "dict",
                "payload": {"custom": "data"},
            }
        ),
        encoding="utf-8",
    )

    output = load_step_output(
        config,
        step_id,
        config.current_iteration_dir,
    )

    assert output == {"custom": "data"}
