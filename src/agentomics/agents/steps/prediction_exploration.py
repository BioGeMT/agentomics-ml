import os

from pydantic import BaseModel, Field
from pydantic.json_schema import SkipJsonSchema
from pydantic_ai import Agent, RunContext, ModelRetry

from agentomics.agents.agent_utils import does_file_contain_iteration_pattern, get_invalid_iteration_folders, get_new_rundir_files
from agentomics.run_logging.evaluate_log_run import run_inference_and_log

class PredictionExploration(BaseModel):
    statistics: str = Field(
        description="""
        Statistics that provide insight into the successes, fails, and biases of the model predictions of the validation set.
        """
    )
    insights: str = Field(
        description="""
        Insights about validation set predictions that are useful for future modeling attempts.
        Don't provide concrete implementation recommendations for improvement.
        """
    )
    files_created: SkipJsonSchema[list[str]] = Field(
        default_factory=list,
        description="""
        List of files created during prediction exploration step. Populated programmatically.
        """
    )

def get_prediction_exploration_prompt(validation_path, inference_path):
    return f"""
        Your next task: Generate predictions on the validation set ({validation_path}) and identify where those predictions succeed, fail, and prediction biases.
        You can use but not modify the inference script ({inference_path}). If you need to write code for prediction generation and/or analysis, create a separate script.
        """

def create_prediction_exploration_agent(config, model, tools):
    prediction_exploration_agent = Agent(
        model=model,
        tools=tools,
        model_settings={'temperature':config.temperature},
        output_type= PredictionExploration,
        retries=config.max_validation_retries,
        deps_type=dict,
    )

    @prediction_exploration_agent.output_validator
    async def validate_prediction_exploration(ctx: RunContext[dict], result: PredictionExploration) -> PredictionExploration:
        if not os.path.exists(config.runs_dir / config.agent_id / "inference.py"):
            raise ModelRetry(f"Inference file does not exist at {config.runs_dir / config.agent_id / 'inference.py'}")
        if does_file_contain_iteration_pattern(config.runs_dir / config.agent_id / "inference.py"):
            raise ModelRetry("Inference file contains references to an iteration folder ('iteration_' detected), which will not accessible during final testing. If you want to re-use a file from a past iteration, copy it into the current working directory and use its path.")
        invalid_iter_folders = get_invalid_iteration_folders(config, ctx.deps['iteration'])
        if len(invalid_iter_folders) > 0:
            raise ModelRetry("An iteration folder or file with 'iteration' in its name was created during this iteration. For the invalid files, rename them. For the invalid folders, move all files out of them to the current working directory, update their dependencies if necessary, and delete the folder. This applies to the following files/folders: " + ", ".join(invalid_iter_folders))
        run_inference_and_log(config, iteration=-1, evaluation_stage='dry_run')
        result.files_created = get_new_rundir_files(config, since_timestamp=ctx.deps['start_time'])
        return result

    return prediction_exploration_agent