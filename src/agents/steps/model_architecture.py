from pydantic import BaseModel, Field
from pydantic.json_schema import SkipJsonSchema
from pydantic_ai import Agent, RunContext

from agents.agent_utils import get_new_rundir_files

class ModelArchitecture(BaseModel):
    architecture: str = Field(
        description="""
        The machine learning model type and architecture for your task.
        """
    )
    hyperparameters: str = Field(
        description="""
        The hyperparameters you have chosen for your model.
        """
    )
    files_created: SkipJsonSchema[list[str]] = Field(
        default_factory=list,
        description="""
        List of files created during model architecture step. Populated programmatically.
        """
    )

def get_model_architecture_prompt():
    return """Your next task: choose the model architecture and hyperparameters.
    """

def create_model_architecture_agent(config, model, tools):
    model_architecture_agent = Agent(
        model=model,
        tools=tools,
        model_settings={'temperature': config.temperature},
        output_type=ModelArchitecture,
        retries=config.max_validation_retries,
        deps_type=dict,
    )

    @model_architecture_agent.output_validator
    async def validate_model_architecture(ctx: RunContext[dict], result):
        result.files_created = get_new_rundir_files(config, since_timestamp=ctx.deps['start_time'])
        return result
    
    return model_architecture_agent