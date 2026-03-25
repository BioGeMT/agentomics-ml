from pydantic import BaseModel, Field
from pydantic.json_schema import SkipJsonSchema
from pydantic_ai import Agent, RunContext

from agentomics.agents.agent_utils import get_new_rundir_files

class DataRepresentation(BaseModel):
    representation: str = Field(
        description="""
        How will the data be represented, including any transformations, encodings, normalizations, features, and label transformations.
        """
    )
    files_created: SkipJsonSchema[list[str]] = Field(
        default_factory=list,
        description="""
        List of files created during data representation step. Populated programmatically.
        """
    )

def get_data_representation_prompt():
    return "Your next task: define the data representation."

def create_data_representation_agent(config, model, tools):
    data_representation_agent = Agent(
        model=model,
        tools=tools,
        model_settings={'temperature': config.temperature},
        output_type=DataRepresentation,
        retries=config.max_validation_retries,
        deps_type=dict,
    )

    @data_representation_agent.output_validator
    async def validate_data_representation(ctx: RunContext[dict], result):
        result.files_created = get_new_rundir_files(config, since_timestamp=ctx.deps['start_time'])
        return result

    return data_representation_agent