from pydantic import BaseModel, Field
from pydantic.json_schema import SkipJsonSchema

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
