from pydantic import BaseModel, Field

class DataRepresentation(BaseModel):
    representation: str = Field(
        description="""
        How will the data be represented, including any transformations, encodings, normalizations, features, and label transformations.
        """
    )
    files_created: list[str] = Field(
        description="""
        A list of files that were created during this step.
        """
    )

def get_data_representation_prompt():
    return "Your next task: define the data representation."
