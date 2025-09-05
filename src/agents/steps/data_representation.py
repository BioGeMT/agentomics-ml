from pydantic import BaseModel, Field

class DataRepresentation(BaseModel):
    representation: str = Field(
        description="""
        How will the data be represented, including any transformations, encodings, normalizations, features, and label transformations.
        """
    )
    reasoning: str = Field(
        description="""
        The reasoning behind your data representation choices.
        """
    )

def get_data_representation_prompt():
    return "Your next task: define the data representation."
