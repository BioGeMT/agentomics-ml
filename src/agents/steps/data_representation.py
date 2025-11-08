from pydantic import BaseModel, Field

class DataRepresentation(BaseModel):
    representation: str = Field(
        description="""
        How will the data be represented, including any transformations, encodings, normalizations, features, and label transformations.
        """
    )
    files_created: list[str]|None = Field(
        description="""
        This field should be passed as an empty list, as this will be overwritten and populated programatically.
        """
    )

def get_data_representation_prompt():
    return "Your next task: define the data representation."
