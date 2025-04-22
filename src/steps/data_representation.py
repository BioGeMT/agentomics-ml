from pydantic import BaseModel, Field

class DataRepresentation(BaseModel):
    representation: str = Field(
        description="""
        The instructions for the coding implementation on how to represent the data before being passed to a Machine Learning model
        """
    )
    feature_engineering: str = Field(
        description="""
        New features to create, including interactions, aggregations, and domain-specific transformations that might improve model performance.
        """
    )
    reasoning: str = Field(
        description="""
        The reasoning behind your choice of representation
        """
    )