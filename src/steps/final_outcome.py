from pydantic import BaseModel, Field

class FinalOutcome(BaseModel):
    path_to_inference_file: str = Field(
        description="Absolute path to the generated inference.py"
    )
