from pydantic import BaseModel, Field

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
    files_created: list[str] = Field(
        description="""
        A list of files that were created during this step. Leave out the '/workspace/runs/<your_id>/' part of each path.
        """
    )

def get_prediction_exploration_prompt(validation_path, inference_path):
    return f"""
        Next task: Generate predictions on the validation set ({validation_path}) and identify where those predictions succeed, fail, and prediction biases.
        You can use but not modify the inference script ({inference_path}). If you need to write code for prediction generation and/or analysis, create a separate script.
        """