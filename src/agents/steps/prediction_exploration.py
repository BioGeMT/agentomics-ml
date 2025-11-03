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

def get_prediction_exploration_prompt(validation_path, inference_path):
    return f"""
        Next task: Use the inference script ({inference_path}) to generate predictions of the validation set ({validation_path}). 
        Identify where those predictions succeed, fail, and prediction biases.
        """