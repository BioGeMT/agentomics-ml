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
        """
    )

def get_prediction_exploration_prompt(validation_path, inference_path):
    #TODO report needs updated with this output type
    return f"""
        Next task: Use the inference script ({inference_path}) to generate predictions of the validation set ({validation_path}). 
        Identify where those predictions succeed, fail, and prediction biases.
        Gather insights that are important to adjust future modeling attempts.
        """