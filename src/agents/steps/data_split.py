from pydantic import BaseModel, Field

class DataSplit(BaseModel):
    train_path: str = Field(description="Path to generated train.csv file")
    val_path: str = Field(description="Path to generated validation.csv file")

def get_data_split_prompt(config):
    """
    Generate the data split prompt with class representation requirements.
    
    Args:
        config: Configuration object
    """

    if config.task_type == 'classification':
        extra_instructions = "Ensure that the validation split contains representative samples from ALL classes."
    elif config.task_type == 'regression':
        extra_instructions = ""
    else:
        raise ValueError(f"Unknown task type: {config.task_type}. Supported types are 'classification' and 'regression'.")

    return f"""
        Split the training dataset into training and validation sets:
        Ensure the validation split is representative of new unseen data, since it will be used for optimizing choices like architecture, hyperparameters, and training strategies.
        {extra_instructions}
        - Save 'train.csv' and 'validation.csv' in {config.runs_dir / config.agent_id}.
        Return the absolute paths to these files.

        If the split already exists and you don't want to use a different splitting strategy, you can choose to return the already existing split files immediately.
        """