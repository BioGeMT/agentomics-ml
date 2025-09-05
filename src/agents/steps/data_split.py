from pydantic import BaseModel, Field

class DataSplit(BaseModel):
    train_path: str = Field(description="Path to generated train.csv file")
    val_path: str = Field(description="Path to generated validation.csv file")

def get_data_split_prompt(config):
    return f"""
        Split the training dataset into training and validation sets:
        Ensure the validation split is representative of new unseen data, since it will be used for optimizing choices like architecture, hyperparameters, and training strategies.
        - Save 'train.csv' and 'validation.csv' in {config.workspace_dir / config.agent_id}.
        Return the absolute paths to these files.
        """