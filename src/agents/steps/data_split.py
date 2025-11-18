from pydantic import BaseModel, Field
from pydantic.json_schema import SkipJsonSchema

class DataSplit(BaseModel):
    train_path: str = Field(description="Path to generated train.csv file")
    val_path: str = Field(description="Path to generated validation.csv file")
    splitting_strategy: str = Field(description="Detailed description of the splitting strategy used")
    files_created: SkipJsonSchema[list[str]] = Field(
        default_factory=list,
        description="""
        List of files created during data split step. Populated programmatically.
        """
    )

def get_data_split_prompt(config, iteration, last_split_strategy="Split does not exist"):
    if config.task_type == 'classification':
        extra_instructions = "Ensure that the validation split contains representative samples from ALL classes."
    elif config.task_type == 'regression':
        extra_instructions = ""
    else:
        raise ValueError(f"Unknown task type: {config.task_type}. Supported types are 'classification' and 'regression'.")

    if(iteration != 0 and last_split_strategy is not None):
        extra_info = f"""
        Note: Train ({config.agent_dataset_dir / "train.csv"}) and validation ({config.agent_dataset_dir / "validation.csv"}) split files from past iteration already exist. 
        If you don't have a reason to change the splitting strategy, return the already existing split files paths immediately, for created files return an empty list, and return this text as the splitting strategy:\n{last_split_strategy}\n.
        """
    else:
        extra_info = ""
    
    train_csv_path = config.agent_dataset_dir / "train.csv"
    return f"""
        Your next task: Split the training dataset ({train_csv_path}) into training and validation sets:
        Ensure the validation split is representative of new unseen data, since it will be used for optimizing choices like architecture, hyperparameters, and training strategies.
        {extra_instructions}
        - Save 'train.csv' and 'validation.csv' in {config.runs_dir / config.agent_id}.
        Return the absolute paths to these files.

        {extra_info}
        """