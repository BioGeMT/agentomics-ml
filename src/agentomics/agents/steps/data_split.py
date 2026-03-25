import os
from pathlib import Path

from pydantic import BaseModel, Field
from pydantic.json_schema import SkipJsonSchema
from pydantic_ai import Agent, RunContext, ModelRetry
import pandas as pd

from agents.agent_utils import get_new_rundir_files

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

def create_data_split_agent(config, model, tools):
    data_split_agent = Agent(
        model=model,
        tools=tools,
        model_settings={'temperature': config.temperature},
        output_type=DataSplit,
        retries=config.max_validation_retries,
        deps_type=dict,
    )

    @data_split_agent.output_validator
    async def validate_split_dataset(ctx: RunContext[dict], result: DataSplit) -> DataSplit:
        if not os.path.exists(result.train_path) or not os.path.exists(result.val_path):
            raise ModelRetry("Split dataset files do not exist.")
        
        train_path = Path(result.train_path)
        val_path = Path(result.val_path)
        if(train_path.name != 'train.csv' or val_path.name != 'validation.csv'):
            # Care to not delete original training or validation data
            original_train_csv_path = config.agent_dataset_dir / "train.csv"
            original_valid_csv_path = config.agent_dataset_dir / "validation.csv"
            if (train_path.resolve() != original_train_csv_path.resolve() and train_path.resolve() != original_valid_csv_path.resolve()):
                train_path.unlink()
            if (val_path.resolve() != original_train_csv_path.resolve() and val_path.resolve() != original_valid_csv_path.resolve()):
                val_path.unlink()

            raise ModelRetry(f"The files must be called exactly 'train.csv' and 'validation.csv'. Files ({train_path.name} and {val_path.name}) have been deleted.")
        
        target_col = 'numeric_label' #TODO generalize and take from metadata.json or config
        train_df = pd.read_csv(result.train_path)
        val_df = pd.read_csv(result.val_path)

        if target_col not in train_df.columns:
            raise ModelRetry(f"Target column {target_col} not found in train dataset {result.train_path}. Columns found: {train_df.columns.tolist()}")
        if target_col not in val_df.columns:
            raise ModelRetry(f"Target column {target_col} not found in validation dataset {result.val_path}. Columns found: {val_df.columns.tolist()}")

        # Validate that both datasets have id column
        if 'id' not in train_df.columns:
            raise ModelRetry(f"ID column 'id' is missing in train dataset {result.train_path}. Columns found: {train_df.columns.tolist()}")
        if 'id' not in val_df.columns:
            raise ModelRetry(f"ID column 'id' is missing in validation dataset {result.val_path}. Columns found: {val_df.columns.tolist()}")

        # Validate that train and validation have no overlapping IDs
        train_ids = set(train_df['id'].dropna().tolist())
        val_ids = set(val_df['id'].dropna().tolist())
        overlapping_ids = train_ids.intersection(val_ids)
        if overlapping_ids:
            raise ModelRetry(f"Train and validation datasets have overlapping IDs. IDs must be unique across train and validation splits.")

        result.files_created = get_new_rundir_files(config, since_timestamp=ctx.deps['start_time'])
        return result
    
    return data_split_agent