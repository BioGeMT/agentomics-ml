from pydantic import BaseModel, Field

class DataSplit(BaseModel):
    train_path: str = Field(description="Path to generated train.csv file")
    val_path: str = Field(description="Path to generated validation.csv file")

def get_data_split_prompt(config, target_classes=None, task_type="classification"):
    """
    Generate the data split prompt with class representation requirements.
    
    Args:
        config: Configuration object
        target_classes: List of target class values (optional, for classification tasks)
        task_type: Type of task ("classification" or "regression")
    """
    class_representation_instruction = ""
    if task_type == "classification" and target_classes:
        class_list = ", ".join(target_classes)
        class_representation_instruction = f"""
        IMPORTANT: For classification tasks, ensure that the validation split contains representative samples from ALL target classes: {class_list}.
        Use stratified sampling to maintain class distribution in both training and validation sets.
        This is crucial for proper model evaluation and hyperparameter optimization."""
    
    return f"""
        Split the training dataset into training and validation sets:
        Ensure the validation split is representative of new unseen data, since it will be used for optimizing choices like architecture, hyperparameters, and training strategies.
        {class_representation_instruction}
        - Save 'train.csv' and 'validation.csv' in {config.runs_dir / config.agent_id}.
        Return the absolute paths to these files.
        """