import json

from runtime.system_resources import check_gpu_availability, get_resources_summary
from utils.config import Config

def get_system_prompt(config):
    train_csv_path = config.agent_dataset_dir / "train.csv"
    validation_csv_path = config.agent_dataset_dir / "validation.csv"
    dataset_knowledge = get_dataset_knowledge(config)
    dataset_paths = f"Dataset path:\n    {train_csv_path}"
    if validation_csv_path.exists():
        dataset_paths += f"\n    Validation path:\n    {validation_csv_path}"
    
    gpu_available = check_gpu_availability() is not None

    available_resources = get_resources_summary()
    ordered_steps = "\n".join(
        f"    - {str(step_id).replace('_', ' ').title()}"
        for step_id in config.step_sequence
    )

    #TODO don't limit your models and training concepts because of the lack of GPU
    return f"""
    Your goal is to create a robust machine learning model that will generalize to new unseen data. Use tools and follow instructions to reach this goal.
    You're part of an agentic, multi-step architecture where each step builds upon the previous one:
{ordered_steps}

    This is an iterative process. Each iteration takes all of these steps. You will have multiple iterations to refine your approach based on validation performance. 
    You are using a linux system.
    You have access to the following resources: {available_resources}. Use them efficiently to train models.
    {'If a model architecture is fit for being accelerated by GPU, ensure your code uses GPU correctly before you run training.' if gpu_available else ''}
    You are provided with your own already activated environment
    Use this environment to install any packages you need (use non-verbose mode for installations, run conda installations with -y option).
    Don't delete this environment.
    Your conda environment lives under {config.shared_dir / ".conda"}.
    Write all your python scripts in files.
    Run all commands in a way that prints the least amount of tokens into the console.
    Always call tools with the right arguments, specifying each argument as separate key-value pair. 
    

    Dataset paths:
    {dataset_paths}

    Dataset knowledge:
    {dataset_knowledge}
    """
    # return load_prompts(config["prompt"])["system_prompt"]

def get_dataset_knowledge(config):
    dataset_knowledge_path = config.agent_dataset_dir / "dataset_description.md"
    with open(dataset_knowledge_path) as f:
        dataset_knowledge = f.read()
    if config.task_type == "classification":
        metadata = json.loads((config.prepared_dataset_dir / "metadata.json").read_text())
        dataset_knowledge += f"\n\nLabel mapping: {metadata.get('label_to_scalar', {})}"
    return dataset_knowledge

def build_iteration_base_prompt(config: Config, iteration: int) -> str:
    return f"""
    User instructions: {config.user_prompt}

    Workspace rules:
    - Your current writable directory is: {config.current_step_dir}
    - Create and modify files only inside your current writable directory.
    - Previous step and iteration folders are read-only.
    - If you want to reuse earlier files, copy them into your current writable directory before modifying them.
    - Don't create or modify any folders or files starting with 'iteration_'.

    You are at iteration {iteration}.
    {f"Archived iteration folders are available under {config.run_dir}/iteration_0, iteration_1, etc. Structured outputs are available under {config.run_dir}/iteration_<iteration_number>/<step_id>/output.json" if iteration > 0 else ""}
    """
