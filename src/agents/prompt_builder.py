from datasets.data_contract import SUPPLEMENTARY_DIR_NAME, TRAIN_SPLIT, VALIDATION_SPLIT

from runtime.read_write_utils import load_dataset_metadata
from runtime.system_resources import check_gpu_availability, get_resources_summary
from utils.config import Config
from utils.task_types import TaskTypes


def get_system_prompt(config: Config):
    train_split_path = config.dataset_dir / TRAIN_SPLIT
    validation_split_path = config.dataset_dir / VALIDATION_SPLIT
    dataset_knowledge = get_dataset_knowledge(config)
    dataset_paths = f"Training split path:\n    {train_split_path}"
    if validation_split_path.exists():
        dataset_paths += f"\n    Validation split path:\n    {validation_split_path}"
    supplementary_path = config.dataset_dir / SUPPLEMENTARY_DIR_NAME
    if supplementary_path.is_dir():
        dataset_paths += (
            f"\n    Supplementary materials that can be used at any point e.g. during training, during decision making, etc.. (read-only):\n    {supplementary_path}"
            "\n    If you need to reference any supplementary file in your scripts, copy it to your current step directory first."
        )
    
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
    Split folder contract:
    - Each split folder contains an input/ folder with the data files and labels.csv with labels keyed by id.
    - The input/ interface is fixed and must not be modified anywhere.
    - The inference script receives only the input/ folder, not the full split — never depend on labels.csv in inference code.
    - Split directories also contain a mini_train/ folder used for automatic system validation and quick development runs. Generated scripts must not hard-code mini_train paths; use the provided CLI data arguments instead.

    Dataset knowledge:
    {dataset_knowledge}
    """

def get_dataset_knowledge(config: Config):
    dataset_knowledge_path = config.dataset_dir / "dataset_description.md"
    with open(dataset_knowledge_path) as f:
        dataset_knowledge = f.read()
    if config.task_type == TaskTypes.CLASSIFICATION:
        metadata = load_dataset_metadata(config)
        dataset_knowledge += f"\n\nLabel mapping: {metadata.get('label_to_scalar', {})}"
    return dataset_knowledge

def build_iteration_base_prompt(config: Config, iteration: int) -> str:
    return f"""
    User instructions: {config.user_prompt}

    Workspace rules:
    - Your current writable directory is: {config.current_step_dir}
    - Create and modify files only inside your current writable directory.
    - Files left in your current writable directory are checkpointed after the step. Do not leave regenerable large caches, downloaded package/model caches, full-dataset tensors/embeddings, or large diagnostic dumps there.
    - If you need temporary caches while working, keep them clearly named inside your current writable directory and delete them before final_result.
    - Previous step and iteration folders are read-only.
    - If you want to reuse earlier files, copy them into your current writable directory before modifying them.
    - Don't create or modify any folders or files starting with 'iteration_'.

    You are at iteration {iteration}.
    {f"Archived iteration folders are available under {config.run_dir}/iteration_0, iteration_1, etc. Structured outputs are available under {config.run_dir}/iteration_<iteration_number>/<step_id>/output.json" if iteration > 0 else ""}
    """
