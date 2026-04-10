import json
from pathlib import Path
from agents.steps.final_outcome import get_final_outcome_prompt
from agents.steps.data_split import get_data_split_prompt
from agents.steps.model_training import get_model_training_prompt


def load_icl_knowledge(config) -> str:
    """Load all cleaned knowledge markdown files and concatenate them."""
    knowledge_dir = config.agent_dataset_dir / "knowledge"
    if not knowledge_dir.is_dir():
        print(f"WARNING: ICL knowledge mode enabled but no knowledge directory found at {knowledge_dir}")
        return ""

    md_files = sorted(knowledge_dir.glob("*.md"))
    if not md_files:
        print(f"WARNING: ICL knowledge mode enabled but no .md files found in {knowledge_dir}")
        return ""

    parts = []
    for md_path in md_files:
        content = md_path.read_text(encoding="utf-8").strip()
        parts.append(f"### {md_path.stem}\n\n{content}")

    print(f"ICL knowledge: loaded {len(parts)} document(s) from {knowledge_dir}")
    return "\n\n---\n\n".join(parts)


def get_system_prompt(config):
    train_csv_path = config.agent_dataset_dir / "train.csv"
    validation_csv_path = config.agent_dataset_dir / "validation.csv"
    dataset_knowledge_path = config.agent_dataset_dir / "dataset_description.md"

    with open(dataset_knowledge_path) as f:
        dataset_knowledge = f.read()
    if config.task_type == "classification":
        metadata = json.loads((config.prepared_dataset_dir / "metadata.json").read_text())
        dataset_knowledge += f"\n\nLabel mapping: {metadata.get('label_to_scalar', {})}"
    dataset_paths = f"Dataset path:\n    {train_csv_path}"
    if validation_csv_path.exists():
        dataset_paths += f"\n    Validation path:\n    {validation_csv_path}"
    
    return f"""
    Your goal is to create a robust machine learning model that will generalize to new unseen data. Use tools and follow instructions to reach this goal.
    You are using a linux system.
    You have access to both CPU and GPU resources. Use them efficiently to train models.
    You are provided with your own already activated environment
    Use this environment to install any packages you need (use non-verbose mode for installations, run conda installations with -y option).
    Don't delete this environment.
    Write all your python scripts in files.
    You can create files only in {config.runs_dir / config.agent_id} directory.
    Run all commands in a way that prints the least amount of tokens into the console.
    Always call tools with the right arguments, specifying each argument as separate key-value pair. 
    

    Dataset paths:
    {dataset_paths}

    Dataset knowledge:
    {dataset_knowledge}
    """
    # return load_prompts(config["prompt"])["system_prompt"]

def get_user_prompt(config):
    user_prompt = config.user_prompt

    # Inject ICL knowledge into user prompt if enabled
    if config.knowledge_mode == "icl":
        icl_knowledge = load_icl_knowledge(config)
        if icl_knowledge:
            user_prompt += f"\n\nDomain knowledge:\n{icl_knowledge}"

    #add to initial user prompt for skipped steps that are validated
    if 'data_split' in config.steps_to_skip:
        user_prompt += "\n\n" + get_data_split_prompt(config)

    if 'model_training' in config.steps_to_skip:
        user_prompt += "\n\n" + get_model_training_prompt(config)

    if 'final_outcome' in config.steps_to_skip:
        user_prompt += "\n\n" + get_final_outcome_prompt(config)

    return user_prompt

def get_iteration_prompt(config, run_index, feedback):
    return f"""
    You have already completed {run_index} runs of your task.
    Here is the feedback from your past runs:
    {feedback}
    Files from your past run are still in your workspace.
    """