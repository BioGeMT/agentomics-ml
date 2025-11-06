import json

def get_system_prompt(config):
    train_csv_path = config.agent_dataset_dir / "train.csv"
    validation_csv_path = config.agent_dataset_dir / "validation.csv"
    dataset_knowledge = get_dataset_knowledge(config)
    dataset_paths = f"Dataset path:\n    {train_csv_path}"
    if validation_csv_path.exists():
        dataset_paths += f"\n    Validation path:\n    {validation_csv_path}"
    
    gpu_available = config.check_gpu_availability() is not None

    available_resources = config.get_resources_summary()

    #TODO don't limit your models and training concepts because of the lack of GPU
    return f"""
    Your goal is to create a robust machine learning model that will generalize to new unseen data. Use tools and follow instructions to reach this goal.
    You are using a linux system.
    You have access to the following resources: {available_resources}. Use them efficiently to train models.
    {'If a model architecture is fit for being accelerated by GPU, ensure your code uses GPU correctly before you run training.' if gpu_available else ''}
    You are provided with your own already activated environment
    Use this environment to install any packages you need (use non-verbose mode for installations, run conda installations with -y option).
    Don't delete this environment.
    Write all your python scripts in files.
    You can create files only in {config.runs_dir / config.agent_id} directory.
    Don't create or modify any folders starting with 'iteration_'.
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

def get_user_prompt(config):
    return config.user_prompt

def get_iteration_prompt(config, run_index, feedback):
    past_iterations_range = f"iteration_0 up to iteration_{run_index-1}" if run_index > 1 else "iteration_0"
    return f"""
    Your original prompt: {config.user_prompt}
    You are at iteration {run_index}. Files from past iterations ({past_iterations_range}) are available in read-only folders: {config.runs_dir / config.agent_id}/iteration_0, iteration_1, etc.
    If you want to reuse any code or files from past iterations, copy them into your current working directory ({config.runs_dir / config.agent_id}). Files in past iteration folders won't be accessible during final inference.
    Instructions to follow for the current iteration:
    {feedback}
    {"You must not modify the train.csv and validation.csv files this iteration." if not config.can_iteration_split_data(run_index) else ""}
    """