import os
import yaml
import json

def load_prompts(name):
    current_dir = os.path.dirname(os.path.abspath(__file__))
    yaml_path = os.path.join(current_dir, name)
    with open(yaml_path, 'r') as file:
        prompt_templates = yaml.safe_load(file)
    return prompt_templates

def get_system_prompt(config):
    with open(config['dataset_dir'] / "metadata.json") as f:
        dataset_metadata = json.load(f)
    train_csv_path = dataset_metadata['train_split'].replace("repository", "workspace")
    dataset_knowledge_path = dataset_metadata['dataset_knowledge'].replace("repository", "workspace")
    with open(dataset_knowledge_path) as f:
        dataset_knowledge = f.read()
    
    return f"""
    Your goal is to create a robust classifier that will generalize to new unseen data. Use tools and follow instructions to reach this goal.
    You are using a linux system.
    You have access to both CPU and GPU resources. Use them efficiently to train models.
    You are provided with your own already activated environment
    Use this environment to install any packages you need (use non-verbose mode for installations, run conda installations with -y option).
    Don't delete this environment.
    Write all your python scripts in files.
    You can create files only in {config['workspace_dir'] / config['agent_id']} directory.
    Run all commands in a way that prints the least amount of tokens into the console.
    Always call tools with the right arguments, specifying each argument as separate key-value pair. 
    

    Dataset path:
    {train_csv_path}

    Dataset knowledge:
    {dataset_knowledge}
    """
    # return load_prompts(config["prompt"])["system_prompt"]

def get_user_prompt(config):
    # Custom instructions go here
    return f"""
    Create the best possible classifier that will generalize to new unseen data.
    """

def get_iteration_prompt(config, run_index, feedback):
    return f"""
    You have already completed {run_index} runs of your task.
    Here is the feedback from your past runs:
    {feedback}
    Files from your past run are still in your workspace.
    """