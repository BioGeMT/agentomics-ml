import os
import yaml

def load_prompts(name):
    current_dir = os.path.dirname(os.path.abspath(__file__))
    yaml_path = os.path.join(current_dir, name)
    with open(yaml_path, 'r') as file:
        prompt_templates = yaml.safe_load(file)
    return prompt_templates