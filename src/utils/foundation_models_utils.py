import os
import yaml

BASE_DIR = os.environ.get('HF_HOME', '/foundation_models')
MODELS_YAML = os.path.join('/foundation_models', 'models.yaml')

def load_models_config():
    with open(MODELS_YAML, 'r') as f:
        return yaml.safe_load(f)

def build_foundation_model_catalog():
    """
    Returns a catalog of foundation models with their summaries, model names, parameters, and documentation paths.
    """
    models_config = load_models_config()
    catalog = {}

    for family_name, meta in models_config.items():
        catalog[family_name] = {
            "summary": meta.get("summary"),
            "path_to_info": meta.get("path_to_info"),
            "models": [
                {
                    "name": model_cfg.get("name"),
                    "params": model_cfg.get("params"),
                }
                for model_cfg in meta.get("models", [])
            ],        
        }
                                                                                                                    
    return catalog

def format_foundation_model_catalog(catalog):
    """
    Formats the foundation model catalog into a string for feedback agent and get_foundation_models_info tool description.
    """
    sections = []
    for family, meta in catalog.items():
        lines = [f"Family: {family}"]

        summary = meta.get("summary")
        lines.append(f"Summary: {summary}")

        models = meta.get("models")
        lines.append("Models HuggingFace id:")
        for model in models:
            name = model.get("name")
            params = model.get("params")
            label = f"{name} ({params} params)" if params else name
            lines.append(f"- {label}")
        
        path_to_info = meta.get("path_to_info")
        lines.append(f"Docs and code snippets: {path_to_info}")
        sections.append("\n".join(lines))
    
    return "\n\n".join(sections)