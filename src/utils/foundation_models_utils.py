import os
import yaml
from transformers import AutoTokenizer, AutoModel, AutoModelForMaskedLM, AutoModelForSequenceClassification
import multimolecule

BASE_DIR = os.environ.get('HF_HOME', '/foundation_models')
MODELS_YAML = os.path.join('/foundation_models', 'models.yaml')

FAMILY_MODEL_CLASSES = {
    'ESM-2': AutoModel,
    'HyenaDNA': AutoModelForSequenceClassification,
    'NucleotideTransformer': AutoModelForMaskedLM,
    'rinalmo': AutoModel,
}

def load_models_config():
    with open(MODELS_YAML, 'r') as f:
        return yaml.safe_load(f)

def download_model(model_name, model_class):
    try:
        AutoTokenizer.from_pretrained(
            model_name,
            trust_remote_code=True
        )
        model_class.from_pretrained(
            model_name,
            trust_remote_code=True
        )
    except Exception as e:
        print(f"Error: {str(e)}")

def get_foundation_models_info():
    config = load_models_config()
    model_infos = []
    for model_type, family_data in config.items():
        summary = family_data.get('summary', '')
        path_to_info = family_data.get('path_to_info', '')
        models = family_data.get('models', [])

        family_info = f"**{model_type}**"
        if summary:
            family_info += f": {summary}"

        model_list = []
        for model_data in models:
            model_name = model_data.get('name')
            params = model_data.get('params', '')
            if params:
                model_list.append(f"  - {model_name} ({params} parameters)")
            else:
                model_list.append(f"  - {model_name}")

        info_parts = [family_info]
        if model_list:
            info_parts.append("\n".join(model_list))
        if path_to_info:
            info_parts.append(f"  Documentation: {path_to_info}")

        model_infos.append("\n".join(info_parts))

    return "\n\n".join(model_infos)

def main():
    config = load_models_config()

    for family_name, family_data in config.items():
        model_class = FAMILY_MODEL_CLASSES.get(family_name)
        models = family_data.get('models')

        for model_data in models:
            model_name = model_data.get('name')
            download_model(
                model_name=model_name,
                model_class=model_class,
            )

if __name__ == "__main__":
    main()