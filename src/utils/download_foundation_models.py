import os
from foundation_models_utils import load_models_config
from transformers import AutoTokenizer, AutoModel, AutoModelForMaskedLM
import multimolecule

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

def main():
    enabled_type = os.environ.get("FOUNDATION_MODEL_TYPE")
    if not enabled_type:
        return

    config = load_models_config()
    if config is None:
        print('INFO: NO FOUNDATION MODELS FOUND IN CONFIG')
        return

    for _, family_data in config.items():
        if family_data.get("type") != enabled_type:
            continue
        models = family_data.get('models')
        hf_class = AutoModel if family_data.get('can_load_with_hf_automodel') else AutoModelForMaskedLM
        for model_data in models:
            model_name = model_data.get('name')
            download_model(
                model_name=model_name,
                model_class=hf_class,
            )

if __name__ == "__main__":
    main()
