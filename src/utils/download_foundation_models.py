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
    config = load_models_config()

    for _, family_data in config.items():
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