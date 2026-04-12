import argparse
from foundation_models_utils import load_models_config

from transformers import AutoTokenizer
from transformers import AutoModel, AutoModelForMaskedLM
import multimolecule  # noqa: F401  # Required for loading specific HF models with remote code.


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
    parser = argparse.ArgumentParser(description="Download foundation models")
    parser.add_argument("--foundation-models-type", required=True, help="Foundation model type to download (dna, rna, molecule, protein, all)")
    parser.add_argument("--models-yaml", required=True, help="Path to the foundation models YAML config file")
    args = parser.parse_args()

    config = load_models_config(args.models_yaml)
    if config is None:
        print('INFO: NO FOUNDATION MODELS FOUND IN CONFIG')
        return

    for _, family_data in config.items():
        if args.foundation_models_type != "all" and family_data.get("type") != args.foundation_models_type:
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
