import yaml
import argparse
from pathlib import Path

def set_config(config_path, api_type, model, base_url, api_key):
    # create all necessary thing in path if necessary
    Path(config_path).parent.mkdir(parents=True, exist_ok=True)
    config = {
        "llm": {
            "api_type": f"{api_type}",
            "model": f"{model}",
            "base_url": f"{base_url}",
            "api_key": f"{api_key}",
        }
    }
    with open(config_path, "w") as f:
        yaml.dump(config, f)

if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Set config for DI")
    parser.add_argument("--config-path", type=str, required=True, help="Path to the config file")
    parser.add_argument("--api-type", type=str, required=True, help="API type (e.g., openrouter)")
    parser.add_argument("--model", type=str, required=True, help="Model name (e.g., gpt-4o-2024-08-06)")
    parser.add_argument("--base-url", type=str, required=True, help="Base URL for the API")
    parser.add_argument("--api-key", type=str, required=True, help="API key for authentication")
    
    args = parser.parse_args()
    
    set_config(args.config_path, args.api_type, args.model, args.base_url, args.api_key)
