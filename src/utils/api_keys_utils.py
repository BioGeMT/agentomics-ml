import argparse
import json
import dotenv
import wandb
from pathlib import Path

from utils.config import Config
from utils.api_keys import create_new_api_key, get_api_key_usage, delete_api_key
from run_logging.wandb_setup import resume_wandb_run

def load_run_config(config_path):
    with open(config_path, 'r') as f:
        config_dict = json.load(f)

    config = Config(
        agent_id=config_dict['agent_id'],
        model_name=config_dict['model_name'],
        feedback_model_name=config_dict['feedback_model_name'],
        dataset=config_dict['dataset'],
        tags=config_dict['tags'],
        val_metric=config_dict['val_metric'],
        workspace_dir=Path(config_dict['workspace_dir']),
        prepared_datasets_dir=Path(config_dict['prepared_dataset_dir']).parent,
        prepared_test_sets_dir=Path(config_dict['prepared_test_set_dir']).parent,
        agent_datasets_dir=Path(config_dict['agent_dataset_dir']).parent,
        user_prompt=config_dict['user_prompt'],
        iterations=config_dict['iterations'],
        task_type=config_dict['task_type'],
    )
    config.wandb_run_id = config_dict.get('wandb_run_id')
    return config

def create_key(args):
    result = create_new_api_key(args.name, args.limit)
    print(f"{result['key']},{result['hash']}")

def cleanup_and_log(args):
    dotenv.load_dotenv()
    config = load_run_config(args.config_path)
    resume_wandb_run(config)

    usage = get_api_key_usage(args.api_key_hash)
    wandb.log({
        "api_usage/limit": usage['limit'],
        "api_usage/usage": usage['usage'],
    })
    print(f"Logged API usage: limit={usage['limit']}, usage={usage['usage']}")

    delete_api_key(args.api_key_hash)

def main():
    parser = argparse.ArgumentParser(description="API key management")
    subparsers = parser.add_subparsers(dest="command", required=True)

    create_parser = subparsers.add_parser("create")
    create_parser.add_argument("--name", required=True)
    create_parser.add_argument("--limit", type=int, required=True)

    cleanup_parser = subparsers.add_parser("cleanup-and-log")
    cleanup_parser.add_argument("--config-path", required=True)
    cleanup_parser.add_argument("--api-key-hash", required=True)

    args = parser.parse_args()

    try:
        if args.command == "create":
            create_key(args)
        elif args.command == "cleanup-and-log":
            cleanup_and_log(args)
    except Exception as e:
        print(f"Error: {str(e)}", file=__import__('sys').stderr)
        __import__('sys').exit(1)

if __name__ == "__main__":
    main()