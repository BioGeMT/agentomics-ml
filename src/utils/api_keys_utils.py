import argparse
import json
from decimal import Decimal, InvalidOperation
import urllib.request
from pathlib import Path

def to_decimal(value):
    try:
        return Decimal(str(value))
    except (InvalidOperation, ValueError, TypeError) as exc:
        raise ValueError(f"Invalid numeric value from API: {value!r}") from exc

def format_decimal_for_csv(value):
    normalized = value.normalize()
    text = format(normalized, "f")
    if "." in text:
        text = text.rstrip("0").rstrip(".")
    return text or "0"

def decimal_to_json_number(value):
    if value == value.to_integral_value():
        return int(value)
    return float(value)

def load_run_config(config_path):
    from utils.config import Config

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
    from utils.api_keys import create_new_api_key, get_api_key_usage

    result = create_new_api_key(args.name, args.limit)
    usage_info = get_api_key_usage(result["hash"])
    limit = to_decimal(usage_info.get("limit"))
    usage = to_decimal(usage_info.get("usage"))
    remaining = limit - usage
    payload = {
        **result,
        "limit": decimal_to_json_number(limit),
        "usage": decimal_to_json_number(usage),
        "remaining": decimal_to_json_number(remaining),
    }
    if args.output == "json":
        print(json.dumps(payload))
    else:
        print(
            f"{result['key']},{result['hash']},"
            f"{format_decimal_for_csv(limit)},"
            f"{format_decimal_for_csv(usage)},"
            f"{format_decimal_for_csv(remaining)}"
        )

def cleanup_and_log(args):
    import dotenv
    import wandb
    from utils.api_keys import get_api_key_usage, delete_api_key
    from run_logging.wandb_setup import resume_wandb_run

    dotenv.load_dotenv()
    config = load_run_config(args.config_path)
    resume_wandb_run(config, dir='./cleanup_wandb_logs')

    usage = get_api_key_usage(args.api_key_hash)
    wandb.log({
        "api_usage/limit": usage['limit'],
        "api_usage/usage": usage['usage'],
    })
    print(f"Logged API usage: limit={usage['limit']}, usage={usage['usage']}")

    delete_api_key(args.api_key_hash)

def fetch_credits(args):
    req = urllib.request.Request(
        "https://openrouter.ai/api/v1/credits",
        headers={"Authorization": f"Bearer {args.api_key}"}
    )
    with urllib.request.urlopen(req, timeout=10) as response:
        payload = json.loads(response.read().decode())

    data = payload.get("data", {})
    total_credits = data.get("total_credits")
    total_usage = data.get("total_usage")
    if total_credits is None or total_usage is None:
        raise ValueError("OpenRouter credit API response missing total_credits or total_usage")
    total_credits_dec = to_decimal(total_credits)
    total_usage_dec = to_decimal(total_usage)
    remaining = total_credits_dec - total_usage_dec

    if args.output == "json":
        print(
            json.dumps(
                {
                    "total_credits": decimal_to_json_number(total_credits_dec),
                    "total_usage": decimal_to_json_number(total_usage_dec),
                    "remaining": decimal_to_json_number(remaining),
                }
            )
        )
    else:
        print(
            f"{format_decimal_for_csv(total_credits_dec)},"
            f"{format_decimal_for_csv(total_usage_dec)},"
            f"{format_decimal_for_csv(remaining)}"
        )

def main():
    parser = argparse.ArgumentParser(description="API key management")
    subparsers = parser.add_subparsers(dest="command", required=True)

    create_parser = subparsers.add_parser("create")
    create_parser.add_argument("--name", required=True)
    create_parser.add_argument("--limit", type=int, required=True)
    create_parser.add_argument("--output", choices=["csv", "json"], default="csv")

    cleanup_parser = subparsers.add_parser("cleanup-and-log")
    cleanup_parser.add_argument("--config-path", required=True)
    cleanup_parser.add_argument("--api-key-hash", required=True)

    credits_parser = subparsers.add_parser("credits")
    credits_parser.add_argument("--api-key", required=True)
    credits_parser.add_argument("--output", choices=["csv", "json"], default="csv")

    args = parser.parse_args()

    try:
        if args.command == "create":
            create_key(args)
        elif args.command == "cleanup-and-log":
            cleanup_and_log(args)
        elif args.command == "credits":
            fetch_credits(args)
    except Exception as e:
        print(f"Error: {str(e)}", file=__import__('sys').stderr)
        __import__('sys').exit(1)

if __name__ == "__main__":
    main()
