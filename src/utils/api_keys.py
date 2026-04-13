import argparse
import os
import sys

import dotenv
import requests
import wandb

from run_logging.wandb_setup import resume_wandb_run
from runtime.read_write_utils import load_config


def create_new_api_key(name, limit):
    response = requests.post(
        _get_base_url(),
        headers=_build_headers(),
        json={
            "name": name,
            "limit": limit,
        },
        timeout=30,
    )
    response.raise_for_status()
    payload = response.json()
    hash = payload["data"]["hash"]
    key = payload["key"]
    return {
        "hash": hash,
        "key": key,
    }

def get_api_key(key_hash):
    response = requests.get(
        f"{_get_base_url()}/{key_hash}",
        headers=_build_headers(include_content_type=False),
        timeout=30,
    )
    response.raise_for_status()
    return response.json()

def get_all_api_keys():
    response = requests.get(
        _get_base_url(),
        headers=_build_headers(),
        timeout=30,
    )
    response.raise_for_status()
    return response.json()

def get_api_key_usage(key_hash):
    key_info = get_api_key(key_hash)
    data = {
        "limit": key_info["data"]["limit"],
        "usage": key_info["data"]["usage"],
    }
    return data

def delete_api_key(key_hash):
    response = requests.delete(
        f"{_get_base_url()}/{key_hash}",
        headers=_build_headers(),
        timeout=30,
    )
    response.raise_for_status()
    print("API KEY DELETED")

def delete_all_keys_with_a_name(name):
    keys = get_all_api_keys()
    print("Existing keys:")
    for key in keys["data"]:
        print(f"Key with name {key['name']}")
    for key in keys["data"]:
        if key["name"] == name:
            delete_api_key(key["hash"])
            print(f"Deleted key {key['hash']} with name {name}")

def _get_base_url() -> str:
    return "https://openrouter.ai/api/v1/keys"

def _build_headers(*, include_content_type: bool = True) -> dict[str, str]:
    headers = {"Authorization": f"Bearer {_get_provisioning_api_key()}"}
    if include_content_type:
        headers["Content-Type"] = "application/json"
    return headers

def _get_provisioning_api_key() -> str:
    api_key = os.getenv("PROVISIONING_OPENROUTER_API_KEY")
    if not api_key:
        raise ValueError("Environment variable PROVISIONING_OPENROUTER_API_KEY is not set.")
    return api_key


def _create_key(args):
    result = create_new_api_key(args.name, args.limit)
    print(f"{result['key']},{result['hash']}")

def _cleanup_and_log(args):
    dotenv.load_dotenv()
    config = load_config(args.config_path)
    resume_wandb_run(config, dir='./cleanup_wandb_logs')

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
            _create_key(args)
        elif args.command == "cleanup-and-log":
            _cleanup_and_log(args)
    except Exception as e:
        print(f"Error: {str(e)}", file=sys.stderr)
        sys.exit(1)

if __name__ == "__main__":
    main()
