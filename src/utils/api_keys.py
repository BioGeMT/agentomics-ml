import requests
import os

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

if __name__ == "__main__":
    delete_all_keys_with_a_name("test")
