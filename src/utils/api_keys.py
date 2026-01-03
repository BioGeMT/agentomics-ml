import requests
import dotenv
import os
from pathlib import Path

dotenv.load_dotenv(Path(__file__).parents[2] / ".env") # load env in the root of the project

PROVISIONING_API_KEY = os.getenv("PROVISIONING_OPENROUTER_API_KEY")
BASE_URL = "https://openrouter.ai/api/v1/keys"

def create_new_api_key(name, limit):
    response = requests.post(
        f"{BASE_URL}",
        headers={
            "Authorization": f"Bearer {PROVISIONING_API_KEY}",
            "Content-Type": "application/json"
        },
        json={
            "name": name,
            "limit": limit,
        }
    )
    assert response.status_code == 200 or response.status_code == 201, f"{response.json()}, {response.status_code}"
    hash = response.json()['data']['hash']
    key = response.json()['key']
    return {
        'hash': hash,
        'key': key,
    }


def get_api_key(key_hash):
    headers = {"Authorization": f"Bearer {PROVISIONING_API_KEY}"}
    response = requests.get(f"{BASE_URL}/{key_hash}", headers=headers)

    if response.status_code != 200:
        print(f"API Error: Status {response.status_code}")
        print(f"Response: {response.text}")
        response.raise_for_status()

    return response.json()

def get_all_api_keys():
    response = requests.get(
    BASE_URL,
        headers={
            "Authorization": f"Bearer {PROVISIONING_API_KEY}",
            "Content-Type": "application/json"
        }
    )
    return response.json()

def get_api_key_usage(key_hash):
    key_info = get_api_key(key_hash)

    # Debug: print what we got
    print(f"API Response type: {type(key_info)}")
    print(f"API Response: {key_info}")

    # Handle unexpected response formats
    if not isinstance(key_info, dict):
        raise ValueError(f"Expected dict from API, got {type(key_info)}: {key_info}")

    if 'data' not in key_info:
        raise ValueError(f"Missing 'data' key in API response. Keys: {list(key_info.keys())}")

    data = {
        'limit': key_info['data']['limit'],
        'usage': key_info['data']['usage'],
    }
    return data

def delete_api_key(key_hash):
    response = requests.delete(
        f"{BASE_URL}/{key_hash}",
        headers={
            "Authorization": f"Bearer {PROVISIONING_API_KEY}",
            "Content-Type": "application/json"
        }
    )
    assert response.status_code == 200
    print("API KEY DELETED")

def delete_all_keys_with_a_name(name):
    keys = get_all_api_keys()
    print("Existing keys:")
    for key in keys['data']:
        print(f"Key with name {key['name']}")
    for key in keys['data']:
        if key['name'] == name:
            delete_api_key(key['hash'])
            print(f"Deleted key {key['hash']} with name {name}")

if __name__ == "__main__":
    delete_all_keys_with_a_name("test")