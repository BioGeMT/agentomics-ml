import requests
import dotenv
import os

dotenv.load_dotenv("/repository/.env")

PROVISIONING_API_KEY = os.getenv("PROVISIONING_OPENROUTER_API_KEY")
BASE_URL = "https://openrouter.ai/api/v1/keys"

def create_new_api_key(name, limit, proxies=None):
    response = requests.post(
        f"{BASE_URL}",
        headers={
            "Authorization": f"Bearer {PROVISIONING_API_KEY}",
            "Content-Type": "application/json"
        },
        json={
            "name": name,
            "limit": limit,
        },
        proxies=proxies,
    )
    assert response.status_code == 200 or response.status_code == 201, f"{response.json()}, {response.status_code}"
    hash = response.json()['data']['hash']
    key = response.json()['key']
    return {
        'hash': hash,
        'key': key,
    }


def get_api_key(key_hash, proxies=None):
    headers = {"Authorization": f"Bearer {PROVISIONING_API_KEY}"}
    response = requests.get(f"{BASE_URL}/{key_hash}", headers=headers, proxies=proxies)
    return response.json()

def get_all_api_keys(proxies=None):
    response = requests.get(
    BASE_URL,
        headers={
            "Authorization": f"Bearer {PROVISIONING_API_KEY}",
            "Content-Type": "application/json"
        },
        proxies=proxies,
    )
    return response.json()

def get_api_key_usage(key_hash, proxies=None):
    key_info = get_api_key(key_hash, proxies=proxies)
    data = {
        'limit': key_info['data']['limit'],
        'usage': key_info['data']['usage'],
    }
    return data

def delete_api_key(key_hash, proxies=None):
    response = requests.delete(
        f"{BASE_URL}/{key_hash}",
        headers={
            "Authorization": f"Bearer {PROVISIONING_API_KEY}",
            "Content-Type": "application/json"
        },
        proxies=proxies,
    )
    assert response.status_code == 200
    print("API KEY DELETED")

def delete_all_keys_with_a_name(name, proxies=None):
    keys = get_all_api_keys(proxies=proxies)
    print("Existing keys:")
    for key in keys['data']:
        print(f"Key with name {key['name']}")
    for key in keys['data']:
        if key['name'] == name:
            delete_api_key(key['hash'], proxies=proxies)
            print(f"Deleted key {key['hash']} with name {name}")

if __name__ == "__main__":
    delete_all_keys_with_a_name("test", proxies=None)