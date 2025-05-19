import requests
import dotenv
import os

dotenv.load_dotenv("/home/jovyan/Vlasta/Agentomics-ML/.env")

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