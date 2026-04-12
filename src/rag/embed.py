import os

import requests
from dotenv import load_dotenv

load_dotenv()

OPENROUTER_URL = "https://openrouter.ai/api/v1/embeddings"
EMBED_MODEL = "qwen/qwen3-embedding-8b"
EMBED_DIMS = 4096


def embed(texts: list[str]) -> list[list[float]]:
    api_key = os.environ.get("OPENROUTER_API_KEY")
    if not api_key:
        raise RuntimeError(
            "OPENROUTER_API_KEY is not set in the environment. "
            "Add it to .env or export it before running the RAG pipeline."
        )

    headers = {
        "Authorization": f"Bearer {api_key}",
        "Content-Type": "application/json",
    }
    payload = {"model": EMBED_MODEL, "input": texts}

    resp = requests.post(OPENROUTER_URL, headers=headers, json=payload)
    resp.raise_for_status()
    data = resp.json()
    return [item["embedding"] for item in data["data"]]
