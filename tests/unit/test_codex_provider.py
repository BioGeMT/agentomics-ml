import asyncio
import base64
import json
import time
from pathlib import Path
from types import SimpleNamespace

from pydantic_ai.models.openai import OpenAIResponsesModel
from pydantic_ai.providers.openai_codex import OpenAICodexCredentials

from agentomics.utils.providers.codex_auth import CodexAuthStore
from agentomics.utils.providers.codex_provider import CodexProvider
from agentomics.utils.providers.provider import get_provided_api_keys

ACCOUNT_ID = "test-account-id"


def _make_jwt(payload: dict) -> str:
    header = {"alg": "none", "typ": "JWT"}

    def encode(part: dict) -> str:
        raw = json.dumps(part, separators=(",", ":")).encode("utf-8")
        return base64.urlsafe_b64encode(raw).decode("ascii").rstrip("=")

    return f"{encode(header)}.{encode(payload)}.signature"


def _write_auth_file(
    auth_path: Path,
    *,
    exp_offset_seconds: int = 3600,
    refresh_token: str = "refresh-token",
    include_account_id: bool = True,
) -> str:
    access_token = _make_jwt(
        {
            "exp": int(time.time()) + exp_offset_seconds,
            "https://api.openai.com/auth": {
                "chatgpt_account_id": ACCOUNT_ID,
            },
        }
    )
    tokens = {
        "access_token": access_token,
        "refresh_token": refresh_token,
        "id_token": _make_jwt({"exp": int(time.time()) + exp_offset_seconds}),
    }
    if include_account_id:
        tokens["account_id"] = ACCOUNT_ID
    auth_path.write_text(
        json.dumps(
            {
                "auth_mode": "chatgpt",
                "OPENAI_API_KEY": None,
                "last_refresh": "2026-04-01T00:00:00+00:00",
                "tokens": tokens,
            }
        ),
        encoding="utf-8",
    )
    return access_token


def test_account_id_falls_back_to_access_token_metadata(tmp_path: Path):
    auth_path = tmp_path / "auth.json"
    _write_auth_file(auth_path, include_account_id=False)

    account_id = CodexAuthStore(auth_path=auth_path).get_account_id()

    assert account_id == ACCOUNT_ID


def test_credentials_are_loaded_from_codex_auth_file(tmp_path: Path):
    auth_path = tmp_path / "auth.json"
    access_token = _write_auth_file(auth_path, include_account_id=False)

    credentials = asyncio.run(CodexAuthStore(auth_path=auth_path).load())

    assert credentials.access_token == access_token
    assert credentials.refresh_token == "refresh-token"
    assert credentials.account_id == ACCOUNT_ID


def test_valid_codex_auth_is_reported_as_available(tmp_path: Path, monkeypatch):
    auth_path = tmp_path / "auth.json"
    _write_auth_file(auth_path)
    monkeypatch.setenv("CODEX_AUTH_FILE", str(auth_path))

    provided = get_provided_api_keys()

    assert provided["Codex"] == ""


def test_model_cache_excludes_hidden_models(tmp_path: Path, monkeypatch):
    models_cache_path = tmp_path / "models_cache.json"
    models_cache_path.write_text(
        json.dumps(
            {
                "models": [
                    {
                        "slug": "gpt-5.4",
                        "display_name": "gpt-5.4",
                        "description": "Latest frontier agentic coding model.",
                        "supported_in_api": True,
                        "visibility": "list",
                        "priority": 1,
                    },
                    {
                        "slug": "gpt-5-codex-mini",
                        "display_name": "gpt-5-codex-mini",
                        "supported_in_api": True,
                        "visibility": "hide",
                        "priority": 2,
                    },
                ]
            }
        ),
        encoding="utf-8",
    )
    monkeypatch.setenv("CODEX_MODELS_CACHE_FILE", str(models_cache_path))
    provider = CodexProvider("")

    models = provider.fetch_models()

    assert [model["id"] for model in models] == ["gpt-5.4"]


def test_rotated_credentials_are_persisted_to_codex_auth_file(tmp_path: Path):
    auth_path = tmp_path / "auth.json"
    _write_auth_file(auth_path, refresh_token="old-refresh-token")
    store = CodexAuthStore(auth_path=auth_path)

    asyncio.run(
        store.save(
            OpenAICodexCredentials(
                access_token="new-access-token",
                refresh_token="new-refresh-token",
                account_id=ACCOUNT_ID,
            )
        )
    )

    saved_state = store.load_state()
    assert saved_state.access_token == "new-access-token"
    assert saved_state.refresh_token == "new-refresh-token"
    assert saved_state.raw_data["auth_mode"] == "chatgpt"
    assert saved_state.raw_data["tokens"]["id_token"]


def test_codex_model_uses_native_codex_provider(tmp_path: Path, monkeypatch):
    auth_path = tmp_path / "auth.json"
    _write_auth_file(auth_path)
    monkeypatch.setenv("CODEX_AUTH_FILE", str(auth_path))
    config = SimpleNamespace(use_proxy=False, llm_response_timeout=30)

    model = CodexProvider("").create_model("gpt-6-sol", config)

    assert isinstance(model, OpenAIResponsesModel)
    assert model.system == "openai-codex"
    assert "temperature" in model.profile["openai_unsupported_model_settings"]
