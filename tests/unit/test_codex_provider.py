import asyncio
import base64
import json
import time
from contextlib import asynccontextmanager
from pathlib import Path
from unittest.mock import AsyncMock, MagicMock, patch

from pydantic_ai.messages import ModelRequest, SystemPromptPart, UserPromptPart
from pydantic_ai.models import ModelRequestParameters
from pydantic_ai.models.openai import OpenAIResponsesModel

from agentomics.utils.providers.codex_auth import CodexAuthStore
from agentomics.utils.providers.codex_provider import CodexProvider, CodexResponsesModel
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


def _capture_codex_outbound_messages(messages):
    captured_messages = None

    @asynccontextmanager
    async def capture_request_stream(
        model,
        outbound_messages,
        model_settings,
        model_request_parameters,
        run_context=None,
    ):
        nonlocal captured_messages
        captured_messages = outbound_messages
        yield MagicMock()

    async def send_request():
        provider = MagicMock()
        provider.client = MagicMock()
        provider.model_profile = None
        model = CodexResponsesModel("gpt-5.4", provider=provider)
        with patch.object(
            OpenAIResponsesModel,
            "request_stream",
            capture_request_stream,
        ):
            async with model.request_stream(
                messages,
                None,
                ModelRequestParameters(),
            ):
                pass

    asyncio.run(send_request())
    return captured_messages


def test_account_id_falls_back_to_access_token_metadata(tmp_path: Path):
    auth_path = tmp_path / "auth.json"
    _write_auth_file(auth_path, include_account_id=False)

    account_id = CodexAuthStore(auth_path=auth_path).get_account_id()

    assert account_id == ACCOUNT_ID


def test_unexpired_access_token_is_reused_without_network_request(tmp_path: Path):
    auth_path = tmp_path / "auth.json"
    access_token = _write_auth_file(auth_path)

    with patch("agentomics.utils.providers.codex_auth.httpx.AsyncClient") as client:
        returned_token = asyncio.run(
            CodexAuthStore(auth_path=auth_path).get_access_token()
        )

    assert returned_token == access_token
    client.assert_not_called()


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
    provider = CodexProvider("https://chatgpt.com/backend-api/codex", "")

    models = provider.fetch_models()

    assert [model["id"] for model in models] == ["gpt-5.4"]


def test_system_prompt_is_sent_as_codex_instructions():
    request = ModelRequest(
        parts=[SystemPromptPart(content="system"), UserPromptPart(content="user")]
    )

    outbound_messages = _capture_codex_outbound_messages([request])

    outbound_request = outbound_messages[0]
    assert outbound_request.instructions == "system"
    assert [part.content for part in outbound_request.parts] == ["user"]


def test_codex_instructions_are_carried_to_followup_requests():
    requests = [
        ModelRequest(
            parts=[
                SystemPromptPart(content="system"),
                UserPromptPart(content="user"),
            ]
        ),
        ModelRequest(parts=[UserPromptPart(content="follow-up")]),
    ]

    outbound_messages = _capture_codex_outbound_messages(requests)

    assert outbound_messages[0].instructions == "system"
    assert outbound_messages[1].instructions == "system"


def test_expiring_access_token_is_refreshed_and_persisted(tmp_path: Path):
    auth_path = tmp_path / "auth.json"
    _write_auth_file(
        auth_path,
        exp_offset_seconds=-60,
        refresh_token="old-refresh-token",
    )
    refreshed_access_token = _make_jwt(
        {
            "exp": int(time.time()) + 3600,
            "https://api.openai.com/auth": {
                "chatgpt_account_id": ACCOUNT_ID,
            },
        }
    )
    response = MagicMock()
    response.json.return_value = {
        "access_token": refreshed_access_token,
        "refresh_token": "new-refresh-token",
        "id_token": _make_jwt({"exp": int(time.time()) + 3600}),
    }
    http_client = AsyncMock()
    http_client.post.return_value = response
    http_client_context = AsyncMock()
    http_client_context.__aenter__.return_value = http_client
    http_client_context.__aexit__.return_value = None

    with patch(
        "agentomics.utils.providers.codex_auth.httpx.AsyncClient",
        return_value=http_client_context,
    ):
        access_token = asyncio.run(
            CodexAuthStore(auth_path=auth_path).get_access_token()
        )

    assert access_token == refreshed_access_token
    refresh_request = http_client.post.await_args.kwargs["data"]
    assert refresh_request["grant_type"] == "refresh_token"
    assert refresh_request["refresh_token"] == "old-refresh-token"
    saved_state = CodexAuthStore(auth_path=auth_path).load_state()
    assert saved_state.access_token == refreshed_access_token
    assert saved_state.refresh_token == "new-refresh-token"
