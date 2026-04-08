import base64
import json
import os
import sys
import tempfile
import time
import unittest
from pathlib import Path
from unittest.mock import AsyncMock, MagicMock, patch

REPO_ROOT = Path(__file__).resolve().parents[1]
SRC_PATH = REPO_ROOT / "src"
if str(SRC_PATH) not in sys.path:
    sys.path.insert(0, str(SRC_PATH))

from pydantic_ai.messages import ModelRequest, SystemPromptPart, UserPromptPart

from src.utils.providers.codex_auth import CodexAuthStore
from src.utils.providers.codex_provider import CodexProvider, CodexResponsesModel
from src.utils.providers.provider import get_provided_api_keys


def make_jwt(payload: dict) -> str:
    header = {"alg": "none", "typ": "JWT"}

    def encode(part: dict) -> str:
        raw = json.dumps(part, separators=(",", ":")).encode("utf-8")
        return base64.urlsafe_b64encode(raw).decode("ascii").rstrip("=")

    return f"{encode(header)}.{encode(payload)}.signature"


class CodexTestCase(unittest.TestCase):
    def setUp(self):
        self.tmpdir = tempfile.TemporaryDirectory()
        self.auth_path = Path(self.tmpdir.name) / "auth.json"
        self.models_cache_path = Path(self.tmpdir.name) / "models_cache.json"
        self.account_id = "test-account-id"

    def tearDown(self):
        self.tmpdir.cleanup()

    def write_auth_file(self, *, exp_offset_seconds: int = 3600, refresh_token: str = "refresh-token") -> str:
        access_token = make_jwt(
            {
                "exp": int(time.time()) + exp_offset_seconds,
                "https://api.openai.com/auth": {
                    "chatgpt_account_id": self.account_id,
                },
            }
        )
        id_token = make_jwt({"exp": int(time.time()) + exp_offset_seconds})
        auth_payload = {
            "auth_mode": "chatgpt",
            "OPENAI_API_KEY": None,
            "last_refresh": "2026-04-01T00:00:00+00:00",
            "tokens": {
                "access_token": access_token,
                "refresh_token": refresh_token,
                "id_token": id_token,
                "account_id": self.account_id,
            },
        }
        self.auth_path.write_text(json.dumps(auth_payload))
        return access_token


class TestCodexAuthStore(CodexTestCase):
    def test_load_state_extracts_account_metadata(self):
        access_token = self.write_auth_file()

        store = CodexAuthStore(auth_path=self.auth_path)
        state = store.load_state()

        self.assertEqual(state.access_token, access_token)
        self.assertEqual(state.account_id, self.account_id)
        self.assertFalse(state.needs_refresh())

    def test_get_provided_api_keys_includes_codex_when_auth_file_exists(self):
        self.write_auth_file()

        with patch.dict(os.environ, {"CODEX_AUTH_FILE": str(self.auth_path)}, clear=True):
            provided = get_provided_api_keys()

        self.assertIn("Codex", provided)
        self.assertEqual(provided["Codex"], "")

    def test_codex_provider_reads_visible_models_from_cache(self):
        self.write_auth_file()
        self.models_cache_path.write_text(
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
            )
        )

        with patch.dict(
            os.environ,
            {
                "CODEX_AUTH_FILE": str(self.auth_path),
                "CODEX_MODELS_CACHE_FILE": str(self.models_cache_path),
            },
            clear=True,
        ):
            provider = CodexProvider("https://chatgpt.com/backend-api/codex", "")
            models = provider.fetch_models()

        self.assertEqual([model["id"] for model in models], ["gpt-5.4"])

    def test_codex_rewrites_system_prompts_into_instructions(self):
        request = ModelRequest(parts=[SystemPromptPart(content="system"), UserPromptPart(content="user")])

        rewritten = CodexResponsesModel._rewrite_messages_for_codex([request])

        self.assertEqual(len(rewritten), 1)
        rewritten_request = rewritten[0]
        self.assertIsInstance(rewritten_request, ModelRequest)
        self.assertEqual(rewritten_request.instructions, "system")
        self.assertEqual([type(part).__name__ for part in rewritten_request.parts], ["UserPromptPart"])


class TestCodexRefresh(CodexTestCase, unittest.IsolatedAsyncioTestCase):
    async def test_refreshes_expiring_access_token_and_persists_new_tokens(self):
        self.write_auth_file(exp_offset_seconds=-60, refresh_token="old-refresh-token")
        refreshed_access_token = make_jwt(
            {
                "exp": int(time.time()) + 3600,
                "https://api.openai.com/auth": {
                    "chatgpt_account_id": self.account_id,
                },
            }
        )

        response = MagicMock()
        response.raise_for_status = MagicMock()
        response.json.return_value = {
            "access_token": refreshed_access_token,
            "refresh_token": "new-refresh-token",
            "id_token": make_jwt({"exp": int(time.time()) + 3600}),
        }

        http_client = AsyncMock()
        http_client.post.return_value = response

        http_client_cm = AsyncMock()
        http_client_cm.__aenter__.return_value = http_client
        http_client_cm.__aexit__.return_value = None

        with patch("src.utils.providers.codex_auth.httpx.AsyncClient", return_value=http_client_cm):
            store = CodexAuthStore(auth_path=self.auth_path)
            access_token = await store.get_access_token()

        self.assertEqual(access_token, refreshed_access_token)
        self.assertEqual(http_client.post.await_count, 1)
        refresh_request = http_client.post.await_args.kwargs["data"]
        self.assertEqual(refresh_request["grant_type"], "refresh_token")
        self.assertEqual(refresh_request["refresh_token"], "old-refresh-token")

        saved_state = CodexAuthStore(auth_path=self.auth_path).load_state()
        self.assertEqual(saved_state.access_token, refreshed_access_token)
        self.assertEqual(saved_state.refresh_token, "new-refresh-token")
