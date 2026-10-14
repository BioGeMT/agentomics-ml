from __future__ import annotations

import base64
import copy
import json
import os
from dataclasses import dataclass
from datetime import datetime, timezone
from pathlib import Path
from typing import Any

from pydantic_ai.providers.openai_codex import OpenAICodexCredentials, OpenAICodexCredentialSource


DEFAULT_CODEX_AUTH_PATH = Path.home() / ".codex" / "auth.json"
DEFAULT_CODEX_MODELS_CACHE_PATH = Path.home() / ".codex" / "models_cache.json"


def get_codex_auth_path() -> Path:
    return Path(os.getenv("CODEX_AUTH_FILE", DEFAULT_CODEX_AUTH_PATH)).expanduser()


def get_codex_models_cache_path() -> Path:
    return Path(os.getenv("CODEX_MODELS_CACHE_FILE", DEFAULT_CODEX_MODELS_CACHE_PATH)).expanduser()


def _decode_jwt_payload(token: str) -> dict[str, Any]:
    parts = token.split(".")
    if len(parts) < 2:
        return {}

    payload = parts[1]
    payload += "=" * (-len(payload) % 4)
    try:
        decoded = base64.urlsafe_b64decode(payload.encode("ascii"))
        data = json.loads(decoded.decode("utf-8"))
        return data if isinstance(data, dict) else {}
    except Exception:
        return {}


def _utcnow_iso() -> str:
    return datetime.now(timezone.utc).isoformat()


@dataclass(slots=True)
class CodexAuthState:
    access_token: str
    refresh_token: str
    account_id: str | None
    raw_data: dict[str, Any]

    @classmethod
    def from_raw(cls, raw_data: dict[str, Any]) -> "CodexAuthState":
        tokens = raw_data.get("tokens") or {}
        access_token = tokens.get("access_token")
        refresh_token = tokens.get("refresh_token")
        if not access_token or not refresh_token:
            raise ValueError("Codex auth file is missing access_token or refresh_token.")

        access_claims = _decode_jwt_payload(access_token)
        auth_claims = access_claims.get("https://api.openai.com/auth", {})
        account_id = tokens.get("account_id") or auth_claims.get("chatgpt_account_id")

        return cls(
            access_token=access_token,
            refresh_token=refresh_token,
            account_id=account_id,
            raw_data=raw_data,
        )


class CodexAuthStore(OpenAICodexCredentialSource):
    """Reads/writes the Codex CLI auth.json for OpenAICodexProvider."""

    def __init__(self, auth_path: Path | None = None):
        self.auth_path = (auth_path or get_codex_auth_path()).expanduser()

    @staticmethod
    def is_available(auth_path: Path | None = None) -> bool:
        path = (auth_path or get_codex_auth_path()).expanduser()
        if not path.is_file():
            return False
        try:
            CodexAuthState.from_raw(json.loads(path.read_text()))
            return True
        except Exception:
            return False

    def load_state(self) -> CodexAuthState:
        raw_data = json.loads(self.auth_path.read_text())
        return CodexAuthState.from_raw(raw_data)

    def get_account_id(self) -> str:
        account_id = self.load_state().account_id
        if not account_id:
            raise ValueError("Codex auth file is missing ChatGPT account metadata. Run `codex login` again.")
        return account_id

    async def load(self) -> OpenAICodexCredentials:
        state = self.load_state()
        return OpenAICodexCredentials(
            access_token=state.access_token,
            refresh_token=state.refresh_token,
            account_id=self.get_account_id(),
        )

    async def save(self, credentials: OpenAICodexCredentials) -> None:
        refreshed = copy.deepcopy(self.load_state().raw_data)
        refreshed.setdefault("tokens", {})
        refreshed["last_refresh"] = _utcnow_iso()
        refreshed["tokens"]["access_token"] = credentials.access_token
        refreshed["tokens"]["refresh_token"] = credentials.refresh_token
        refreshed["tokens"]["account_id"] = credentials.account_id
        self._write_state(refreshed)

    def _write_state(self, raw_data: dict[str, Any]) -> None:
        self.auth_path.parent.mkdir(parents=True, exist_ok=True)
        tmp_path = self.auth_path.with_suffix(f"{self.auth_path.suffix}.tmp")
        tmp_path.write_text(json.dumps(raw_data, indent=2) + "\n")
        try:
            os.chmod(tmp_path, 0o600)
        except OSError:
            pass
        tmp_path.replace(self.auth_path)
