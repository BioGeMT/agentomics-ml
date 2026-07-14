from __future__ import annotations

import asyncio
import base64
import copy
import json
import os
import time
from dataclasses import dataclass
from datetime import datetime, timezone
from pathlib import Path
from typing import Any

import httpx


DEFAULT_CODEX_AUTH_PATH = Path.home() / ".codex" / "auth.json"
DEFAULT_CODEX_MODELS_CACHE_PATH = Path.home() / ".codex" / "models_cache.json"
DEFAULT_CODEX_TOKEN_URL = "https://auth.openai.com/oauth/token"
DEFAULT_CODEX_CLIENT_ID = "app_EMoamEEZ73f0CkXaXp7hrann"
TOKEN_REFRESH_SKEW_SECONDS = 300


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
    id_token: str | None
    account_id: str | None
    access_token_exp: int | None
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

        access_token_exp = access_claims.get("exp")
        if not isinstance(access_token_exp, int):
            access_token_exp = None

        return cls(
            access_token=access_token,
            refresh_token=refresh_token,
            id_token=tokens.get("id_token"),
            account_id=account_id,
            access_token_exp=access_token_exp,
            raw_data=raw_data,
        )

    def needs_refresh(self, skew_seconds: int = TOKEN_REFRESH_SKEW_SECONDS) -> bool:
        if self.access_token_exp is None:
            return False
        return time.time() >= self.access_token_exp - skew_seconds


class CodexAuthStore:
    def __init__(
        self,
        auth_path: Path | None = None,
        *,
        token_url: str | None = None,
        client_id: str | None = None,
        proxy_url: str | None = None,
        timeout: float = 30.0,
    ):
        self.auth_path = (auth_path or get_codex_auth_path()).expanduser()
        self.token_url = token_url or DEFAULT_CODEX_TOKEN_URL
        self.client_id = client_id or DEFAULT_CODEX_CLIENT_ID
        self.proxy_url = proxy_url
        self.timeout = timeout
        self._refresh_lock = asyncio.Lock()

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

    async def get_access_token(self) -> str:
        state = self.load_state()
        if not state.needs_refresh():
            return state.access_token

        async with self._refresh_lock:
            state = self.load_state()
            if not state.needs_refresh():
                return state.access_token

            refreshed_state = await self._refresh_state(state)
            self._write_state(refreshed_state.raw_data)
            return refreshed_state.access_token

    async def _refresh_state(self, state: CodexAuthState) -> CodexAuthState:
        form_data = {
            "grant_type": "refresh_token",
            "refresh_token": state.refresh_token,
            "client_id": self.client_id,
        }
        headers = {"Content-Type": "application/x-www-form-urlencoded"}

        async with httpx.AsyncClient(proxy=self.proxy_url, timeout=self.timeout) as client:
            response = await client.post(self.token_url, data=form_data, headers=headers)
            response.raise_for_status()

        token_response = response.json()
        if not isinstance(token_response, dict) or "access_token" not in token_response:
            raise ValueError("Codex token refresh did not return an access_token.")

        refreshed = copy.deepcopy(state.raw_data)
        refreshed.setdefault("tokens", {})
        refreshed["last_refresh"] = _utcnow_iso()
        refreshed["tokens"]["access_token"] = token_response["access_token"]
        refreshed["tokens"]["refresh_token"] = token_response.get("refresh_token") or state.refresh_token
        if token_response.get("id_token"):
            refreshed["tokens"]["id_token"] = token_response["id_token"]

        access_claims = _decode_jwt_payload(refreshed["tokens"]["access_token"])
        auth_claims = access_claims.get("https://api.openai.com/auth", {})
        if auth_claims.get("chatgpt_account_id"):
            refreshed["tokens"]["account_id"] = auth_claims["chatgpt_account_id"]

        return CodexAuthState.from_raw(refreshed)

    def _write_state(self, raw_data: dict[str, Any]) -> None:
        self.auth_path.parent.mkdir(parents=True, exist_ok=True)
        tmp_path = self.auth_path.with_suffix(f"{self.auth_path.suffix}.tmp")
        tmp_path.write_text(json.dumps(raw_data, indent=2) + "\n")
        try:
            os.chmod(tmp_path, 0o600)
        except OSError:
            pass
        tmp_path.replace(self.auth_path)
