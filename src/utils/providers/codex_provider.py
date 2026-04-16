from __future__ import annotations

from contextlib import asynccontextmanager
from dataclasses import replace
import json
from typing import Any, AsyncIterator, Dict, List, Optional

import httpx
from openai import AsyncOpenAI
from pydantic_ai.models.openai import OpenAIResponsesModel
from pydantic_ai.messages import ModelMessage, ModelRequest, ModelResponse, SystemPromptPart
from pydantic_ai.models import ModelRequestParameters, StreamedResponse
from pydantic_ai.providers.openai import OpenAIProvider as PydanticOpenAIProvider
from pydantic_ai.settings import ModelSettings
from rich.panel import Panel

from utils.config import Config
from utils.user_input import get_user_input_for_int

from .codex_auth import CodexAuthStore, get_codex_models_cache_path
from .provider import Provider


class AsyncCodexOpenAI(AsyncOpenAI):
    def __init__(self, auth_store: CodexAuthStore, **kwargs):
        self._codex_auth_store = auth_store
        super().__init__(api_key=self._codex_auth_store.get_access_token, **kwargs)

    @property
    def default_headers(self) -> dict[str, str]:
        headers = super().default_headers
        headers["ChatGPT-Account-Id"] = self._codex_auth_store.get_account_id()
        return headers


class CodexResponsesModel(OpenAIResponsesModel):
    @staticmethod
    def _merge_codex_settings(model_settings: ModelSettings | None) -> ModelSettings:
        merged_settings = dict(model_settings or {})
        merged_settings["openai_store"] = False
        return merged_settings

    @staticmethod
    def _rewrite_messages_for_codex(messages: list[ModelMessage]) -> list[ModelMessage]:
        rewritten_messages: list[ModelMessage] = []
        carried_instructions: str | None = None
        for message in messages:
            if not isinstance(message, ModelRequest):
                rewritten_messages.append(message)
                continue

            system_parts = [part.content for part in message.parts if isinstance(part, SystemPromptPart)]

            instructions = message.instructions or carried_instructions

            if system_parts:
                joined = "\n\n".join(system_parts)
                instructions = "\n\n".join([instructions, joined]) if instructions else joined

            if instructions:
                carried_instructions = instructions

            rewritten_messages.append(
                replace(
                    message,
                    parts=[part for part in message.parts if not isinstance(part, SystemPromptPart)],
                    instructions=instructions,
                )
            )

        return rewritten_messages
    async def request(
        self,
        messages: list[ModelRequest | ModelResponse],
        model_settings: ModelSettings | None,
        model_request_parameters: ModelRequestParameters,
    ) -> ModelResponse:
        messages = self._rewrite_messages_for_codex(messages)
        async with super().request_stream(
            messages,
            self._merge_codex_settings(model_settings),
            model_request_parameters,
        ) as response:
            async for _ in response:
                pass
            return response.get()

    @asynccontextmanager
    async def request_stream(
        self,
        messages: list[ModelMessage],
        model_settings: ModelSettings | None,
        model_request_parameters: ModelRequestParameters,
        run_context: Any | None = None,
    ) -> AsyncIterator[StreamedResponse]:
        messages = self._rewrite_messages_for_codex(messages)
        async with super().request_stream(
            messages,
            self._merge_codex_settings(model_settings),
            model_request_parameters,
            run_context=run_context,
        ) as response:
            yield response


class CodexProvider(Provider):
    def __init__(self, base_url: str, list_models_endpoint: str | None = None):
        super().__init__(name="Codex", base_url=base_url, list_models_endpoint=list_models_endpoint)
        self.auth_store = CodexAuthStore(proxy_url=self.get_proxy_url())

    def fetch_models(self) -> Optional[List[Dict]]:
        cache_path = get_codex_models_cache_path()
        if not cache_path.is_file():
            self.console.print(f"Codex model cache not found: {cache_path}", style="red")
            return None

        try:
            raw_models = json.loads(cache_path.read_text()).get("models", [])
            models: list[dict] = []
            for raw_model in raw_models:
                if raw_model.get("supported_in_api") is False:
                    continue
                if raw_model.get("visibility") == "hide":
                    continue

                slug = raw_model.get("slug") or raw_model.get("id")
                if not slug:
                    continue

                models.append(
                    {
                        "id": slug,
                        "display_name": raw_model.get("display_name", slug),
                        "description": raw_model.get("description", ""),
                        "priority": raw_model.get("priority", 999),
                    }
                )

            if not models:
                self.console.print("No Codex models available in the local cache.", style="red")
                return None

            models.sort(key=lambda model: (model.get("priority", 999), model.get("id", "")))
            return models
        except Exception as e:
            self.console.print(f"Failed to read Codex model cache: {e}")
            return None

    def display_models(self, models: List[Dict] = None) -> None:
        if models is None:
            models = self.fetch_models()
        if not models:
            return

        self.console.print(f"[bold blue]Available Models[/bold blue] ({len(models)} models)\n")

        lines = []
        max_num_width = len(str(len(models)))
        max_name_width = max(len(model.get("display_name", model.get("id", ""))) for model in models)

        for i, model in enumerate(models, 1):
            num_str = str(i).rjust(max_num_width)
            display_name = model.get("display_name", model.get("id", ""))
            description = model.get("description", "")
            lines.append(f"[dim]{num_str}.[/dim] [cyan]{display_name}[/cyan]")
            if description:
                lines.append(f"   [dim]{description}[/dim]")

        panel_width = min(max_num_width + max_name_width + 14, 120)
        panel = Panel(
            "\n".join(lines),
            title="[bold green]Codex[/bold green]",
            title_align="left",
            border_style="green",
            width=panel_width,
        )
        self.console.print(panel)

    def interactive_model_selection(self, limit: int = None) -> Optional[str]:
        models = self.fetch_models()
        if not models:
            self.console.print("Could not load Codex models", style="red")
            return None

        if limit and limit < len(models):
            models = models[:limit]

        self.display_models(models)

        choice = get_user_input_for_int(
            prompt_text=f"Select model (1-{len(models)}) or Enter for first",
            default=1,
            valid_options=list(range(1, len(models) + 1)),
        )

        if choice:
            return models[choice - 1].get("id")

        return None

    def create_model(self, model_name: str, config: Config) -> OpenAIResponsesModel:
        # Fail early with a clear message instead of deferring until the first request.
        self.auth_store.get_account_id()

        proxy_url = self.get_proxy_url()
        async_http_client = httpx.AsyncClient(
            proxy=proxy_url if config.use_proxy else None,
            timeout=config.llm_response_timeout,
        )
        client = AsyncCodexOpenAI(
            auth_store=self.auth_store,
            base_url=self.base_url,
            http_client=async_http_client,
        )
        return CodexResponsesModel(
            model_name=model_name,
            provider=PydanticOpenAIProvider(openai_client=client),
        )
