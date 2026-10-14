from __future__ import annotations

import json
from typing import Dict, List, Optional

import httpx2
from pydantic_ai.models.openai import OpenAIResponsesModel
from pydantic_ai.providers.openai_codex import OpenAICodexProvider
from rich.panel import Panel

from agentomics.utils.config import Config
from agentomics.utils.user_input import get_user_input_for_int

from .codex_auth import CodexAuthStore, get_codex_models_cache_path
from .provider import Provider


class CodexProvider(Provider):
    def __init__(self, list_models_endpoint: str | None = None):
        # OpenAICodexProvider always calls the ChatGPT Codex endpoint, so there is no base URL to configure.
        super().__init__(name="Codex", base_url="", list_models_endpoint=list_models_endpoint)
        self.auth_store = CodexAuthStore()

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

        lines = []
        max_num_width = len(str(len(models)))
        max_name_width = max(len(model.get("display_name", model.get("id", ""))) for model in models)

        for i, model in enumerate(models, 1):
            num_str = str(i).rjust(max_num_width)
            display_name = model.get("display_name", model.get("id", ""))
            description = model.get("description", "")
            lines.append(f"[dim]{num_str}.[/dim] [white]{display_name}[/white]")
            if description:
                lines.append(f"   [dim]{description}[/dim]")

        panel_width = min(max_num_width + max_name_width + 14, 120)
        panel = Panel(
            "\n".join(lines),
            title="[bold white]Codex[/bold white]",
            title_align="left",
            border_style="cyan",
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
        async_http_client = httpx2.AsyncClient(
            proxy=proxy_url if config.use_proxy else None,
            timeout=config.llm_response_timeout,
        )
        return OpenAIResponsesModel(
            model_name=model_name,
            provider=OpenAICodexProvider(
                credential_source=self.auth_store,
                http_client=async_http_client,
            ),
        )
