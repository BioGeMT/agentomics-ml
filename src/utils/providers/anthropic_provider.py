import httpx
import os
from typing import List, Dict

from pydantic_ai.models.anthropic import AnthropicModel
from pydantic_ai.providers.anthropic import AnthropicProvider as PydanticAnthropicProvider # would clash with our class name
from rich.panel import Panel

from utils.config import Config
from .provider import Provider

class AnthropicProvider(Provider):
    def __init__(self, api_key: str, base_url: str, list_models_endpoint: str):
        super().__init__(name="Anthropic", api_key=api_key, base_url=base_url, list_models_endpoint=list_models_endpoint)
        self.headers = {"x-api-key": api_key, "anthropic-version": "2023-06-01"}
    
    def display_models(self, models: List[Dict] = None) -> None:
        """Ovveriding method in Provider class. Anthropic available models in a table format."""
        if models is None:
            models = self.fetch_models()

        self.console.print(f"[bold blue]Available Models[/bold blue] ({len(models)} models)\n")

        lines = []
        max_num_width = len(str(len(models)))
        max_name_width = max(len(model.get('display_name', '')) for model in models)

        for i, model in enumerate(models, 1):
            num_str = str(i).rjust(max_num_width)
            lines.append(f"[dim]{num_str}.[/dim] [cyan]{model.get('display_name', '')}[/cyan]")

        panel_width = max_num_width + max_name_width + 10
        panel = Panel("\n".join(lines), title="[bold green]Anthropic[/bold green]",
                     title_align="left", border_style="green", width=panel_width)
        self.console.print(panel)

    def create_model(self, model_name: str, config: Config) -> AnthropicModel:
        """Ovveriding method in Provider class. Create Anthropic model instance."""
        proxy_url = os.getenv("HTTP_PROXY")

        async_http_client = httpx.AsyncClient(
            proxy=proxy_url if config.use_proxy else None,
            timeout=config.llm_response_timeout
        )

        return AnthropicModel(
            model_name=model_name,
            provider=PydanticAnthropicProvider( #no need to specify base url
                api_key=self.api_key,
                http_client=async_http_client
            )
        )