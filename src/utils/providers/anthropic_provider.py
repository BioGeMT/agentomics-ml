import httpx
import os
from typing import List, Dict

from pydantic_ai.models.anthropic import AnthropicModel
from pydantic_ai.providers.anthropic import AnthropicProvider as PydanticAnthropicProvider # would clash with our class name
from rich.table import Table
from rich import box

from utils.config import Config
from .provider import Provider

class AnthropicProvider(Provider):
    def __init__(self, api_key: str, base_url: str, list_models_endpoint: str):
        super().__init__(name="Anthropic", base_url=base_url, list_models_endpoint=list_models_endpoint)
        self.api_key = api_key
        self.headers = {"x-api-key": api_key, "anthropic-version": "2023-06-01"}
    
    def display_models(self, models: List[Dict] = None) -> None:
        """Ovveriding method in Provider class. Anthropic available models in a table format."""
        if models is None:
            models = self.fetch_models()

        table = Table(title=f"Available Anthropic Models ({len(models)} found)", box=box.ROUNDED)
        table.add_column("#", style="cyan", no_wrap=True, width=4)
        table.add_column("Model", style="green")
        
        for i, model in enumerate(models, 1):
            display_name = model.get("display_name")
            table.add_row(str(i), display_name)
        
        self.console.print(table)

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