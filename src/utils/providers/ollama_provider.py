import requests
from typing import List, Optional, Dict

from rich.table import Table
from rich import box
from pydantic_ai.providers.openai import OpenAIProvider
from pydantic_ai.models.openai import OpenAIModel

from utils.config import Config
from utils.user_input import get_user_input_for_int
from .provider import Provider

class OllamaProvider(Provider):
    def __init__(self, base_url: str, list_models_endpoint: str):
        super().__init__(name="Ollama", base_url=base_url, list_models_endpoint=list_models_endpoint)

    def fetch_models(self) -> Optional[List[Dict]]:
        """Ovveriding method in Provider class. Fetch all models from Ollama API locally."""
        try:
            response = requests.get(self.list_models_endpoint, timeout=10)
            response.raise_for_status()
            models = response.json().get("models", [])
            return models
        except Exception as e:
            self.console.print(f"Failed to fetch models: {e}")
            return None
    
    def interactive_model_selection(self, limit: int = None) -> Optional[str]:
        """Ovveriding method in Provider class. Interactive Ollama model selection."""
        models = self.fetch_models()

        if not models:
            self.console.print("Could not fetch models from Ollama", style="red")
            return None
        
        if limit and limit < len(models): models = models[:limit]

        self.display_models(models)
        
        choice = get_user_input_for_int(
          prompt_text=f"Select model (1-{len(models)}) or Enter for first",
          default=1,
          valid_options=list(range(1, len(models) + 1))
        )

        if choice:
            return models[choice - 1].get("model")
        
        return None
    
    def display_models(self, models: List[dict] = None) -> None:
        """Ovveriding method in Provider class. Display Ollama available models in a table format."""
        if models is None:
            models = self.fetch_models()
        
        table = Table(title=f"Available Ollama Models ({len(models)} found)", box=box.ROUNDED)
        table.add_column("#", style="cyan", no_wrap=True, width=4)
        table.add_column("Model Name", style="green")
        table.add_column("Parameter Size", style="yellow")
        table.add_column("Quantization", style="blue")
        
        for i, model in enumerate(models, 1):
            model_name = model.get("model")
            parameter_size = model.get("details", {}).get("parameter_size", "N/A")
            quantization_level = model.get("details", {}).get("quantization_level", "N/A")
            
            table.add_row(str(i), model_name, parameter_size, quantization_level)
        
        self.console.print(table)

    def create_model(self, model_name: str, config: Config) -> OpenAIModel:
          """Ovveriding method in Provider class. Create OpenAI model instance (Ollama compatible)."""
          return OpenAIModel(
              model_name=model_name,
              provider=OpenAIProvider(base_url=self.base_url)
          )