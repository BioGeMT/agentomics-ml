import requests
from typing import List, Optional, Dict

from rich.panel import Panel
from rich.columns import Columns
from rich.console import Group
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

        families = {}
        for model in models:
            family = model.get("details", {}).get("family", "unknown")
            if family not in families:
                families[family] = []
            families[family].append(model)

        self.console.print(f"[bold blue]Available Models[/bold blue] ({len(models)} models from {len(families)} families)\n")

        family_boxes = []
        global_index = 1
        max_num_width = len(str(len(models)))

        for family, family_models in families.items():
            lines = []
            for model in family_models:
                model_name = model.get("model")
                parameter_size = model.get("details", {}).get("parameter_size", "N/A")
                quantization_level = model.get("details", {}).get("quantization_level", "N/A")

                num_str = str(global_index).rjust(max_num_width)
                lines.append(f"[dim]{num_str}.[/dim] [cyan]{model_name}[/cyan]")
                lines.append(f"   Size: [yellow]{parameter_size}[/yellow]  Quant: [blue]{quantization_level}[/blue]")
                global_index += 1

            panel = Panel("\n".join(lines), title=f"[bold green]{family.title()}[/bold green]",
                         title_align="left", border_style="green")
            family_boxes.append((len(family_models) * 2, panel))

        num_cols = 4
        columns = [[] for _ in range(num_cols)]
        col_heights = [0] * num_cols

        sorted_boxes = sorted(family_boxes, key=lambda x: x[0], reverse=True)

        for box_height, panel in sorted_boxes:
            min_col = col_heights.index(min(col_heights))
            columns[min_col].append(panel)
            col_heights[min_col] += box_height

        col_renderables = [Group(*col) for col in columns if col]
        self.console.print(Columns(col_renderables, padding=(0, 1), expand=False))

    def create_model(self, model_name: str, config: Config) -> OpenAIModel:
          """Ovveriding method in Provider class. Create OpenAI model instance (Ollama compatible)."""
          return OpenAIModel(
              model_name=model_name,
              provider=OpenAIProvider(base_url=self.base_url)
          )