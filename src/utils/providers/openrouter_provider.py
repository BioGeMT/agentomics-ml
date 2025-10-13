from typing import Dict, List, Optional

from rich.table import Table
from rich import box

from utils.user_input import get_user_input_for_int
from .provider import Provider

class OpenRouterProvider(Provider):
    def __init__(self, api_key: str, base_url: str, list_models_endpoint: str):
        super().__init__(name="OpenRouter", api_key=api_key, base_url=base_url, list_models_endpoint=list_models_endpoint)
        self.headers = {"Authorization": f"Bearer {api_key}"}
    
    def is_byok_model(self, model: Dict) -> bool:
        """Check if model requires separate API key (BYOK)."""
        description = model.get("description", "").lower()
        
        # Check for BYOK indicators
        byok_indicators = ["byok", "requires api key", "add your api key", "bring your own key"]
        if any(indicator in description for indicator in byok_indicators):
            return True

        return False
    
    def is_coding_model(self, model: Dict) -> bool:
        model_id = model.get("id", "").lower()
        description = model.get("description", "").lower()
        
        exclude_patterns = ["whisper", "dall-e", "tts", "embedding", "vision", "image", "audio"]
        if any(pattern in f"{model_id} {description}" for pattern in exclude_patterns):
            return False
        
        return True
    
    def supports_tool_use(self, model: Dict) -> bool:
        """Check if model supports tool use/function calling."""
        # Check supported_parameters field for tool support
        supported_parameters = model.get("supported_parameters", [])
        if "tools" in supported_parameters or "tool_choice" in supported_parameters:
            return True
        
        # Fallback: Check for explicit tool use support in description
        model_id = model.get("id", "").lower()
        description = model.get("description", "").lower()
        
        tool_keywords = ["tool", "function", "function calling", "tool calling", "tools"]
        if any(keyword in description for keyword in tool_keywords):
            return True
        

        
        return False
    
    def get_filtered_models(self, limit: int = 20) -> Optional[List[Dict]]:
        """Get filtered coding/reasoning models from OpenRouter API."""
        models = self.fetch_models()
        if not models:
            return None
        
        filtered = []
        for model in models:
            model_id = model.get("id", "")
            pricing = model.get("pricing", {})
            
            # Skip unwanted models
            if (not pricing or 
                model_id.startswith("openrouter/") or
                ":free" in model_id or
                self.is_byok_model(model) or
                not self.is_coding_model(model) or
                not self.supports_tool_use(model)):
                continue
            
            try:
                prompt_cost = float(pricing.get("prompt", "0"))
                completion_cost = float(pricing.get("completion", "0"))
                
                if prompt_cost == 0 and completion_cost == 0:
                    continue
                
                # Format model data
                provider = model_id.split("/")[0] if "/" in model_id else "unknown"
                model_data = {
                    "id": model_id,
                    "provider": provider,
                    "prompt_cost_per_million": prompt_cost * 1_000_000,
                    "completion_cost_per_million": completion_cost * 1_000_000,
                    "description": model.get("description", ""),
                    "context_length": model.get("context_length", 0),
                    "supports_tools": True  # All models that pass filtering support tools
                }
                filtered.append(model_data)
                    
            except (ValueError, TypeError):
                continue
        
        return filtered[:limit]

    def display_models(self, models: List[Dict] = None) -> None:
        """Ovveriding method in Provider class. Display OpenRouter models in a unified table format."""
        if models is None:
            models = self.get_filtered_models()
        
        # Group by provider for ordering
        providers = {}
        for model in models:
            provider = model.get("provider", "unknown")
            if provider not in providers:
                providers[provider] = []
            providers[provider].append(model)
        
        # Summary
        summary = Table(show_header=False, box=box.SIMPLE, padding=(0, 1))
        summary.add_column("", style="bold blue", width=70)
        summary.add_row("Coding & Reasoning AI Models with Tool Support")
        summary.add_row(f"{len(models)} models from {len(providers)} companies")
        self.console.print(summary)
        self.console.print()
        
        # Main table
        table = Table(show_header=True, header_style="bold blue", expand=False, box=box.ROUNDED, padding=(0, 1))
        table.add_column("#", style="dim", width=4, justify="center")
        table.add_column("Company", style="green", no_wrap=True, width=12)
        table.add_column("Model", style="cyan", no_wrap=True, width=35)
        table.add_column("Prompt $/M", justify="right", style="yellow", width=12)
        table.add_column("Output $/M", justify="right", style="yellow", width=12)
        table.add_column("Context", justify="right", style="magenta", width=10)
        table.add_column("Tools", justify="center", style="blue", width=8)
        
        # Provider display names
        provider_names = {
            "openai": "OpenAI", "anthropic": "Anthropic", "google": "Google",
            "meta-llama": "Meta", "deepseek": "DeepSeek", "mistralai": "Mistral AI",
            "qwen": "Qwen", "cohere": "Cohere"
        }
        
        global_index = 1
        for provider, provider_models in providers.items():
            provider_display = provider_names.get(provider, provider.title())
            
            for model in provider_models:
                # Clean model name
                clean_name = model["id"].split("/", 1)[1] if "/" in model["id"] else model["id"]
                
                # Format context
                context = f"{model['context_length']//1000}K" if model['context_length'] >= 1000 else str(model['context_length'])
                
                row = [
                    str(global_index),
                    provider_display,
                    clean_name,
                    f"${model['prompt_cost_per_million']:.2f}",
                    f"${model['completion_cost_per_million']:.2f}",
                    context,
                    "Yes" if model.get("supports_tools", False) else "No"
                ]
                
                table.add_row(*row)
                global_index += 1
        
        self.console.print(table)
    
    def interactive_model_selection(self, limit: int = 20) -> Optional[str]:
        """Ovveriding method in Provider class. Interactive model selection for OpenRouter models."""
        models = self.get_filtered_models(limit)
        
        if not models:
            self.console.print("Could not fetch models")
            return None
        
        self.display_models(models)
        
        choice = get_user_input_for_int(
          prompt_text=f"Select model (1-{len(models)}) or Enter for first",
          default=1,
          valid_options=list(range(1, len(models) + 1))
        )

        if choice:
            return models[choice - 1].get("id")
        
        return None