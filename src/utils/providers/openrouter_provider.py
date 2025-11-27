from typing import Dict, List, Optional

from rich.table import Table
from rich import box
from rich.panel import Panel
from rich.columns import Columns
from rich.console import Group

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

        exclude_patterns = ["whisper", "dall-e", "tts", "embedding", "vision", "image", "audio", "research"]
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
        excluded_providers = ["nousresearch"]
        excluded_models = []
        filtered = []
        for model in models:
            model_id = model.get("id", "")
            pricing = model.get("pricing", {})
            if model_id in excluded_models:
                continue
            if any(f"{provider}/" in model_id for provider in excluded_providers):
                continue
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

    def display_models(self, models: List[Dict] = None) -> List[Dict]:
        if models is None:
            models = self.get_filtered_models()

        providers = {}
        for model in models:
            provider = model.get("provider", "unknown")
            if provider not in providers:
                providers[provider] = []
            providers[provider].append(model)

        self.console.print(f"[bold blue]Available Models[/bold blue] ({len(models)} models from {len(providers)} companies)\n")

        provider_names = {
            "openai": "OpenAI", "anthropic": "Anthropic", "google": "Google",
            "meta-llama": "Meta", "deepseek": "DeepSeek", "mistralai": "Mistral AI",
            "qwen": "Qwen", "cohere": "Cohere"
        }

        company_boxes = []

        for provider, provider_models in providers.items():
            provider_display = provider_names.get(provider, provider.title())
            box_height = len(provider_models) * 2

            model_data = []
            max_width = 0
            for model in provider_models:
                clean_name = model["id"].split("/", 1)[1] if "/" in model["id"] else model["id"]
                ctx_len = model['context_length']
                if ctx_len >= 1_000_000:
                    context = f"{ctx_len//1_000_000}M"
                elif ctx_len >= 1000:
                    context = f"{ctx_len//1000}K"
                else:
                    context = str(ctx_len)
                input_cost = f"${model['prompt_cost_per_million']:.1f}/M"
                output_cost = f"${model['completion_cost_per_million']:.1f}/M"

                cost_line_len = len(f"   Input: {input_cost}  Output: {output_cost}  Context: {context}")
                max_width = max(max_width, cost_line_len)

                model_data.append((clean_name, input_cost, output_cost, context))

            panel_width = min(max_width + 8, 60)
            company_boxes.append((box_height, panel_width, model_data, provider_display, provider_models))

        terminal_width = self.console.width
        avg_panel_width = sum(box[1] for box in company_boxes) / len(company_boxes) if company_boxes else 60
        padding_per_col = 2
        num_cols = max(1, min(4, int(terminal_width / (avg_panel_width + padding_per_col))))

        columns = [[] for _ in range(num_cols)]
        col_heights = [0] * num_cols
        col_max_widths = [0] * num_cols

        sorted_boxes = sorted(company_boxes, key=lambda x: x[0], reverse=True)
        model_columns = [[] for _ in range(num_cols)]
        for box_height, panel_width, model_data, provider_display, provider_models in sorted_boxes:
            min_col = col_heights.index(min(col_heights))
            columns[min_col].append((model_data, provider_display))
            model_columns[min_col].extend(provider_models)
            col_heights[min_col] += box_height
            col_max_widths[min_col] = max(col_max_widths[min_col], panel_width)
        display_order_models = [model for col in model_columns for model in col]

        
        col_renderables = []
        global_index = 1
        max_num_width = len(str(len(models)))
        for col_idx, col in enumerate(columns):
            panels = []
            for model_data, provider_display in col:
                lines = []
                for clean_name, input_cost, output_cost, context in model_data:
                    num_str = str(global_index).rjust(max_num_width)
                    lines.append(f"[dim]{num_str}.[/dim] [cyan]{clean_name}[/cyan]")
                    lines.append(f"   Input: [yellow]{input_cost}[/yellow]  Output: [yellow]{output_cost}[/yellow]  Context: [magenta]{context}[/magenta]")
                    global_index += 1
                panel = Panel("\n".join(lines), title=f"[bold green]{provider_display}[/bold green]",
                             title_align="left", border_style="green", width=col_max_widths[col_idx])
                panels.append(panel)
            col_renderables.append(Group(*panels))

        self.console.print(Columns(col_renderables, padding=(0, 1), expand=False))
        self.console.print("\n[dim]Prices per million tokens[/dim]")
        return display_order_models
    
    def interactive_model_selection(self, limit: int = 20) -> Optional[str]:
        """Ovveriding method in Provider class. Interactive model selection for OpenRouter models."""
        models = self.get_filtered_models(limit)
        
        if not models:
            self.console.print("Could not fetch models")
            return None
        
        display_order_models = self.display_models(models)
        
        choice = get_user_input_for_int(
          prompt_text=f"Select model (1-{len(display_order_models)}) or Enter for first",
          default=1,
          valid_options=list(range(1, len(display_order_models) + 1))
        )

        if choice:
            return display_order_models[choice - 1].get("id")
        
        return None