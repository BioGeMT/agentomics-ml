import requests
import os
import sys
from typing import Dict, List, Optional
from rich.console import Console
from rich.table import Table
from rich import box

console = Console()

class OpenRouterAPI:
    """OpenRouter API client for model fetching and filtering."""
    
    def __init__(self, api_key: str):
        self.api_key = api_key
        self.base_url = "https://openrouter.ai/api/v1/models"
        self.headers = {"Authorization": f"Bearer {api_key}"}
    
    def fetch_models(self) -> Optional[List[Dict]]:
        """Fetch all models from OpenRouter API."""
        try:
            response = requests.get(self.base_url, headers=self.headers, timeout=10)
            response.raise_for_status()
            models = response.json().get("data", [])
            console.print(f"Retrieved {len(models)} models from OpenRouter API")
            return models
        except Exception as e:
            console.print(f"Failed to fetch models: {e}")
            return None
    
    def is_byok_model(self, model: Dict) -> bool:
        """Check if model requires separate API key (BYOK)."""
        description = model.get("description", "").lower()
        model_id = model.get("id", "").lower()
        
        # Check for BYOK indicators
        byok_indicators = ["byok", "requires api key", "add your api key", "bring your own key"]
        if any(indicator in description for indicator in byok_indicators):
            console.print(f"Detected BYOK model: {model_id}")
            return True
        
        # Check for suspiciously low pricing
        pricing = model.get("pricing", {})
        try:
            prompt_cost = float(pricing.get("prompt", "0"))
            completion_cost = float(pricing.get("completion", "0"))
            if prompt_cost < 0.0000001 or completion_cost < 0.0000001:
                return True
        except (ValueError, TypeError):
            pass
        
        return False
    
    def is_coding_model(self, model: Dict) -> bool:
        """Check if model is suitable for coding/reasoning tasks."""
        model_id = model.get("id", "").lower()
        description = model.get("description", "").lower()
        
        # Coding/reasoning keywords
        coding_keywords = ["code", "coding", "instruct", "reasoning", "o1", "preview"]
        if any(keyword in f"{model_id} {description}" for keyword in coding_keywords):
            return True
        
        # Major coding models
        coding_models = ["gpt-4", "claude-3", "gemini", "llama-3", "deepseek", "qwen", "mistral"]
        if any(pattern in model_id for pattern in coding_models):
            return True
        
        # Exclude non-coding models
        exclude_patterns = ["whisper", "dall-e", "tts", "embedding", "vision", "image", "audio"]
        if any(pattern in f"{model_id} {description}" for pattern in exclude_patterns):
            return False
        
        return False
    
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
        """Get filtered coding/reasoning models."""
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
                
                # Skip suspiciously cheap models
                if prompt_cost < 0.000001:
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
        
        # Sort by popularity (provider priority + model quality)
        def popularity_score(model):
            provider = model["provider"]
            model_id = model["id"].lower()
            
            provider_priority = {
                "openai": 0, "anthropic": 1, "google": 2, "meta-llama": 3, 
                "deepseek": 4, "mistralai": 5, "qwen": 6
            }
            
            model_priority = 0
            if any(pattern in model_id for pattern in ["gpt-4o", "claude-3.5-sonnet", "o1-preview"]):
                model_priority = 0
            elif any(pattern in model_id for pattern in ["gpt-4-turbo", "claude-3-opus", "gemini-1.5-pro"]):
                model_priority = 1
            elif any(pattern in model_id for pattern in ["gpt-4", "claude-3-sonnet"]):
                model_priority = 2
            else:
                model_priority = 3
            
            return (provider_priority.get(provider, 99), model_priority)
        
        filtered.sort(key=popularity_score)
        return filtered[:limit]

def display_models(models: List[Dict]):
    """Display models in a unified table format."""
    if not models:
        console.print("No coding/reasoning models available")
        return
    
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
    console.print(summary)
    console.print()
    
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
            # Highlight flagship models
            style = {"style": "bold"} if any(pattern in model["id"].lower() 
                   for pattern in ["gpt-4o", "claude-3.5", "o1-preview", "gemini-1.5-pro"]) else {}
            
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
            
            table.add_row(*row, **style)
            global_index += 1
    
    console.print(table)

def get_model_choices(api_key: str, limit: int = 20) -> List[str]:
    """Get list of model IDs for selection."""
    api = OpenRouterAPI(api_key)
    models = api.get_filtered_models(limit)
    return [model["id"] for model in models] if models else []

def interactive_model_selection(api_key: str, limit: int = 20) -> Optional[str]:
    """Interactive model selection."""
    console.print("Fetching models from OpenRouter...")
    
    api = OpenRouterAPI(api_key)
    models = api.get_filtered_models(limit)
    
    if not models:
        console.print("Could not fetch models")
        return None
    
    display_models(models)
    console.print(f"\nSelect a model (1-{len(models)}) or press Enter for default:")
    
    try:
        choice = input("Enter choice: ").strip()
        if not choice:
            return models[0]["id"]
        
        choice_num = int(choice) - 1
        if 0 <= choice_num < len(models):
            selected = models[choice_num]["id"]
            console.print(f"Selected: {selected}")
            return selected
        else:
            console.print("Invalid choice")
            return None
    except (ValueError, KeyboardInterrupt):
        console.print("Invalid input")
        return None

# Weave Integration Functions
def setup_weave_costs(weave_client, model_id: str, api_key: str) -> bool:
    """Set up cost tracking in Weave for the specified model."""
    console.print(f"Setting up cost tracking for model: {model_id}")
    
    api = OpenRouterAPI(api_key)
    models = api.fetch_models()
    
    if not models:
        console.print(f"No cost information available for model: {model_id}")
        console.print(f"Cost tracking will be disabled for this session")
        return False
    
    # Find the model and get its costs
    for model in models:
        if model.get("id") == model_id:
            pricing = model.get("pricing", {})
            if pricing:
                try:
                    input_cost = float(pricing.get("prompt", "0"))
                    output_cost = float(pricing.get("completion", "0"))
                    
                    # Add costs to Weave
                    weave_client.add_cost(
                        llm_id=model_id,
                        prompt_token_cost=input_cost,
                        completion_token_cost=output_cost
                    )
                    
                    console.print(f"Cost tracking enabled:")
                    console.print(f"   • Model: {model_id}")
                    console.print(f"     - Prompt tokens: ${input_cost:.8f}/token")
                    console.print(f"     - Completion tokens: ${output_cost:.8f}/token")
                    
                    return True
                except (ValueError, TypeError) as e:
                    console.print(f"Invalid pricing format: {e}")
                    return False
    
    console.print(f"Model {model_id} not found")
    return False

def init_weave_with_costs(project_name: str, model_id: str, api_key: str):
    """Initialize Weave and set up cost tracking for the specified model."""
    import weave
    
    console.print(f"Initializing Weave for project: {project_name}")
    
    try:
        weave_client = weave.init(project_name)
        
        # Set up costs for the model
        setup_success = setup_weave_costs(weave_client, model_id, api_key)
        
        if setup_success:
            console.print(f"Weave integration ready with cost tracking")
        else:
            console.print(f"Weave initialized but cost tracking unavailable")
        
        return weave_client
        
    except Exception as e:
        console.print(f"Failed to initialize Weave: {e}")
        return None

# CLI Interface
if __name__ == "__main__":
    import argparse
    
    parser = argparse.ArgumentParser(description='OpenRouter API utilities')
    parser.add_argument('--limit', type=int, default=20, help='Number of models to display')
    parser.add_argument('--format', choices=['display', 'selection'], default='display',
                       help='Output format')
    args = parser.parse_args()
    
    api_key = os.getenv("OPENROUTER_API_KEY")
    if not api_key:
        console.print("OPENROUTER_API_KEY is required")
        console.print("Get your key from: https://openrouter.ai/keys")
        sys.exit(1)
    
    api = OpenRouterAPI(api_key)
    models = api.get_filtered_models(args.limit)
    
    if models:
        if args.format == 'display':
            display_models(models)
        else:  # selection format
            for i, model in enumerate(models, 1):
                tools_support = "1" if model.get("supports_tools", False) else "0"
                print(f'{i}|{model["id"]}|{model["prompt_cost_per_million"]:.3f}|{model["completion_cost_per_million"]:.3f}|{tools_support}')
    else:
        console.print("Failed to fetch models")
        sys.exit(1)
