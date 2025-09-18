import yaml
from typing import Dict, List, Optional
from pathlib import Path
import os
import httpx
import requests

from pydantic_ai.models.openai import OpenAIModel
from pydantic_ai.providers.openai import OpenAIProvider
from openai import AsyncOpenAI
from rich.console import Console
from rich.table import Table
from rich import box

from utils.config import Config
from utils.user_input import get_user_input_for_int


class Provider():
    """Parent provider class. All providers must extend this class."""
    console = Console()
    
    def __init__(self, name: str, base_url: str, api_key: Optional[str] = None, list_models_endpoint: Optional[str] = None):
        self.name = name
        self.api_key = api_key
        self.base_url = base_url
        self.list_models_endpoint = list_models_endpoint
        self.headers = {}
    
    def fetch_models(self) -> Optional[List[Dict]]:
        """Get available models. Override in subclasses when necessary."""
        try:
            response = requests.get(self.list_models_endpoint, headers=self.headers, timeout=10)
            response.raise_for_status()
            models = response.json().get("data", [])
            self.console.print(f"Fetched {len(models)} models from {self.name}")
            return models
        except Exception as e:
            self.console.print(f"Failed to fetch models from {self.name}: {e}")
            return None
    
    def display_models(self, models: List[Dict] = None) -> None:
        """Display models. Override in subclasses when necessary."""
        if models is None:
            models = self.fetch_models()
        
        if not models:
            print(f"No models available for {self.name}")
            return
        
        table = Table(title=f"Available {self.name} Models ({len(models)})", box=box.ROUNDED)
        table.add_column("#", style="cyan", no_wrap=True, width=4)
        table.add_column("Model Name", style="green")
        
        for i, model in enumerate(models, 1):
            table.add_row(str(i), model['id'])
        
        self.console.print(table)

        
    
    def interactive_model_selection(self, limit: int = None) -> Optional[str]:
        """Interactive model selection. Override in subclasses when necessary."""
        models = self.fetch_models()
        if not models:
            self.console.print(f"Could not fetch models from {self.name}", style="red")
            return None
        
        if limit and limit < len(models): models = models[:limit]

        self.display_models(models)
        
        choice = get_user_input_for_int(
          prompt_text=f"Select model (1-{len(models)}) or Enter for first",
          default=1,
          valid_options=list(range(1, len(models) + 1))
        )

        if choice:
            return models[choice - 1].get("id")
        
        return None
    
    def create_model(self, model_name: str, config: Config) -> OpenAIModel:
          """Create pydantic-ai model instance (OpenAI sdk friendly). Override in subclasses when necessary."""
          proxy_url = os.getenv("HTTP_PROXY")
          async_http_client = httpx.AsyncClient(
              proxy=proxy_url if config.use_proxy else None,
              timeout=config.llm_response_timeout
          )
          client = AsyncOpenAI(
              base_url=self.base_url,
              api_key=self.api_key,
              http_client=async_http_client,
          )
          return OpenAIModel(
              model_name=model_name,
              provider=OpenAIProvider(openai_client=client)
          )

    @staticmethod
    def get_provider_config(provider_name: str) -> Dict:
        yaml_path = Path(__file__).parent / "configured_providers.yaml"

        with open(yaml_path, 'r') as file:
            provider_config = yaml.safe_load(file)

            for provider in provider_config.get("providers", []):
                if provider.get("name").lower() == provider_name.lower():
                    return provider
        raise ValueError(f"Provider {provider_name} not available in {yaml_path} file")
    
    @staticmethod
    def get_available_providers() -> List[str]:
        yaml_path = Path(__file__).parent / "configured_providers.yaml"

        with open(yaml_path, 'r') as file:
            config = yaml.safe_load(file)

        providers = []
        for provider in config.get("providers", []):
            provider_name = provider.get("name")
            if provider_name:
                providers.append(provider_name)

        return providers

    @staticmethod
    def create_provider(provider_name: str, api_key: str):
        """Factory method to create providers."""
        provider_config = Provider.get_provider_config(provider_name)
        base_url = provider_config["base_url"]
        name_lower = provider_name.lower()
        
        if name_lower == "openrouter":
            from .openrouter_provider import OpenRouterProvider
            return OpenRouterProvider(api_key, base_url, provider_config["list_models_endpoint"])
        elif name_lower == "ollama":
            from .ollama_provider import OllamaProvider
            return OllamaProvider(base_url, provider_config["list_models_endpoint"])
        elif name_lower == "anthropic":
            from .anthropic_provider import AnthropicProvider
            return AnthropicProvider(api_key, base_url, provider_config["list_models_endpoint"])
        elif name_lower == "openai":
            from .openai_provider import OpenAiProvider
            return OpenAiProvider(api_key, base_url, provider_config["list_models_endpoint"])
        # elif name_lower == "googleai": requires pydanticai version update
        #     from .googleai_provider import GoogleAiProvider
        #     return GoogleAiProvider(api_key, base_url, list_models_endpoint)
        else: #user given provider, must be OpenAI compatible (https://ai.pydantic.dev/models/overview/#openai-compatible-providers)
            return Provider(provider_name, api_key, base_url)
        

def get_provided_api_keys() -> Dict[str, str]:
    """Retrive from the environment the provided API keys for all available providers."""
    provided_keys = {}
    available_providers = Provider.get_available_providers()

    for provider_name in available_providers:
        provider_config = Provider.get_provider_config(provider_name)
        api_key_env = provider_config.get("apikey", "").replace("${", "").replace("}", "")

        if (api_key_env and os.getenv(api_key_env)) or (provider_name.lower() == "ollama" and os.getenv("OLLAMA_BASE_URL")):
              provided_keys[provider_name] = os.getenv(api_key_env, "")

    return provided_keys #{"provider": api_key, ...}

def choose_provider(available_keys) -> int:
    """Prompt user to choose a provider when multiple API keys are given."""
    console = Console()
    table = Table(title="Available API Providers", box=box.ROUNDED)
    table.add_column("#", style="cyan", no_wrap=True)
    table.add_column("Provider", style="green")

    keys_list = list(available_keys.keys())
    for i, key in enumerate(keys_list, 1):
        table.add_row(str(i), key)

    console.print(table)
    prompt = "Multiple provider API keys found. Select the provider to use:"
    return get_user_input_for_int(prompt, default=1, valid_options=list(range(1, len(available_keys)+1)))

def get_provider_and_api_key() -> tuple[str, str]:
    """Get provider name and API key from environment variables. If multiple keys are found, prompt user to select one."""
    console = Console()

    api_keys_provided = get_provided_api_keys()
    if len(api_keys_provided) == 0:
        console.print("No provider API keys found. Please set at least one of the following environment variables:", style="red")
        console.print(list_required_api_keys(), style="cyan")
        raise ValueError("No API keys found in environment variables")
    elif len(api_keys_provided) > 1:
        selection = choose_provider(api_keys_provided)
        selected_provider_name = list(api_keys_provided.keys())[selection-1]

        api_key = api_keys_provided[selected_provider_name]
        provider = selected_provider_name
    else: # only one key provided
        provider = list(api_keys_provided.keys())[0]
        api_key = list(api_keys_provided.values())[0]

    return api_key, provider

def list_required_api_keys() -> str:
    """List required environment variables for all available providers."""
    available_providers = Provider.get_available_providers()
    required_env_vars = []

    for provider_name in available_providers:
        provider_config = Provider.get_provider_config(provider_name)
        api_key_env = provider_config.get("apikey", "").replace("${", "").replace("}", "")
        if api_key_env:  # Skip Ollama which has empty apikey
            required_env_vars.append(api_key_env)
        elif provider_name.lower() == "ollama":
            required_env_vars.append("OLLAMA_BASE_URL")
    
    return ", ".join(required_env_vars)