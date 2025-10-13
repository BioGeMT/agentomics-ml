from typing import List, Optional

from .provider import Provider
from utils.user_input import get_user_input_for_int

class OpenAiProvider(Provider):
    def __init__(self, api_key: str, base_url: str, list_models_endpoint: str):
        super().__init__(name="OpenAI", api_key=api_key, base_url=base_url, list_models_endpoint=list_models_endpoint)
        self.headers = {"Authorization": f"Bearer {api_key}"}
        
    def filter_models(self) -> List:
        """Fetch and filter models from OpenAI API."""
        models = self.fetch_models()
        if not models:
            return None
        # non-exhaustive list of patterns to exclude
        exclude_patterns = ["whisper", "dall-e", "tts", "embedding", "vision", "image", "audio", "moderation", "sora", "transcribe", "realtime"]
        
        filtered_models = []
        for model in models:
            model_id = model.get("id", "")
            if any(pattern in model_id.lower() for pattern in exclude_patterns):
                continue
            filtered_models.append(model)
    
        return filtered_models
    
    def interactive_model_selection(self, limit: int = None) -> Optional[str]:
        """Ovveriding method in Provider class. Interactive OpenAI model selection."""
        models = self.filter_models()
        # sort descending by first digit in model name
        models.sort(key=lambda m: int(''.join(filter(str.isdigit, m.get("id", "")))[:1]) if any(c.isdigit() for c in m.get("id", "")) else 0, reverse=True)

        if not models:
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