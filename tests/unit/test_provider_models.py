from types import SimpleNamespace

import pytest
from pydantic_ai.models.openai import OpenAIResponsesModel
from pydantic_ai.models.openrouter import OpenRouterModel

from agentomics.utils.providers.anthropic_provider import AnthropicProvider
from agentomics.utils.providers.openai_provider import OpenAiProvider
from agentomics.utils.providers.openrouter_provider import OpenRouterProvider


def test_openai_model_uses_responses():
    config = SimpleNamespace(use_proxy=False, llm_response_timeout=30)

    model = OpenAiProvider("test-key", "https://openai.test/v1", "").create_model("gpt-5.4", config)

    assert isinstance(model, OpenAIResponsesModel)
    assert str(model.client.base_url) == "https://openai.test/v1/"


def test_openrouter_model_uses_openrouter_model_class():
    config = SimpleNamespace(use_proxy=False, llm_response_timeout=30)

    model = OpenRouterProvider("test-key", "https://openrouter.test/api/v1", "").create_model("openai/gpt-5.4", config)

    assert isinstance(model, OpenRouterModel)
    assert str(model.client.base_url) == "https://openrouter.test/api/v1/"


@pytest.mark.parametrize("provider_class", [AnthropicProvider, OpenRouterProvider])
def test_planning_settings_ask_for_high_reasoning(provider_class):
    provider = provider_class("test-key", "https://provider.test", "")

    settings = provider.get_high_reasoning_model_settings(kwargs={"temperature": 0.7})

    assert settings == {"thinking": "high", "temperature": 0.7}
