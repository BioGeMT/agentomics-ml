from types import SimpleNamespace

import pytest
from pydantic_ai.models import ModelRequestParameters

from agentomics.agents.steps.iteration_plan import IterationPlanStep
from agentomics.utils.providers.anthropic_provider import AnthropicProvider
from agentomics.utils.providers.codex_auth import CodexAuthStore
from agentomics.utils.providers.codex_provider import CodexProvider
from agentomics.utils.providers.openai_provider import OpenAiProvider
from agentomics.utils.providers.openrouter_provider import OpenRouterProvider
from agentomics.utils.providers.provider import Provider


@pytest.mark.parametrize(
    ("create_provider", "model_name"),
    [
        (lambda: OpenAiProvider("test-key", "https://openai.test/v1", ""), "gpt-5.4"),
        (CodexProvider, "gpt-6.1-sol"),
        (lambda: AnthropicProvider("test-key", "https://anthropic.test", ""), "claude-opus-4-7"),
        (lambda: OpenRouterProvider("test-key", "https://openrouter.test/api/v1", ""), "openai/gpt-5.4"),
        (lambda: Provider("custom", "https://custom.test/v1", "test-key"), "gpt-5.4"),
    ],
    ids=["openai", "codex", "anthropic", "openrouter", "custom"],
)
def test_planning_agent_asks_for_high_reasoning(monkeypatch, create_provider, model_name):
    monkeypatch.setattr(CodexAuthStore, "get_account_id", lambda self: "test-account")
    config = SimpleNamespace(use_proxy=False, llm_response_timeout=30, max_validation_retries=1, temperature=0.7)
    provider = create_provider()
    model = provider.create_model(model_name, config)

    agent = IterationPlanStep(config, model, model, provider, tools=[]).create_agent()
    _, parameters = model.prepare_request(agent.model_settings, ModelRequestParameters())

    assert parameters.thinking == "high"


def test_custom_provider_gets_its_key_and_base_url(monkeypatch):
    monkeypatch.setattr(Provider, "get_provider_config", staticmethod(lambda name: {"base_url": "https://custom.test/v1"}))

    provider = Provider.create_provider("Vercel", "test-key")

    assert (provider.base_url, provider.api_key) == ("https://custom.test/v1", "test-key")
