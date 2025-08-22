import os
import httpx
from enum import Enum
import dotenv

from pydantic_ai.models.openai import OpenAIModel
from pydantic_ai.providers.openai import OpenAIProvider
from openai import AsyncOpenAI

class StrEnum(str, Enum):
    def __str__(self):
        return self.value
    
    def __repr__(self):
        return self.value

class MODELS(StrEnum):
    GPT4o="gpt-4o-2024-08-06"
    GPT4o_mini="gpt-4o-mini-2024-07-18"
    OPENROUTER_GPT4o="openrouter/openai/gpt-4"
    OPENROUTER_SONNET_37="anthropic/claude-3.7-sonnet"
    GPT4_1="openai/gpt-4.1-2025-04-14"
    GPT4_1_mini="gpt-4.1-mini-2025-04-14"
    GEMINI_2_5="google/gemini-2.5-pro-preview-03-25"
    GPT_O4_mini="openai/o4-mini"
    OLLAMA_GPT_OSS_20B="ollama/gpt-oss:20b"
    OLLAMA_DEVSTRAL_24B="ollama/devstral:24b"

    @staticmethod
    def is_local_model(model_name):
        return model_name.startswith("ollama/")
    
    @staticmethod
    def get_local_model_name(model_name):
        if MODELS.is_local_model(model_name):
            return model_name.replace("ollama/", "")
        raise Exception("The model in input is not local")
    
def create_model(model_name, config):
    dotenv.load_dotenv()

    if MODELS.is_local_model(model_name):
        ollama_model_name = MODELS.get_local_model_name(model_name)
        return OpenAIModel(
            model_name=ollama_model_name,
            provider=OpenAIProvider(base_url='http://localhost:11434/v1'), #base ollama endpoint
        )
    else:
        proxy_url = os.getenv("HTTP_PROXY")
        openrouter_api_key = os.getenv("OPENROUTER_API_KEY")

        async_http_client = httpx.AsyncClient(
            proxy=proxy_url if config.use_proxy else None,
            timeout= config.llm_response_timeout,
        )
        client = AsyncOpenAI(
            base_url='https://openrouter.ai/api/v1',
            api_key=openrouter_api_key,
            http_client=async_http_client,
        )

        return OpenAIModel(
            model_name=model_name,
            provider=OpenAIProvider(openai_client=client)
        )