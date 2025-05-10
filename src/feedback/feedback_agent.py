from pydantic_ai import Agent
from pydantic_ai.models.openai import OpenAIModel
from pydantic_ai.providers.openai import OpenAIProvider
import httpx
from openai import AsyncOpenAI
import dotenv
import os

def create_feedback_agent(model, config):
    feedback_agent = Agent(
        model=model,
        model_settings={'temperature': config['temperature']},
        result_retries=config["max_validation_retries"]
    )   
    
    return feedback_agent

async def get_feedback(context, config, new_metrics, best_metrics, is_new_best, api_key, extra_info="") -> str:
    dotenv.load_dotenv()
    proxy_url = os.getenv('PROXY_URL')
    async_http_client = httpx.AsyncClient(
            proxy=proxy_url if config["use_proxy"] else None,
            timeout= config["llm_response_timeout"],
        )
    client = AsyncOpenAI(
        base_url='https://openrouter.ai/api/v1',
        api_key=api_key,
        http_client=async_http_client,
    )
    model = OpenAIModel(
        config['feedback_model'],
        provider=OpenAIProvider(openai_client=client)
    )

    agent = create_feedback_agent(model, config)
    if is_new_best:
        prompt_suffix = "This is the best run so far. "
    else:
        prompt_suffix = "This is not the best run so far. "
    prompt_suffix += extra_info
    #TODO handle new or best metrics being empty dicts
    user_prompt = f"Summarize the current state of the run and provide feedback for the next iteration. Metrics from your current run are: {new_metrics}. Metrics from the past best run are: {best_metrics}. {prompt_suffix}"
    print('FEEDBACK AGENT PROMPT:', user_prompt)
    feedback = await agent.run(
        user_prompt = user_prompt,
        result_type=None,
        message_history=context #TODO remove system prompt from context?
    )

    return feedback.data