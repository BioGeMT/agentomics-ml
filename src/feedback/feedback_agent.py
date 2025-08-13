from pydantic_ai import Agent
from pydantic_ai.models.openai import OpenAIModel
from pydantic_ai.providers.openai import OpenAIProvider
import httpx
from openai import AsyncOpenAI
import dotenv
import os
import time
import weave

@weave.op(call_display_name="Create Feedback Agent")
def create_feedback_agent(model, config):
    feedback_agent = Agent(
        model=model,
        model_settings={'temperature': config.temperature},
        result_retries=config.max_validation_retries
    )
    
    return feedback_agent

@weave.op(call_display_name=lambda call: f"Get Feedback - Iteration {call.inputs.get('iteration', 0) + 1}")
async def get_feedback(context, config, new_metrics, best_metrics, is_new_best, api_key, iteration, aggregated_feedback=None, extra_info="") -> str:
    if iteration == config.iterations - 1 : return "Last iteration, no feedback needed"
    dotenv.load_dotenv()
    proxy_url = os.getenv('PROXY_URL')
    async_http_client = httpx.AsyncClient(
        proxy=proxy_url if config.use_proxy else None,
        timeout= config.llm_response_timeout,
    )
    client = AsyncOpenAI(
        base_url='https://openrouter.ai/api/v1',
        api_key=api_key,
        http_client=async_http_client,
    )
    model = OpenAIModel(
        config.feedback_model,
        provider=OpenAIProvider(openai_client=client)
    )

    agent = create_feedback_agent(model, config)
    
    if is_new_best:
        prompt_suffix = "This is the best run so far. "
    else:
        prompt_suffix = "This is not the best run so far. " 
    
    if aggregated_feedback:
        prompt_suffix += f"The aggregated feedback from the previous iterations is: {aggregated_feedback}."
    
    prompt_suffix += extra_info

    feedback_prompt = f"""
    Summarize the current state and provide detailed feedback on how to fix errors and improve the steps executed:
    1. Data exploration: describe the data and the features you explored.
    2. Data representation: any transformations, encodings, normalizations, features
    3. Model architecture: the machine learning model type and architecture for your task.
    4. Model training: the training process, including hyperparameters and optimizers.

    The current iteration returned the following metrics: {new_metrics}.
    Metrics from the past best run are: {best_metrics}.

    Provide insights on how to improve the model generalization performance.

    {prompt_suffix}.
    """
    
    feedback = await agent.run(
        user_prompt = feedback_prompt,
        output_type=None,
        message_history=context #TODO remove system prompt from context?
    )
    time.sleep(3)
    return feedback.data

@weave.op(call_display_name="Aggregate Historical Feedback")
def aggregate_feedback(feedback_list):
    if len(feedback_list) == 1: #TODO first iteration list contains None
        return None
    
    aggregated_feedback = ""
    
    filtered_list = [f for f in feedback_list if f is not None]

    for i, feedback in enumerate(filtered_list):
        if i == len(feedback_list) - 1: #if last feedback
            aggregated_feedback += f"Most recent iteration: Iteration {i}:\n{feedback}\n\n"
        else:
            aggregated_feedback += f"Past iterations: Iteration {i}:\n{feedback}\n\n"
    
    return aggregated_feedback
