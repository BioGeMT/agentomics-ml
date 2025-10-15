from pydantic_ai import Agent
import time
import weave

def create_feedback_agent(model, config):
    feedback_agent = Agent(
        model=model,
        model_settings={'temperature': config.temperature},
        result_retries=config.max_validation_retries
    )
    
    return feedback_agent

@weave.op(call_display_name="Get Feedback")
async def get_feedback(context, config, new_metrics, best_metrics, is_new_best, model, iteration, aggregated_feedback=None, extra_info="") -> str:
    if iteration == config.iterations - 1 : return "Last iteration, no feedback needed"
    
    agent = create_feedback_agent(model, config)
    
    if is_new_best:
        prompt_suffix = "This is the best run so far. "
    else:
        prompt_suffix = "This is not the best run so far. " 
    
    if aggregated_feedback:
        prompt_suffix += f"The aggregated feedback from the previous iterations is: {aggregated_feedback}."
    
    prompt_suffix += extra_info

    #TODO only give advice on splitting if val set is not provided explicitly
    feedback_prompt = f"""
    Summarize the current iteration steps:
    1. Data exploration: describe the data and the features explored.
    2. Data splitting: describe how was the data split.
    3. Data representation: any transformations, encodings, normalizations, features.
    4. Model architecture: the machine learning model type and architecture.
    5. Model training: the training process, including hyperparameters and optimizers.
    6. Prediction exploration: insights about the prediction biases.

    The current iteration returned the following metrics: {new_metrics}.
    Metrics from the past best run are: {best_metrics}.

    Provide feedback on how to fix errors and improve generalization to a future unseen test set by changing any of the steps.
    Provide only feedback that you expect to be impactful.
    Your feedback can suggest anything from small changes up to completelly changing the strategy of a step.
    You may skip steps that don't need changed.
    If you choose step 2 (Data splitting) needs change, never suggest cross-validations split or any other split that would result in more than two files (train.csv and validation.csv).
    You're providing feedback to another LLM, never offer that you will take any actions to fix or implement fixes yourself.
    
    {prompt_suffix}.
    """
    
    print("CONSTRUCTING ITERATION FEEDBACK...")
    feedback = await agent.run(
        user_prompt = feedback_prompt,
        output_type=None,
        message_history=context #TODO remove system prompt from context?
    )
    time.sleep(3)
    return feedback.data

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
