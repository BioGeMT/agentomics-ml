from pydantic_ai import Agent
import time
import weave
import traceback
from utils.exceptions import FeedbackAgentFailed
from pydantic import BaseModel, Field
from agents.steps.model_inference import ModelInference
from agents.steps.model_training import ModelTraining

class IterationSummary(BaseModel):
    data_exploration_summary: str = Field(
        description="""
        Summary of the data and the features explored
        """
    )
    data_split_summary: str = Field(
        description="""
        Summary of how was the data split.
        """
    )
    data_representation_summary: str = Field(
        description="""
        Summary of any used transformations, encodings, normalizations, features, ...
        """
    )
    model_architecture_summary: str = Field(
        description="""
        Summary of the machine learning model type and architecture.
        """
    )
    model_training_summary: str = Field(
        description="""
        Summary of the training process, including hyperparameters and optimizers.
        """
    )
    prediction_exploration_summary: str = Field(
        description="""
        Summary of the insights about the prediction biases.
        """
    )

def create_feedback_agent(model, config):
    feedback_agent = Agent(
        model=model,
        model_settings={'temperature': config.temperature},
        result_retries=config.max_validation_retries
    )
    
    return feedback_agent

async def get_iteration_summary(structured_outputs, model, config) -> IterationSummary:
    agent = create_feedback_agent(model, config)

    training_step_output = next((step for step in structured_outputs if isinstance(step, ModelTraining)), None)
    inference_step_output = next((step for step in structured_outputs if isinstance(step, ModelInference)), None)

    if(training_step_output):
        with open(training_step_output.path_to_train_file) as f:
            training_script_content = f.read()
    else:
        training_script_content = "No training script produced"

    if(inference_step_output):
        with open(inference_step_output.path_to_inference_file) as f:
            inference_script_content = f.read()
    else:
        inference_script_content = "No inference script produced"

    summary_prompt = f"""
    Summarize the current iteration steps.

    Step outputs:
    {structured_outputs}

    Contents of the training script:
    [START]
    {training_script_content}
    [END]

    Contents of the inference script:
    [START]
    {inference_script_content}
    [END]
    """
    try:
        summary = await agent.run(
            user_prompt=summary_prompt,
            output_type=IterationSummary,
        )
        return summary.output
    except Exception as e:
        trace = traceback.format_exc()
        print('--------------- ERROR TRACEBACK ---------------')
        print('Feedback agent failed', trace)
        print('--------------- ERROR TRACEBACK ---------------')
        raise FeedbackAgentFailed(
            message="Feedback didnt finish properly", 
            context_messages=[],
            exception_trace=trace,
        )

#TODO if agent fails -> we can construct feedback absed on context contains ALL messages from the failed step (long string of failures typically)
@weave.op(call_display_name="Get Feedback")
async def get_feedback(structured_outputs, config, new_metrics, best_metrics, is_new_best, model, iteration, iter_to_summary, iter_to_metrics, val_split_changed, extra_info="") -> str:
    if iteration == config.iterations - 1 : return "Last iteration, no feedback needed"
    
    agent = create_feedback_agent(model, config)
    past_iters_aggregation = aggregate_past_iterations(iter_to_summary=iter_to_summary, iter_to_metrics=iter_to_metrics)
    extra_info = f'Extra info for the current iteration: {extra_info}' if extra_info else ''

    if val_split_changed:
        extra_info+="""
        The train/validation split has been changed this iteration. 
        This renders older iterations' metrics non-comparable as they were measured on different splits. 
        Due to this, the older iterations are no longer concidered for best-iteration candidates.
        """

    #TODO enforce experimentation when stagnant? 
    #TODO balance exploration vs exploitation based on remaining iterations?
    #TODO Emphasize that old split metric should not be concidered much unless the split is better - should prioritize TEST metric and be representative/difficult

    feedback_prompt = f"""
    Previous iterations summaries:
    {past_iters_aggregation}

    Current iteration outputs:
    {structured_outputs}

    {extra_info}

    Best metrics (out of any iteration) so far: {best_metrics}.
    The current iteration resulted in the following metrics: {new_metrics} and is {'not' if not is_new_best else ''} the best iteration run so far. 
    The most important metric for this task is {config.val_metric}. 

    Main goal: Provide feedback to the current iteration on how to improve generalization to a future unseen test set by changing any of the steps.
    
    Your feedback can suggest anything from small changes up to completelly changing the strategy of a step.
    Use the information from previous iterations and their metrics to improve your suggestions.
    You may skip steps that don't need changed.
    {'If you choose data splitting needs change, never suggest cross-validations split or any other split that would result in more than two files (train.csv and validation.csv).' if not config.explicit_valid_set_provided else ''}
    You're providing feedback to another LLM, never offer that you will take any actions to fix or implement fixes yourself.
    """
    
    print("CONSTRUCTING ITERATION FEEDBACK...")
    try:
        feedback = await agent.run(
            user_prompt = feedback_prompt,
            output_type=None,
            message_history=None
        )
        time.sleep(3)
        return feedback.data
    except Exception as e:
        trace = traceback.format_exc()
        print('--------------- ERROR TRACEBACK ---------------')
        print('Feedback agent failed', trace)
        print('--------------- ERROR TRACEBACK ---------------')
        raise FeedbackAgentFailed(
            message="Feedback didnt finish properly", 
            context_messages=[],
            exception_trace=trace,
        )

def aggregate_past_iterations(iter_to_summary, iter_to_metrics, ignore_last=True):
    aggregation = ""
    num_of_iters = len(iter_to_summary)
    if(ignore_last):
        num_of_iters -=1 #Current iteration is already in the dicts, the feedback agent will get this info anyways
    for i in range(num_of_iters):
        summary = iter_to_summary[i]
        metrics = iter_to_metrics[i]
        aggregation += f"""
        Iteration {i}
        Summary:
        {summary}
        Metrics: 
        {metrics}

        """
    return aggregation