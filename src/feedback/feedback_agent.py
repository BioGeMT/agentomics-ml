from pydantic_ai import Agent
import time
import weave
import traceback
from utils.exceptions import FeedbackAgentFailed
from pydantic import BaseModel, Field
from agents.steps.model_inference import ModelInference
from agents.steps.model_training import ModelTraining
from utils.printing_utils import truncate_float
from utils.snapshots import get_best_iteration

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
        retries=config.max_validation_retries
    )
    
    return feedback_agent


@weave.op(call_display_name="Get Feedback")
async def get_feedback(config, is_new_best, model, iteration, iter_to_outputs, iter_to_metrics, iter_to_split_changed, val_split_changed, iter_to_duration, extra_info="") -> str:
    if iteration == config.iterations - 1 : return "Last iteration, no feedback needed"
    
    agent = create_feedback_agent(model, config)
    next_iteration_index = iteration + 1
    num_of_iters = len(iter_to_outputs)
    iter_to_split_version, split_version_to_iters = get_iter_split_infos(iter_to_outputs, iter_to_split_changed)
    lastest_split_version = iter_to_split_version[num_of_iters-1]
    best_metric_iteration = get_best_iteration(config)
    all_iters_aggregation = aggregate_past_iterations(
        iter_to_outputs=iter_to_outputs, 
        iter_to_metrics=iter_to_metrics, 
        current_iter_val_split_changed=val_split_changed, 
        current_iter_extra_info=extra_info, 
        current_iter_is_new_best=is_new_best, 
        iter_to_duration=iter_to_duration,
        iter_to_split_version=iter_to_split_version,
        split_version_to_iters=split_version_to_iters,
    )

    if config.can_iteration_split_data(next_iteration_index):
        #agent can split next iter
        splitting_info = f"If you choose data splitting needs change, never suggest cross-validations split or any other split that would result in more than two files (train.csv and validation.csv). Keep in mind that using a more representative validation split will result in a better selected 'best iteration model' and therefore a better final hidden test set metrics."
        splitting_info += f"If the split should stay the same (currently split version is {lastest_split_version}), instruct to 'Re-use the current split'."
        splitting_info += f"\nSplitting is allowed for the next {config.split_allowed_iterations - next_iteration_index} iteration/s. After that, the latest split version will be used for all future iterations."
    else:
        #agent can NOT split next iter
        splitting_info = "Instruct to skip the splitting step, as the next iteration agent cannot split the data."

    time_info = ""
    time_left_in_seconds = config.time_deadline - time.time() if config.time_deadline is not None else None
    if time_left_in_seconds is not None:
        time_info = f"{time_left_in_seconds / 3600:.2f} hours"
    else:
        iterations_left = config.iterations - iteration - 1
        time_info = f"{iterations_left} iterations"
        
    feedback_prompt = f"""
    {len(iter_to_outputs)} iterations completed in the current run so far
    [Iterations summaries (run history)]
    {all_iters_aggregation}
    [End of iteration summaries]

    The main goal of the run is to maximize the hidden test set generalization performance (main metric:{config.val_metric}) that will use the 'best iteration model' (currently model from iteration {best_metric_iteration}). Only models using the latest split are candidates for this 'best iteration model'.
    There are {time_info} left before the run ends. Then, the 'best iteration model' will be extracted and automatically evaluated on the hidden test set.
    Once an iteration produces a model with a better {config.val_metric} metric, that model will overwrite the 'best iteration model'.

    Your task is to provide concrete instructions for the next iteration agent on what to do/implement/pursue in each step of the next iteration.
    Use the information from previous iterations and their metrics to inform your instructions.
    Your instructions can suggest anything from reusing a step from any iteration, making small changes to it, up to completelly changing the strategy of a step.
    The iteration agent will always perform the steps in the same sequence.
    Make your instructions concrete, don't offer various choice branches. The exception to this is that you might offer branching instructions for a specific step conditioned on the results of the previous steps of that iteration.
    If you instruct the agent to re-use or partially re-use a step or code from a certain iteration, refer to that iteration by its number.
    The iteration agent will have access to the code and steps' outputs from all the past iterations.
    For example if you want to instruct to re-use the same data representation from iteration 5, instruct "Re-use the data representation from iteration 5".
    The data exploration and splitting steps can be instructed to be skipped completely if they're not needed.
    {splitting_info}

    You're providing instructions to an LLM agent, never offer that you will take any actions to fix or implement fixes yourself.
    Provide only actionable instructions, don't include "Why this helps", "Expected outcomes", or any other non-actionable information.
    If you refer to concrete files, use their name, extension, and the iteration they're from. Don't refer to them by their full path.

    Balance exploration of new approaches and optimizing already working approaches based on the iteration history and remaining time/iterations. 
    Remember that the goal is to maximize the hidden test set performance that will use the saved best model (currently iteration {best_metric_iteration}).
    This 'best iteration model' is saved and will not be overridden by a worse model, therefore you can safely instruct the agent to experiment with more exploratory models, representaiton etc... if you choose to.

    Once the next iteration finishes, the iterations summaries (run history) will be updated with its results and you will have an opportunity to provide another set of instructions etc.. until the run ends.
    """
    
    print("CONSTRUCTING ITERATION FEEDBACK...")
    try:
        feedback = await agent.run(
            user_prompt = feedback_prompt,
            output_type=None,
            message_history=None
        )
        time.sleep(3)
        return feedback.output
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

def get_iter_split_infos(iter_to_outputs, iter_to_split_changed):
    num_of_iters = len(iter_to_outputs)
    iter_to_split_version = {} #has the most recent iteration as well
    split_version = 0
    for i in range(num_of_iters):
        if(iter_to_split_changed[i]):
            split_version+=1
        iter_to_split_version[i] = split_version
    split_version_to_iters = {}
    for i, split_version in iter_to_split_version.items():
        split_version_to_iters[split_version] = split_version_to_iters.get(split_version, []) + [i]
    return iter_to_split_version, split_version_to_iters 

def aggregate_past_iterations(iter_to_outputs, iter_to_metrics, current_iter_val_split_changed, current_iter_extra_info, current_iter_is_new_best, iter_to_duration, iter_to_split_version, split_version_to_iters):
    num_of_iters = len(iter_to_outputs)
    lastest_split_version = iter_to_split_version[num_of_iters-1]

    aggregation = ""
    for i in range(num_of_iters):
        iter_split_version = iter_to_split_version[i]
        iters_with_the_same_split = split_version_to_iters[iter_split_version].copy()
        iters_with_the_same_split.remove(i)
        truncated_iter_metrics = {k:truncate_float(v) for k,v in iter_to_metrics[i].items()}
        if(iter_to_duration and i in iter_to_duration):
            iter_duration = f"{iter_to_duration[i]/3600:.2f} hours" 
        else:
            iter_duration = None

        split_info = f"This iteration used train/validation split strategy version {iter_split_version}. "
        if(len(iters_with_the_same_split) > 0):
            split_info+=f"This is the same as itertations {iters_with_the_same_split}. "
        if(iter_split_version != lastest_split_version):
            split_info += f"This iteration's split (version {iter_split_version}) is different from the latest iteration's split ({lastest_split_version}), therefore this iteration metrics can no longer be considered for a 'best iteration model' candidate. "
        else:
            split_info += f"This iteration's split (version {iter_split_version}) is the latest split version, therefore its metrics are considered for a 'best iteration model' candidate. "
        if(len(set(iter_to_split_version.values())) > 1):
            split_info += "Note that metrics calculated from different split versions are not directly comparable. "
        
        if i != num_of_iters -1: 
            aggregation += f"""
            Iteration {i} {f"(Duration: {iter_duration})" if iter_duration is not None else ""}
            Steps' outputs:
            {iter_to_outputs[i]}
            Metrics:
            {truncated_iter_metrics}
            {split_info}
            """
        else: #last iteration (current):
            extra_info = current_iter_extra_info
            if current_iter_val_split_changed:
                extra_info+="""
                The train/validation split has been changed this iteration. 
                This renders older iterations' metrics non-comparable as they were measured on different splits. 
                Due to this, the metrics of older iterations are no longer considered for 'best iteration model' candidates.
                """

            is_best_info = f"The current iteration is {'not ' if not current_iter_is_new_best else ''}the best iteration run so far{', therefore it is currently selected as the solution to your task' if current_iter_is_new_best else '.'}"
            aggregation += f"""
            Iteration {i} (Current iteration) {f"(Duration: {iter_duration})" if iter_duration is not None else ""}
            Steps' outputs:
            {iter_to_outputs[i]}
            {extra_info}
            Metrics:
            {truncated_iter_metrics}
            {is_best_info}
            {split_info}
            """

    return aggregation