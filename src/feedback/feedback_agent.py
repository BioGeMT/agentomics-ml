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


#TODO if agent fails -> we can construct feedback absed on context contains ALL messages from the failed step (long string of failures typically)
@weave.op(call_display_name="Get Feedback")
async def get_feedback(structured_outputs, config, new_metrics, best_metrics, is_new_best, model, iteration, iter_to_outputs, iter_to_metrics, iter_to_split_changed, val_split_changed, iter_to_duration, extra_info="") -> str:
    if iteration == config.iterations - 1 : return "Last iteration, no feedback needed"
    
    agent = create_feedback_agent(model, config)
    next_iteration_index = iteration + 1
    all_iters_aggregation = aggregate_past_iterations(iter_to_outputs=iter_to_outputs, iter_to_metrics=iter_to_metrics, iter_to_split_changed=iter_to_split_changed, current_iter_val_split_changed=val_split_changed, current_iter_extra_info=extra_info, current_iter_is_new_best=is_new_best, iter_to_duration=iter_to_duration)
    best_metric_iteration = get_best_iteration(config)
    extra_info = f'Extra info for the current iteration: {extra_info}' if extra_info else ''
    best_metrics = {k:truncate_float(v) for k,v in best_metrics.items()}
    new_metrics = {k:truncate_float(v) for k,v in new_metrics.items()}

    if val_split_changed:
        extra_info+="""
        The train/validation split has been changed this iteration. 
        This renders older iterations' metrics non-comparable as they were measured on different splits. 
        Due to this, the metrics of older iterations are no longer considered for best-iteration candidates.
        """

    #TODO Emphasize that old split metric should not be considered much unless the split is better - should prioritize TEST metric and be representative/difficult
    #TODO never say youre doing well etc.. -> just try to optimize further???
    #TODO allow fallbacking to previous experiments?

    if config.can_iteration_split_data(next_iteration_index):
        #agent can split next iter
        splitting_info = 'If you choose data splitting needs change, never suggest cross-validations split or any other split that would result in more than two files (train.csv and validation.csv). Unless instructed to be changed, the current split will be re-used for the next iteration.'
        splitting_info += f"\nSplitting is allowed for the next {config.split_allowed_iterations - next_iteration_index} iteration/s. After that, the latest split will be used for all future iterations."
    else:
        #agent can NOT split next iter
        splitting_info = "Don't provide any feedback on how to change the train/validation split, as the next iteration agent cannot split the data."
    

    if best_metric_iteration is not None:
        best_metrics_info = f"Best iteration out of any previous iteration using the same data split is iteration {best_metric_iteration}. This is based on its {config.val_metric} metric. "
        if(not is_new_best):
            best_metrics_info += f"Iteration {best_metric_iteration} is still the best iteration, therefore it is currently selected as the solution to your task."
        best_metrics_info += f"\nMetrics from iteration {best_metric_iteration}: {best_metrics}."

    else:
        best_metrics_info = ""

    #TODO make feedback structured?
    time_info = ""
    time_left_in_seconds = config.time_deadline - time.time() if config.time_deadline is not None else None
    if time_left_in_seconds is not None:
        time_info = f"There are {time_left_in_seconds / 3600:.2f} hours left."
    else:
        iterations_left = config.iterations - iteration - 1
        time_info = f"There are {iterations_left} iterations left."
        
    #TODO dont refer to concrete files (paths have changed from rundir/file to rundir_iteration_x/file)
    #TODO dont call it suggestions/feedback but commands?
    #TODO adjust prompts if there's no best iteration yet
    feedback_prompt = f"""
    Iterations summaries:
    {all_iters_aggregation}

    MAIN GOAL
    Main goal: Provide concrete feedback for the next iteration to improve generalization on an unseen test set by changing one or more steps.
    
    COMPARING ITERATIONS INFO
    The most important metric for this task is {config.val_metric}. 
    Use the information from previous iterations and their metrics to guide your recommendations.

    GENERAL GUIDELINES [remove guidelines word]
    You're providing feedback to an LLM agent, never offer that you will take any actions to fix or implement fixes yourself.
    Provide only actionable feedback, don't include "Why this helps", "Expected outcomes", or any other non-actionable information.

    DIRECTIONS GUIDELINES
    {time_info}. After that, only the best iteration based on validation {config.val_metric} model using the latest split (currently iteration {best_metric_iteration}) will be judged using the final hidden test set.
    Balance exploration and exploitation based on the iteration history and remaining time/iterations. As the best model is saved, you can safely instruct the agent to experiment with more exploratory models, representaiton etc... if you decide to???
    [TODO] IF you hit a wall with a certain approach, feel free to explore...

    RECOMMENDATIONS GUIDELINES
    Your feedback can suggest anything from small changes up to completelly changing the strategy of a step.
    Make your recommendations concrete, don't offer various choice branches.
    If you instruct the agent to re-use a step from a certain iteration, refer to that iteration by its number.
    You may skip steps that don't need changed.
    [gpt] - Prefer practical, testable adjustments (data splits/quality, representation, model architecture, training hyperparams, evaluation).
    [TODO] You can refer other iterations by their number in feedback, the agent will have acess to them + summaries on demand
    {splitting_info} #TODO here?
    
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

def aggregate_past_iterations(iter_to_outputs, iter_to_metrics, iter_to_split_changed, current_iter_val_split_changed, current_iter_extra_info, current_iter_is_new_best, iter_to_duration):
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
            split_info += f"This iteration's split (version {iter_split_version}) is different from the latest iteration's split ({lastest_split_version}), therefore this iteration metrics can no longer be considered for a best-iteration candidate. "
        else:
            split_info += f"This iteration's split (version {iter_split_version}) is the latest split version, therefore its metrics are considered for a best-iteration candidate. "
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
                Due to this, the metrics of older iterations are no longer considered for best-iteration candidates.
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
            #TODO - [for each iter] this iteration took X seconds/minutes/hours.

    return aggregation