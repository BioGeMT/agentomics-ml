import os
import json

from pydantic_ai import Agent, ModelRetry
import weave

from agents.agent_utils import run_agent
from agents.prompts.prompts_utils import get_iteration_prompt, get_user_prompt, get_system_prompt
from agents.steps.final_outcome import FinalOutcome, get_final_outcome_prompt
from agents.steps.data_split import DataSplit, get_data_split_prompt
from agents.steps.model_architecture import ModelArchitecture, get_model_architecture_prompt
from agents.steps.data_representation import DataRepresentation, get_data_representation_prompt
from agents.steps.data_exploration import DataExploration, get_data_exploration_prompt
from agents.steps.model_training import ModelTraining, get_model_training_prompt
from utils.config import Config
from utils.report_logger import save_step_output
from run_logging.evaluate_log_run import run_inference_and_log

def create_agents(config: Config, model, tools):
    text_output_agent = Agent( # this is data exploration, representation, architecture reasoning agent
        model=model,
        system_prompt=get_system_prompt(config),
        tools=tools,
        model_settings={'temperature': config.temperature},
        retries=config.max_run_retries,
        result_retries=config.max_validation_retries,
    )

    split_dataset_agent = Agent(
        model=model,
        tools=tools,
        model_settings={'temperature': config.temperature},
        output_type=DataSplit,
        result_retries=config.max_validation_retries,
    )

    training_agent = Agent(
        model=model,
        tools=tools,
        model_settings={'temperature': config.temperature},
        output_type=ModelTraining,
        result_retries=config.max_validation_retries,
    )

    inference_agent = Agent(
        model=model,
        tools=tools,
        model_settings={'temperature':config.temperature},
        output_type= FinalOutcome,
        result_retries=config.max_validation_retries,
    )
    @split_dataset_agent.output_validator
    async def validate_split_dataset(result: DataSplit) -> DataSplit:
        if not os.path.exists(result.train_path) or not os.path.exists(result.val_path):
            raise ModelRetry("Split dataset files do not exist.")
        return result
    
    @training_agent.output_validator
    async def validate_training(result: ModelTraining) -> ModelTraining:
        if not os.path.exists(result.path_to_train_file):
            raise ModelRetry("Train file does not exist.")
        if not os.path.exists(result.path_to_model_file):
            raise ModelRetry("Model file does not exist.")
        return result

    @inference_agent.output_validator
    async def validate_inference(result: FinalOutcome) -> FinalOutcome:
        if not os.path.exists(result.path_to_inference_file):
            raise ModelRetry("Inference file does not exist.")
        run_inference_and_log(config, iteration=-1, evaluation_stage='dry_run') 
        return result      

    return {
        "text_output_agent": text_output_agent,
        "split_dataset_agent": split_dataset_agent,
        "training_agent": training_agent,
        "inference_agent": inference_agent,
    } 


async def run_architecture(text_output_agent: Agent, inference_agent: Agent, split_dataset_agent: Agent, training_agent: Agent, config: Config, base_prompt: str, iteration: int):
    messages_data_exploration, data_exploration_output = await run_agent(
        agent=text_output_agent,
        user_prompt=base_prompt + get_data_exploration_prompt(),
        max_steps=config.max_steps,
        output_type=DataExploration, # this is overriding the output_type
        message_history=None,
    )
    save_step_output(config, 'data_exploration', data_exploration_output, iteration)

    if iteration == 0:
        messages_split, data_split = await run_agent(
            agent=split_dataset_agent,
            user_prompt=get_data_split_prompt(config),
            max_steps=config.max_steps,
            message_history=messages_data_exploration,
            )
        save_step_output(config, 'data_split', data_split, iteration)
    else:
        messages_split = messages_data_exploration

    messages_representation, data_representation = await run_agent(
        agent=text_output_agent,
        user_prompt=get_data_representation_prompt(),
        max_steps=config.max_steps,
        output_type=DataRepresentation, # this is overriding the output_type
        message_history=messages_split,
    )
    save_step_output(config, 'data_representation', data_representation, iteration)

    messages_architecture, model_architecture = await run_agent(
        agent=text_output_agent,
        user_prompt=get_model_architecture_prompt(),
        max_steps=config.max_steps,
        output_type=ModelArchitecture, # this is overriding the output_type
        message_history=messages_representation,
    )
    save_step_output(config, 'model_architecture', model_architecture, iteration)

    messages_training, model_training = await run_agent(
        agent=training_agent, 
        user_prompt=get_model_training_prompt(), 
        max_steps=config.max_steps,
        message_history=messages_architecture,
    )
    save_step_output(config, 'model_training', model_training, iteration)

    _messages, final_outcome = await run_agent(
        agent=inference_agent, 
        user_prompt=get_final_outcome_prompt(config), 
        max_steps=config.max_steps,
        message_history=messages_training,
    )
    save_step_output(config, 'final_outcome', final_outcome, iteration)

    return _messages

@weave.op(call_display_name=lambda call: f"Iteration {call.inputs.get('iteration', 0) + 1}")
async def run_iteration(config: Config, model, iteration, feedback, tools):
    agents_dict = create_agents(config=config, model=model, tools=tools)

    if iteration == 0:
        base_prompt = get_user_prompt(config)
    else:
        base_prompt = get_iteration_prompt(config, iteration, feedback)

    messages = await run_architecture(
        text_output_agent=agents_dict["text_output_agent"],
        split_dataset_agent=agents_dict["split_dataset_agent"],
        training_agent=agents_dict["training_agent"],
        inference_agent=agents_dict["inference_agent"],
        config=config,
        base_prompt=base_prompt,
        iteration=iteration,
    )
    return messages