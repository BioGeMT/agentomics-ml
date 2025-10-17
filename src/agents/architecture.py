import os
import json

from pydantic_ai import Agent, ModelRetry
import weave
import pandas as pd
from pathlib import Path

from agents.agent_utils import run_agent
from agents.prompts.prompts_utils import get_iteration_prompt, get_user_prompt, get_system_prompt
from agents.steps.model_inference import ModelInference, get_model_inference_prompt
from agents.steps.data_split import DataSplit, get_data_split_prompt
from agents.steps.model_architecture import ModelArchitecture, get_model_architecture_prompt
from agents.steps.data_representation import DataRepresentation, get_data_representation_prompt
from agents.steps.data_exploration import DataExploration, get_data_exploration_prompt
from agents.steps.model_training import ModelTraining, get_model_training_prompt
from agents.steps.prediction_exploration import PredictionExploration, get_prediction_exploration_prompt
from utils.config import Config
from utils.report_logger import save_step_output
from run_logging.evaluate_log_run import run_inference_and_log

def create_agents(config: Config, model, tools):
    text_output_agent = Agent( # this is data exploration, representation, architecture reasoning, prediction exploration agent
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
        output_type= ModelInference,
        result_retries=config.max_validation_retries,
    )
    @split_dataset_agent.output_validator
    async def validate_split_dataset(result: DataSplit) -> DataSplit:
        if not os.path.exists(result.train_path) or not os.path.exists(result.val_path):
            raise ModelRetry("Split dataset files do not exist.")
        
        train_path = Path(result.train_path)
        val_path = Path(result.val_path)
        if(train_path.name != 'train.csv' or val_path.name != 'validation.csv'):
            # Care to not delete original training data
            original_train_csv_path = config.agent_dataset_dir / "train.csv"
            if (train_path.resolve() != original_train_csv_path.resolve()):
                train_path.unlink()
            if (val_path.resolve() != original_train_csv_path.resolve()):
                val_path.unlink()

            raise ModelRetry(f"The files must be called exactly 'train.csv' and 'validation.csv'. Files ({train_path.name} and {val_path.name}) have been deleted.")
        
        target_col = 'numeric_label' #TODO generalize and take from metadata.json or config
        for path in [result.train_path, result.val_path]:
            df = pd.read_csv(path)
            if target_col not in df.columns:
                raise ModelRetry(f"Target column {target_col} not found in dataset {path}. Columns found: {df.columns.tolist()}")
        return result
    
    @training_agent.output_validator
    async def validate_training(result: ModelTraining) -> ModelTraining:
        if not os.path.exists(result.path_to_train_file):
            raise ModelRetry("Train file does not exist.")
        if not os.path.exists(result.path_to_model_file):
            raise ModelRetry("Model file does not exist.")
        return result

    @inference_agent.output_validator
    async def validate_inference(result: ModelInference) -> ModelInference:
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
        user_prompt=base_prompt + get_data_exploration_prompt(iteration),
        max_steps=config.max_steps,
        output_type=DataExploration, # this is overriding the output_type
        message_history=None,
    )
    save_step_output(config, 'data_exploration', data_exploration_output, iteration)

    split_allowed_iterations = config.split_allowed_iterations
    if not config.explicit_valid_set_provided and iteration < split_allowed_iterations:
        messages_split, data_split = await run_agent(
            agent=split_dataset_agent,
            user_prompt=get_data_split_prompt(config, iteration),
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

    messages_inference, model_inference = await run_agent(
        agent=inference_agent, 
        user_prompt=get_model_inference_prompt(config), 
        max_steps=config.max_steps,
        message_history=messages_training,
    )
    save_step_output(config, 'model_inference', model_inference, iteration)

    if not config.explicit_valid_set_provided:
        val_path = config.runs_dir / config.agent_id / 'validation.csv'
    else:
        val_path = config.agent_dataset_dir / config.dataset / "validation.csv"

    _messages, prediction_exploration = await run_agent(
        agent=text_output_agent,
        user_prompt=get_prediction_exploration_prompt(validation_path=val_path,inference_path=model_inference.path_to_inference_file),
        max_steps=config.max_steps,
        output_type=PredictionExploration,
        message_history=messages_inference,
    )
    save_step_output(config, 'prediction_exploration', prediction_exploration, iteration)

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