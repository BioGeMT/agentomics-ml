import os
import json

from pydantic_ai import Agent, ModelRetry
import weave
import pandas as pd
from pathlib import Path
from pydantic_ai.messages import ToolCallPart

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
        system_prompt=get_system_prompt(config), # Passed only to first step when message history empty
        tools=tools,
        model_settings={'temperature': config.temperature},
        retries=config.max_validation_retries,
    )

    split_dataset_agent = Agent(
        model=model,
        tools=tools,
        model_settings={'temperature': config.temperature},
        output_type=DataSplit,
        retries=config.max_validation_retries,
    )

    training_agent = Agent(
        model=model,
        tools=tools,
        model_settings={'temperature': config.temperature},
        output_type=ModelTraining,
        retries=config.max_validation_retries,
    )

    inference_agent = Agent(
        model=model,
        tools=tools,
        model_settings={'temperature':config.temperature},
        output_type= ModelInference,
        retries=config.max_validation_retries,
    )
    @split_dataset_agent.output_validator
    async def validate_split_dataset(result: DataSplit) -> DataSplit:
        if not os.path.exists(result.train_path) or not os.path.exists(result.val_path):
            raise ModelRetry("Split dataset files do not exist.")
        
        train_path = Path(result.train_path)
        val_path = Path(result.val_path)
        if(train_path.name != 'train.csv' or val_path.name != 'validation.csv'):
            # Care to not delete original training or validation data
            original_train_csv_path = config.agent_dataset_dir / "train.csv"
            original_valid_csv_path = config.agent_dataset_dir / "validation.csv"
            if (train_path.resolve() != original_train_csv_path.resolve() and train_path.resolve() != original_valid_csv_path.resolve()):
                train_path.unlink()
            if (val_path.resolve() != original_train_csv_path.resolve() and val_path.resolve() != original_valid_csv_path.resolve()):
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


def get_final_result_messages(all_messages):
    final_result_response = all_messages[-2]
    final_result_tool_output_msg = all_messages[-1]
    assert any([isinstance(part, ToolCallPart) and part.tool_name=='final_result' for part in final_result_response.parts]) #TODO delete or move to tests
    return [final_result_response,final_result_tool_output_msg]

def fabricate_final_result_messages(structured_output, model_name):
    from pydantic_ai.messages import ModelResponse, ToolCallPart, ToolReturnPart, ModelRequest, TextPart
    import datetime
    output_dict = vars(structured_output)
    # output_dict['note'] = 'same as last iteration'
    tool_call_id = "pyd_ai_a6dd842e1f254405a457acaeae7afa4c" 
    response_msg = ModelResponse(
        parts=[
            TextPart(content='', part_kind='text'),
            ToolCallPart(tool_name="final_result", args = output_dict, tool_call_id=tool_call_id, part_kind='tool-call'),
        ],
        timestamp=datetime.datetime.now(),
        kind='response', 
        model_name=model_name,
    )
    request_msg = ModelRequest(
        parts=[
            ToolReturnPart(tool_name="final_result", content="Final result processed.", tool_call_id=tool_call_id, 
                           part_kind='tool-return', timestamp=datetime.datetime.now())
        ],
        kind='request',
    )
    return [response_msg, request_msg]

def get_sytem_and_user_prompt_messages(all_messages, to_remove):
    first_message = all_messages[0]
    assert any([part.part_kind=='system-prompt' for part in first_message.parts]) #TODO delete or move to tests
    assert any([part.part_kind=='user-prompt' for part in first_message.parts]) #TODO delete or move to tests
    user_prompt_part = [part for part in first_message.parts if part.part_kind=='user-prompt'][0]
    user_prompt_part.content = user_prompt_part.content.replace(to_remove, "") #Remove a non-global part of the prompt
    return [first_message]

async def run_architecture_compressed(text_output_agent: Agent, inference_agent: Agent, split_dataset_agent: Agent, training_agent: Agent, config: Config, base_prompt: str, iteration: int, last_split_strategy: str):
    persistent_messages = []
    structured_outputs = []
    ctx_replacer_msg = "\nSummarized outputs from your previous steps are in previous messages."

    messages_data_exploration, data_exploration_output = await run_agent(
        agent=text_output_agent,
        user_prompt=base_prompt + get_data_exploration_prompt(iteration), #base prompt has feedback (if non-0 iter) and user prompt
        max_steps=config.max_steps,
        output_type=DataExploration, # this is overriding the output_type
        message_history=None,
    )
    persistent_messages+=get_sytem_and_user_prompt_messages(messages_data_exploration, to_remove=get_data_exploration_prompt(iteration))
    persistent_messages+=get_final_result_messages(messages_data_exploration)
    structured_outputs.append(data_exploration_output)
    save_step_output(config, 'data_exploration', data_exploration_output, iteration)
    
    split_allowed_iterations = config.split_allowed_iterations
    if not config.explicit_valid_set_provided and iteration < split_allowed_iterations:
        messages_split, data_split = await run_agent(
            agent=split_dataset_agent,
            user_prompt=get_data_split_prompt(config=config, iteration=iteration, last_split_strategy=last_split_strategy)+ctx_replacer_msg,
            max_steps=config.max_steps,
            message_history=persistent_messages,
        )
        persistent_messages+=get_final_result_messages(messages_split)
        structured_outputs.append(data_split)
        save_step_output(config, 'data_split', data_split, iteration)
    else:
        assert last_split_strategy is not None, f'Agent didnt have a chance to split data, provide a non-0 allowed split iterations (currently {config.split_allowed_iterations})'
        manual_data_split_step = DataSplit(
            train_path = str(config.runs_dir / config.agent_id / 'train.csv'),
            val_path = str(config.runs_dir / config.agent_id / 'validation.csv'),
            splitting_strategy = last_split_strategy,
        )
        persistent_messages+=fabricate_final_result_messages(manual_data_split_step, model_name=config.model_name)
        structured_outputs.append(manual_data_split_step)
        #TODO Save step output for the report? saving can be done outside now that we return structured outputs

    messages_representation, data_representation = await run_agent(
        agent=text_output_agent,
        user_prompt=get_data_representation_prompt()+ctx_replacer_msg,
        max_steps=config.max_steps,
        output_type=DataRepresentation, # this is overriding the output_type
        message_history=persistent_messages,
    )
    persistent_messages+=get_final_result_messages(messages_representation)
    structured_outputs.append(data_representation)
    save_step_output(config, 'data_representation', data_representation, iteration)

    messages_architecture, model_architecture = await run_agent(
        agent=text_output_agent,
        user_prompt=get_model_architecture_prompt()+ctx_replacer_msg,
        max_steps=config.max_steps,
        output_type=ModelArchitecture, # this is overriding the output_type
        message_history=persistent_messages,
    )
    persistent_messages+=get_final_result_messages(messages_architecture)
    structured_outputs.append(model_architecture)
    save_step_output(config, 'model_architecture', model_architecture, iteration)

    messages_training, model_training = await run_agent(
        agent=training_agent, 
        user_prompt=get_model_training_prompt()+ctx_replacer_msg, 
        max_steps=config.max_steps,
        message_history=persistent_messages,
    )
    persistent_messages+=get_final_result_messages(messages_training)
    structured_outputs.append(model_training)
    save_step_output(config, 'model_training', model_training, iteration)

    messages_inference, model_inference = await run_agent(
        agent=inference_agent, 
        user_prompt=get_model_inference_prompt(config)+ctx_replacer_msg, 
        max_steps=config.max_steps,
        message_history=persistent_messages,
    )
    persistent_messages+=get_final_result_messages(messages_inference)
    structured_outputs.append(model_inference)
    save_step_output(config, 'model_inference', model_inference, iteration)

    if not config.explicit_valid_set_provided:
        val_path = config.runs_dir / config.agent_id / 'validation.csv'
    else:
        val_path = config.agent_dataset_dir / config.dataset / "validation.csv"

    prediction_messages, prediction_exploration = await run_agent(
        agent=text_output_agent,
        user_prompt=get_prediction_exploration_prompt(validation_path=val_path,inference_path=model_inference.path_to_inference_file)+ctx_replacer_msg,
        max_steps=config.max_steps,
        output_type=PredictionExploration,
        message_history=persistent_messages,
    )
    persistent_messages+=get_final_result_messages(prediction_messages) #not used
    structured_outputs.append(prediction_exploration)
    save_step_output(config, 'prediction_exploration', prediction_exploration, iteration)

    #TODO messages for feedback -> should be compressed or whole context?

    return structured_outputs

@weave.op(call_display_name=lambda call: f"Iteration {call.inputs.get('iteration', 0) + 1}")
async def run_iteration(config: Config, model, iteration, summary, feedback, tools, last_split_strategy):
    agents_dict = create_agents(config=config, model=model, tools=tools)

    if iteration == 0:
        base_prompt = get_user_prompt(config)
    else:
        base_prompt = get_iteration_prompt(config, iteration, summary, feedback)

    #TODO parametrize compressed vs normal runs
    structured_outputs = await run_architecture_compressed(
        text_output_agent=agents_dict["text_output_agent"],
        split_dataset_agent=agents_dict["split_dataset_agent"],
        training_agent=agents_dict["training_agent"],
        inference_agent=agents_dict["inference_agent"],
        config=config,
        base_prompt=base_prompt,
        iteration=iteration,
        last_split_strategy=last_split_strategy,
    )
    return structured_outputs