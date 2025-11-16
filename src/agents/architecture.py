import os
import datetime

from pydantic_ai import Agent, ModelRetry, RunContext
import weave
import pandas as pd
from pathlib import Path
from pydantic_ai.messages import ModelResponse, ToolCallPart, ToolReturnPart, ModelRequest, TextPart, ModelMessage

from agents.agent_utils import run_agent
from agents.prompts.prompts_utils import get_iteration_prompt, get_system_prompt, get_iteration_0_prompt
from agents.steps.model_inference import ModelInference, get_model_inference_prompt, lock_inference_file
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
    data_exploration_agent = Agent(
        model=model,
        system_prompt=get_system_prompt(config), # Passed only to first step when message history empty
        tools=tools,
        model_settings={'temperature': config.temperature},
        output_type=DataExploration,
        retries=config.max_validation_retries,
        deps_type=dict,
    )
    split_dataset_agent = Agent(
        model=model,
        tools=tools,
        model_settings={'temperature': config.temperature},
        output_type=DataSplit,
        retries=config.max_validation_retries,
        deps_type=dict,
    )
    data_representation_agent = Agent(
        model=model,
        tools=tools,
        model_settings={'temperature': config.temperature},
        output_type=DataRepresentation,
        retries=config.max_validation_retries,
        deps_type=dict,
    )
    model_architecture_agent = Agent(
        model=model,
        tools=tools,
        model_settings={'temperature': config.temperature},
        output_type=ModelArchitecture,
        retries=config.max_validation_retries,
        deps_type=dict,
    )
    training_agent = Agent(
        model=model,
        tools=tools,
        model_settings={'temperature': config.temperature},
        output_type=ModelTraining,
        retries=config.max_validation_retries,
        deps_type=dict,
    )
    inference_agent = Agent(
        model=model,
        tools=tools,
        model_settings={'temperature':config.temperature},
        output_type= ModelInference,
        retries=config.max_validation_retries,
        deps_type=dict,
    )
    prediction_exploration_agent = Agent(
        model=model,
        tools=tools,
        model_settings={'temperature':config.temperature},
        output_type= PredictionExploration,
        retries=config.max_validation_retries,
        deps_type=dict,
    )

    @data_exploration_agent.output_validator
    async def validate_data_exploration(ctx: RunContext[dict], result):
        result.files_created = get_new_rundir_files(config, since_timestamp=ctx.deps['start_time'])
        return result
    @data_representation_agent.output_validator
    async def validate_data_representation(ctx: RunContext[dict], result):
        result.files_created = get_new_rundir_files(config, since_timestamp=ctx.deps['start_time'])
        return result
    @model_architecture_agent.output_validator
    async def validate_model_architecture(ctx: RunContext[dict], result):
        result.files_created = get_new_rundir_files(config, since_timestamp=ctx.deps['start_time'])
        return result

    @split_dataset_agent.output_validator
    async def validate_split_dataset(ctx: RunContext[dict], result: DataSplit) -> DataSplit:
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

        result.files_created = get_new_rundir_files(config, since_timestamp=ctx.deps['start_time'])
        return result
    
    @training_agent.output_validator
    async def validate_training(ctx: RunContext[dict], result: ModelTraining) -> ModelTraining:
        if not os.path.exists(result.path_to_train_file):
            raise ModelRetry(f"Train file does not exist. {result.path_to_train_file}")
        if not os.path.exists(result.path_to_model_file):
            raise ModelRetry(f"Model file does not exist at {result.path_to_model_file}")
        if does_file_contain_string(result.path_to_train_file, "iteration_"):
            raise ModelRetry("Train file contains path containing a forbidden string 'iteration_' or references an iteration folder, which will not accessible during final testing. If you want to re-use a file from a past iteration, copy it into the current working directory and use its path.")
        result.files_created = get_new_rundir_files(config, since_timestamp=ctx.deps['start_time'])
        return result

    @inference_agent.output_validator
    async def validate_inference(ctx: RunContext[dict], result: ModelInference) -> ModelInference:
        if not os.path.exists(result.path_to_inference_file):
            raise ModelRetry(f"Inference file does not exist at {result.path_to_inference_file}")
        if does_file_contain_string(result.path_to_inference_file, "iteration_"):
            raise ModelRetry("Inference file contains path containing a forbidden string 'iteration_' or references an iteration folder, which will not accessible during final testing. If you want to re-use a file from a past iteration, copy it into the current working directory and use its path.")
        if does_file_contain_string(result.path_to_inference_file, "train.csv") or does_file_contain_string(result.path_to_inference_file, "validation.csv"):
            raise ModelRetry("Inference file contains references to dataset split files ('train.csv' or 'validation.csv' detected), which will not be accessible during final testing.")
        run_inference_and_log(config, iteration=-1, evaluation_stage='dry_run')
        lock_inference_file(result.path_to_inference_file)
        result.files_created = get_new_rundir_files(config, since_timestamp=ctx.deps['start_time'])
        return result      
    
    @prediction_exploration_agent.output_validator
    async def validate_prediction_exploration(ctx: RunContext[dict], result: PredictionExploration) -> PredictionExploration:
        if not os.path.exists(config.runs_dir / config.agent_id / "inference.py"):
            raise ModelRetry(f"Inference file does not exist at {config.runs_dir / config.agent_id / 'inference.py'}")
        if does_file_contain_string(config.runs_dir / config.agent_id / "inference.py", "iteration_"):
            raise ModelRetry("Inference file contains references to an iteration folder ('iteration_' detected), which will not accessible during final testing. If you want to re-use a file from a past iteration, copy it into the current working directory and use its path.")
        invalid_iter_folders = get_invalid_iteration_folders(config, ctx.deps['iteration'])
        if len(invalid_iter_folders) > 0:
            raise ModelRetry("An iteration folder was created during this iteration. Move all files out of it to the current working directory, update their dependencies if necessary, and delete the folder. This applies to the following folders: " + ", ".join(invalid_iter_folders))
        run_inference_and_log(config, iteration=-1, evaluation_stage='dry_run')
        result.files_created = get_new_rundir_files(config, since_timestamp=ctx.deps['start_time'])
        return result

    return {
        "data_exploration_agent": data_exploration_agent,
        "split_dataset_agent": split_dataset_agent,
        "data_representation_agent": data_representation_agent,
        "model_architecture_agent": model_architecture_agent,
        "training_agent": training_agent,
        "inference_agent": inference_agent,
        "prediction_exploration_agent": prediction_exploration_agent,
    } 

def get_invalid_iteration_folders(config, iteration):
    run_dir = config.runs_dir / config.agent_id
    valid_folders = [f"iteration_{i}" for i in range(iteration)]
    invalid_folders = []
    for element in run_dir.iterdir():
        if "iteration_" in element.name and element.name not in valid_folders:
            invalid_folders.append(element.name)
    return invalid_folders

def does_file_contain_string(file_path, search_string) -> bool:
    with open(file_path, 'r') as file:
        content = file.read()
        return search_string in content

def get_final_result_messages(all_messages):
    final_result_response = all_messages[-2]
    final_result_tool_output_msg = all_messages[-1]
    assert any([isinstance(part, ToolCallPart) and part.tool_name=='final_result' for part in final_result_response.parts]) #TODO delete or move to tests
    return [final_result_response,final_result_tool_output_msg]

def fabricate_final_result_messages(structured_output, model_name):
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

def replace_message_result_with_validated_files(messages: list[ModelMessage], config, since_timestamp):
    for message in messages[-2:]: #only replace files_created in the last output messages
        for part in message.parts:
            if isinstance(part, ToolCallPart) and part.tool_name=="final_result":
                dict_args = part.args_as_dict()
                dict_args['files_created'] = get_new_rundir_files(config=config, since_timestamp=since_timestamp)
                part.args = dict_args

def get_sytem_and_user_prompt_messages(all_messages, to_remove):
    first_message = all_messages[0]
    assert any([part.part_kind=='system-prompt' for part in first_message.parts]) #TODO delete or move to tests
    assert any([part.part_kind=='user-prompt' for part in first_message.parts]) #TODO delete or move to tests
    user_prompt_part = [part for part in first_message.parts if part.part_kind=='user-prompt'][0]
    user_prompt_part.content = user_prompt_part.content.replace(to_remove, "") #Remove a non-global part of the prompt
    return [first_message]

def get_new_rundir_files(config, since_timestamp, ignore_iter_folders=True):
    run_dir = config.runs_dir / config.agent_id
    new_files = []
    for element in run_dir.iterdir():
        if ignore_iter_folders and "iteration_" in element.name and element.is_dir():
            continue
        #Check modified time
        if datetime.datetime.fromtimestamp(element.stat().st_mtime) > since_timestamp:
            new_files.append(element.name)
    return new_files

async def run_architecture_compressed(data_exploration_agent: Agent, data_representation_agent: Agent, model_architecture_agent: Agent, inference_agent: Agent, split_dataset_agent: Agent, training_agent: Agent, prediction_exploration_agent: Agent, config: Config, base_prompt: str, iteration: int, last_split_strategy: str):
    persistent_messages = []
    structured_outputs = []
    ctx_replacer_msg = "\nSummarized outputs from your previous steps are in previous messages."

    data_exploration_deps = {'start_time': datetime.datetime.now()}
    messages_data_exploration, data_exploration_output = await run_agent(
        agent=data_exploration_agent,
        user_prompt=base_prompt + get_data_exploration_prompt(iteration), #base prompt has feedback (if non-0 iter) and user prompt
        max_steps=config.max_steps,
        message_history=None,
        deps=data_exploration_deps,
    )
    replace_message_result_with_validated_files(messages_data_exploration, config, since_timestamp=data_exploration_deps['start_time'])
    persistent_messages+=get_sytem_and_user_prompt_messages(messages_data_exploration, to_remove=get_data_exploration_prompt(iteration))
    persistent_messages+=get_final_result_messages(messages_data_exploration)
    structured_outputs.append(data_exploration_output)
    
    split_allowed_iterations = config.split_allowed_iterations
    if not config.explicit_valid_set_provided and iteration < split_allowed_iterations:
        data_split_deps = {'start_time': datetime.datetime.now()}
        messages_split, data_split = await run_agent(
            agent=split_dataset_agent,
            user_prompt=get_data_split_prompt(config=config, iteration=iteration, last_split_strategy=last_split_strategy)+ctx_replacer_msg,
            max_steps=config.max_steps,
            message_history=persistent_messages,
            deps=data_split_deps,
        )
        replace_message_result_with_validated_files(messages_split, config, since_timestamp=data_split_deps['start_time'])
        persistent_messages+=get_final_result_messages(messages_split)
        structured_outputs.append(data_split)
    else:
        assert last_split_strategy is not None, f'Agent didnt have a chance to split data, provide a non-0 allowed split iterations (currently {config.split_allowed_iterations})'
        manual_data_split_step = DataSplit(
            train_path = str(config.runs_dir / config.agent_id / 'train.csv'),
            val_path = str(config.runs_dir / config.agent_id / 'validation.csv'),
            splitting_strategy = last_split_strategy,
            files_created=[],
        )
        persistent_messages+=fabricate_final_result_messages(manual_data_split_step, model_name=config.model_name)
        structured_outputs.append(manual_data_split_step)

    representation_deps = {'start_time': datetime.datetime.now()}
    messages_representation, data_representation = await run_agent(
        agent=data_representation_agent,
        user_prompt=get_data_representation_prompt()+ctx_replacer_msg,
        max_steps=config.max_steps,
        message_history=persistent_messages,
        deps=representation_deps,
    )
    replace_message_result_with_validated_files(messages_representation, config, since_timestamp=representation_deps['start_time'])
    persistent_messages+=get_final_result_messages(messages_representation)
    structured_outputs.append(data_representation)

    arch_deps = {'start_time': datetime.datetime.now()}
    messages_architecture, model_architecture = await run_agent(
        agent=model_architecture_agent,
        user_prompt=get_model_architecture_prompt()+ctx_replacer_msg,
        max_steps=config.max_steps,
        message_history=persistent_messages,
        deps=arch_deps,
    )
    replace_message_result_with_validated_files(messages_architecture, config, since_timestamp=arch_deps['start_time'])
    persistent_messages+=get_final_result_messages(messages_architecture)
    structured_outputs.append(model_architecture)

    training_deps = {'start_time': datetime.datetime.now()}
    messages_training, model_training = await run_agent(
        agent=training_agent, 
        user_prompt=get_model_training_prompt()+ctx_replacer_msg, 
        max_steps=config.max_steps,
        message_history=persistent_messages,
        deps=training_deps,
    )
    replace_message_result_with_validated_files(messages_training, config, since_timestamp=training_deps['start_time'])
    persistent_messages+=get_final_result_messages(messages_training)
    structured_outputs.append(model_training)

    inference_deps = {'start_time': datetime.datetime.now()}
    messages_inference, model_inference = await run_agent(
        agent=inference_agent, 
        user_prompt=get_model_inference_prompt(config)+ctx_replacer_msg, 
        max_steps=config.max_steps,
        message_history=persistent_messages,
        deps=inference_deps,
    )
    replace_message_result_with_validated_files(messages_inference, config, since_timestamp=inference_deps['start_time'])
    persistent_messages+=get_final_result_messages(messages_inference)
    structured_outputs.append(model_inference)

    if not config.explicit_valid_set_provided:
        val_path = config.runs_dir / config.agent_id / 'validation.csv'
    else:
        val_path = config.agent_dataset_dir / config.dataset / "validation.csv"

    prediction_deps = {'iteration': iteration, 'start_time': datetime.datetime.now()}
    prediction_messages, prediction_exploration = await run_agent(
        agent=prediction_exploration_agent,
        user_prompt=get_prediction_exploration_prompt(validation_path=val_path,inference_path=model_inference.path_to_inference_file)+ctx_replacer_msg,
        max_steps=config.max_steps,
        message_history=persistent_messages,
        deps=prediction_deps,
    )
    replace_message_result_with_validated_files(prediction_messages, config, since_timestamp=prediction_deps['start_time'])
    persistent_messages+=get_final_result_messages(prediction_messages) #not used
    structured_outputs.append(prediction_exploration)

    for structured_output in structured_outputs:
        save_step_output(config, type(structured_output).__name__, structured_output, iteration)

    return structured_outputs

@weave.op(call_display_name=lambda call: f"Iteration {call.inputs.get('iteration', 0)}")
async def run_iteration(config: Config, model, iteration, feedback, tools, last_split_strategy):
    agents_dict = create_agents(config=config, model=model, tools=tools)

    if iteration == 0:
        base_prompt = get_iteration_0_prompt()
    else:
        base_prompt = get_iteration_prompt(config, iteration, feedback)

    #TODO parametrize compressed vs normal runs
    structured_outputs = await run_architecture_compressed(
        data_exploration_agent=agents_dict['data_exploration_agent'],
        data_representation_agent=agents_dict['data_representation_agent'],
        split_dataset_agent=agents_dict["split_dataset_agent"],
        model_architecture_agent=agents_dict['model_architecture_agent'],
        training_agent=agents_dict["training_agent"],
        inference_agent=agents_dict["inference_agent"],
        prediction_exploration_agent=agents_dict["prediction_exploration_agent"],
        config=config,
        base_prompt=base_prompt,
        iteration=iteration,
        last_split_strategy=last_split_strategy,
    )
    return structured_outputs