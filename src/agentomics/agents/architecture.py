import datetime

from pydantic_ai import Agent
import weave
from rich.console import Console

from agentomics.agents.prompts.prompts_utils import get_iteration_prompt, get_iteration_0_prompt
from agentomics.agents.steps.model_inference import get_model_inference_prompt, create_model_inference_agent
from agentomics.agents.steps.data_split import DataSplit, get_data_split_prompt, create_data_split_agent
from agentomics.agents.steps.model_architecture import get_model_architecture_prompt, create_model_architecture_agent
from agentomics.agents.steps.data_representation import get_data_representation_prompt, create_data_representation_agent
from agentomics.agents.steps.data_exploration import get_data_exploration_prompt, create_data_exploration_agent
from agentomics.agents.steps.model_training import get_model_training_prompt, create_model_training_agent
from agentomics.agents.steps.prediction_exploration import get_prediction_exploration_prompt, create_prediction_exploration_agent
from agentomics.agents.agent_utils import run_agent, get_final_result_messages, fabricate_final_result_messages, replace_message_result_with_validated_files, get_sytem_and_user_prompt_messages
from agentomics.utils.config import Config
from agentomics.utils.report_logger import save_step_output

console = Console()

def create_agents(config: Config, model, tools):
    return {
        "data_exploration_agent": create_data_exploration_agent(config, model, tools),
        "split_dataset_agent": create_data_split_agent(config, model, tools),
        "data_representation_agent": create_data_representation_agent(config, model, tools),
        "model_architecture_agent": create_model_architecture_agent(config, model, tools),
        "training_agent": create_model_training_agent(config, model, tools),
        "inference_agent": create_model_inference_agent(config, model, tools),
        "prediction_exploration_agent": create_prediction_exploration_agent(config, model, tools),
    }

async def run_architecture_compressed(data_exploration_agent: Agent, data_representation_agent: Agent, model_architecture_agent: Agent, inference_agent: Agent, split_dataset_agent: Agent, training_agent: Agent, prediction_exploration_agent: Agent, config: Config, base_prompt: str, iteration: int, last_split_strategy: str):
    persistent_messages = []
    structured_outputs = []
    ctx_replacer_msg = "\nSummarized outputs from your previous steps are in previous messages."

    data_exploration_deps = {'start_time': datetime.datetime.now()}
    console.print(f"[bold purple]ITERATION {iteration} | DATA EXPLORATION STEP[/bold purple]")
    messages_data_exploration, data_exploration_output = await run_agent(
        agent=data_exploration_agent,
        user_prompt=base_prompt + get_data_exploration_prompt(iteration), #base prompt has feedback (if non-0 iter) and user prompt
        max_steps=config.max_steps,
        message_history=None,
        deps=data_exploration_deps,
    )
    replace_message_result_with_validated_files(messages_data_exploration, config, since_timestamp=data_exploration_deps['start_time'])
    persistent_messages+=get_sytem_and_user_prompt_messages(messages_data_exploration, to_remove=get_data_exploration_prompt(iteration))
    persistent_messages+=get_final_result_messages(messages_data_exploration, data_exploration_output, config.model_name)
    structured_outputs.append(data_exploration_output)
    
    data_split_step = None
    console.print(f"[bold purple]ITERATION {iteration} | SPLITTING STEP[/bold purple]")
    if not config.explicit_valid_set_provided and config.can_iteration_split_now_cached(iteration=iteration):
        data_split_deps = {'start_time': datetime.datetime.now()}
        messages_split, data_split = await run_agent(
            agent=split_dataset_agent,
            user_prompt=get_data_split_prompt(config=config, iteration=iteration, last_split_strategy=last_split_strategy)+ctx_replacer_msg,
            max_steps=config.max_steps,
            message_history=persistent_messages,
            deps=data_split_deps,
        )
        replace_message_result_with_validated_files(messages_split, config, since_timestamp=data_split_deps['start_time'])
        persistent_messages+=get_final_result_messages(messages_split, data_split, config.model_name)
        data_split_step=data_split
        structured_outputs.append(data_split)
    else:
        if config.explicit_valid_set_provided:
            console.print("[bold yellow]Validation set provided by user — skipping data splitting step.[/bold yellow]")
            split_strategy = last_split_strategy or "provided"
            manual_data_split_step = DataSplit(
                train_path=str(config.agent_dataset_dir / "train.csv"),
                val_path=str(config.agent_dataset_dir / "validation.csv"),
                splitting_strategy=split_strategy,
                files_created=[],
            )
        else:
            assert last_split_strategy is not None, (
                f"Agent didnt have a chance to split data, provide a non-0 allowed split iterations "
                f"(currently {config.split_allowed_iterations})"
            )
            manual_data_split_step = DataSplit(
                train_path=str(config.runs_dir / config.agent_id / "train.csv"),
                val_path=str(config.runs_dir / config.agent_id / "validation.csv"),
                splitting_strategy=last_split_strategy,
                files_created=[],
            )
        persistent_messages+=fabricate_final_result_messages(manual_data_split_step, model_name=config.model_name)
        data_split_step=manual_data_split_step
        structured_outputs.append(manual_data_split_step)

    representation_deps = {'start_time': datetime.datetime.now()}
    console.print(f"[bold purple]ITERATION {iteration} | REPRESENTATION STEP[/bold purple]")
    messages_representation, data_representation = await run_agent(
        agent=data_representation_agent,
        user_prompt=get_data_representation_prompt()+ctx_replacer_msg,
        max_steps=config.max_steps,
        message_history=persistent_messages,
        deps=representation_deps,
    )
    replace_message_result_with_validated_files(messages_representation, config, since_timestamp=representation_deps['start_time'])
    persistent_messages+=get_final_result_messages(messages_representation, data_representation, config.model_name)
    structured_outputs.append(data_representation)

    arch_deps = {'start_time': datetime.datetime.now()}
    console.print(f"[bold purple]ITERATION {iteration} | ARCHITECTURE STEP[/bold purple]")
    messages_architecture, model_architecture = await run_agent(
        agent=model_architecture_agent,
        user_prompt=get_model_architecture_prompt()+ctx_replacer_msg,
        max_steps=config.max_steps,
        message_history=persistent_messages,
        deps=arch_deps,
    )
    replace_message_result_with_validated_files(messages_architecture, config, since_timestamp=arch_deps['start_time'])
    persistent_messages+=get_final_result_messages(messages_architecture, model_architecture, config.model_name)
    structured_outputs.append(model_architecture)

    training_deps = {'start_time': datetime.datetime.now(), 'train_csv_path':data_split_step.train_path, 'validation_csv_path':data_split_step.val_path, 'run_dir': config.runs_dir / config.agent_id}
    console.print(f"[bold purple]ITERATION {iteration} | TRAINING STEP[/bold purple]")
    messages_training, model_training = await run_agent(
        agent=training_agent, 
        user_prompt=get_model_training_prompt(config)+ctx_replacer_msg, 
        max_steps=config.max_steps,
        message_history=persistent_messages,
        deps=training_deps,
    )
    replace_message_result_with_validated_files(messages_training, config, since_timestamp=training_deps['start_time'])
    persistent_messages+=get_final_result_messages(messages_training, model_training, config.model_name)
    structured_outputs.append(model_training)

    inference_deps = {'start_time': datetime.datetime.now()}
    console.print(f"[bold purple]ITERATION {iteration} | INFERENCE STEP[/bold purple]")
    messages_inference, model_inference = await run_agent(
        agent=inference_agent, 
        user_prompt=get_model_inference_prompt(config, training_artifacts_dir=model_training.path_to_artifacts_dir)+ctx_replacer_msg, 
        max_steps=config.max_steps,
        message_history=persistent_messages,
        deps=inference_deps,
    )
    replace_message_result_with_validated_files(messages_inference, config, since_timestamp=inference_deps['start_time'])
    persistent_messages+=get_final_result_messages(messages_inference, model_inference, config.model_name)
    structured_outputs.append(model_inference)

    if not config.explicit_valid_set_provided:
        val_path = config.runs_dir / config.agent_id / 'validation.csv'
    else:
        val_path = config.agent_dataset_dir / "validation.csv"

    prediction_deps = {'iteration': iteration, 'start_time': datetime.datetime.now()}
    console.print(f"[bold purple]ITERATION {iteration} | PREDICTION EXPLORATION STEP[/bold purple]")
    prediction_messages, prediction_exploration = await run_agent(
        agent=prediction_exploration_agent,
        user_prompt=get_prediction_exploration_prompt(validation_path=val_path,inference_path=model_inference.path_to_inference_file)+ctx_replacer_msg,
        max_steps=config.max_steps,
        message_history=persistent_messages,
        deps=prediction_deps,
    )
    replace_message_result_with_validated_files(prediction_messages, config, since_timestamp=prediction_deps['start_time'])
    persistent_messages+=get_final_result_messages(prediction_messages, prediction_exploration, config.model_name)  # not used
    structured_outputs.append(prediction_exploration)

    for structured_output in structured_outputs:
        save_step_output(config, type(structured_output).__name__, structured_output, iteration)

    return structured_outputs

@weave.op(call_display_name=lambda call: f"Iteration {call.inputs.get('iteration', 0)}")
async def run_iteration(config: Config, model, iteration, feedback, tools, last_split_strategy):
    agents_dict = create_agents(config=config, model=model, tools=tools)

    if iteration == 0:
        base_prompt = get_iteration_0_prompt(config)
    else:
        base_prompt = get_iteration_prompt(config, iteration, feedback)

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