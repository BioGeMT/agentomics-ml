import traceback
import datetime
import re
import string
import random

from pydantic_ai import Agent, capture_run_messages
from pydantic_ai.usage import UsageLimits
from pydantic_ai.messages import ModelResponse, ToolCallPart, ToolReturnPart, ModelRequest, TextPart, ModelMessage
import weave

from agentomics.utils.exceptions import IterationRunFailed
from agentomics.utils.printing_utils import pretty_print_node

@weave.op(call_display_name=lambda call: f"Agent Step - {call.inputs['output_type'].__name__ if call.inputs.get('output_type', None) else call.inputs['agent']._output_type.__name__}")
async def run_agent(agent: Agent, user_prompt: str, max_steps: int, message_history: list | None, output_type = None, verbose: bool = True, deps=None):
    with capture_run_messages() as messages:
        try:
            async with agent.iter(
                user_prompt=user_prompt,
                usage_limits=UsageLimits(request_limit=max_steps),
                output_type=output_type,
                message_history=message_history,
                deps=deps,
            ) as agent_run:
                async for node in agent_run:
                    if(verbose):
                        pretty_print_node(node)
                return agent_run.result.all_messages(), agent_run.result.output

        except Exception as e:
            trace = traceback.format_exc()
            if(verbose):
                print('--------------- ERROR TRACEBACK ---------------')
                print('Agent run failed', trace)
                print('--------------- ERROR TRACEBACK ---------------')
            raise IterationRunFailed(
                message="Run didnt finish properly", 
                context_messages=messages,
                exception_trace=trace,
            )
        
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

def does_file_contain_string(file_path, search_string) -> bool:
    with open(file_path, 'r') as file:
        content = file.read()

    # the search_string must be withing a string in the python file (between ' or ") and start after the first quote symbol, doesnt match comments, variables, etc.
    pattern = rf"(['\"]){re.escape(search_string)}.*?\1"
    return re.search(pattern, content, re.DOTALL) is not None

def does_file_contain_iteration_pattern(file_path) -> bool:
    with open(file_path, 'r') as file:
        content = file.read()
    pattern = r"(['\"])iteration_\d+.*?\1"
    return re.search(pattern, content, re.DOTALL) is not None

def get_invalid_iteration_folders(config, iteration):
    run_dir = config.runs_dir / config.agent_id
    valid_folders = [f"iteration_{i}" for i in range(iteration)]
    invalid_folders = []
    for element in run_dir.iterdir():
        if "iteration_" in element.name and element.name not in valid_folders:
            invalid_folders.append(element.name)
    return invalid_folders

def get_final_result_messages(all_messages, structured_output=None, model_name=None):
    if any(isinstance(part, ToolCallPart) and part.tool_name=="final_result" for part in all_messages[-2].parts):
        return [all_messages[-2], all_messages[-1]]
    
    return fabricate_final_result_messages(structured_output, model_name) # step agent finishes with JSON output

def fabricate_final_result_messages(structured_output, model_name):
    output_dict = vars(structured_output)
    # output_dict['note'] = 'same as last iteration'
    tool_call_id = ''.join(random.choices(string.ascii_letters + string.digits, k=9))
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