import pprint
import textwrap

from pydantic_ai.messages import SystemPromptPart, UserPromptPart, TextPart, ToolCallPart, ToolReturnPart, RetryPromptPart
from pydantic_ai import CallToolsNode, ModelRequestNode, UserPromptNode
from pydantic_graph.nodes import End

class bcolors:
    HEADER = '\033[95m'
    OKBLUE = '\033[94m'
    OKCYAN = '\033[96m'
    OKGREEN = '\033[92m'
    WARNING = '\033[93m'
    FAIL = '\033[91m'
    ENDC = '\033[0m'
    BOLD = '\033[1m'
    UNDERLINE = '\033[4m'
    ORANGE = '\033[38;5;208m'

def pretty_print(content, width=120, color=None):
    if color:
        content = f"{color}{content}{bcolors.ENDC}"
    if isinstance(content, str):
        print(textwrap.fill(content, width=width))
    else:
        pprint.pprint(content, width=width)

def pretty_print_node(node):
    if(isinstance(node, CallToolsNode)):
        # print(node.model_response.parts)
        for part in node.model_response.parts:
            # print(type(part).__name__)
            if(isinstance(part, TextPart)):
                pretty_print(part.content)
            elif(isinstance(part, ToolCallPart)):
                pretty_print(part.tool_name, color=bcolors.OKCYAN)
                if(part.tool_name == 'final_result'):
                    continue
                    #dont print args since it will be printed 
                pretty_print(part.args, color=bcolors.OKCYAN)
            else:
                pretty_print(f"DEVINFO: Unexpected part type (in CallToolsNode): {type(part)}")
                pretty_print(part)
            # print(part.content)
    elif(isinstance(node, ModelRequestNode)):
        # print(node.request.parts)
        for part in node.request.parts:
            # print(type(part).__name__)
            if(isinstance(part, SystemPromptPart)):
                pretty_print(part.content)
            elif(isinstance(part, UserPromptPart)):
                pretty_print(part.content)
            elif(isinstance(part, ToolReturnPart)):
                pretty_print(f"{part.tool_name} output")
                pretty_print(part.content)
            elif(isinstance(part, RetryPromptPart)):
                pretty_print("Output Validation Failed, Retry info:")
                pretty_print(part.content)
            else:
                pretty_print(f"DEVINFO: Unexpected part type (in ModelRequestNode): {type(part)}")
                pretty_print(part)
    elif(isinstance(node, UserPromptNode)):
        pass #duplicated
        # print(node.user_prompt) 
    elif(isinstance(node, End)):
        output = node.data.output
        #for each attribute in output object, print it
        for attr, value in output.__dict__.items():
            if(attr not in ['tool_name', 'tool_call_id']):
                pretty_print(attr, color=bcolors.BOLD+bcolors.ORANGE)
                pretty_print(value)
    else:
        pretty_print(f"DEVINFO: Unexpected node type: {type(node)}")
        pretty_print(node)