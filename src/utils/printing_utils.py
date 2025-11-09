import pprint
import textwrap
import json

from pydantic_ai.messages import SystemPromptPart, UserPromptPart, TextPart, ToolCallPart, ToolReturnPart, RetryPromptPart ,ThinkingPart
from pydantic_ai import CallToolsNode, ModelRequestNode, UserPromptNode
from pydantic_graph.nodes import End
from pygments import highlight
from pygments.lexers import PythonLexer
from pygments.formatters import TerminalTrueColorFormatter as TF
from pygments.formatters import TerminalFormatter as TF

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

def pretty_print_code(code):
    """
    Pretty print Python code with syntax highlighting and line numbers.
    args:
        code (str): The code string to pretty print.
    """
    if not code:
        return

    # normalize escapes and dedent
    code = code.replace('\\r\\n', '\n').replace('\\r', '\n')
    code = code.replace('\\t', '\t').replace('\\n', '\n')
    code = textwrap.dedent(code).strip('\n') + '\n'

    header = f"{bcolors.BOLD}{bcolors.OKCYAN}----- CODE START -----{bcolors.ENDC}"
    footer = f"{bcolors.BOLD}{bcolors.OKCYAN}-----  CODE END  -----{bcolors.ENDC}"
    print(header)

    try:
        # Try to use pygments for nice terminal highlighting
        # Prefer truecolor formatter if available, fall back to standard terminal formatter
        try:
            formatter = TF()
        except Exception:
            formatter = TF()

        highlighted = highlight(code, PythonLexer(), formatter)
        lines = highlighted.splitlines()
        width = len(str(len(lines)))
        for i, ln in enumerate(lines, 1):
            lineno = f"{bcolors.BOLD}{bcolors.OKBLUE}{str(i).rjust(width)}{bcolors.ENDC} "
            # ln already contains color codes from pygments
            print(f"{lineno}{ln}")
    except Exception:
        # Fallback: simple numbered output without external coloring
        lines = code.splitlines()
        width = len(str(len(lines)))
        for i, ln in enumerate(lines, 1):
            lineno = f"{bcolors.BOLD}{bcolors.OKBLUE}{str(i).rjust(width)}{bcolors.ENDC} "
            print(f"{lineno}{ln}")

    print(footer)

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
                if(part.tool_name=='write_python'):
                    args_dict = json.loads(str(part.args))
                    pretty_print_code(args_dict['code'])
                    pretty_print("file_path: "+args_dict['file_path'], color=bcolors.OKCYAN)
                    continue

                pretty_print(part.args, color=bcolors.OKCYAN)
            elif(isinstance(part, ThinkingPart)):
                pretty_print(part.content, color=bcolors.OKGREEN)
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

def truncate_float(f, decimals=3):
    factor = 10 ** decimals
    return int(f * factor) / factor