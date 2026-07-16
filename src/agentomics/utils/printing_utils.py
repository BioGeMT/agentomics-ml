import json
import os
import pprint
import textwrap

from pydantic_ai import CallToolsNode, ModelRequestNode, UserPromptNode
from pydantic_ai.messages import SystemPromptPart, UserPromptPart, TextPart, ToolCallPart, ToolReturnPart, RetryPromptPart, ThinkingPart
from pydantic_graph.nodes import End
from pygments import highlight
from pygments.formatters import TerminalFormatter as TF
from pygments.lexers import PythonLexer

from agentomics.utils.config import Config


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
    PRETTY_BLUE = '\033[38;5;33m'

def pretty_print(content, width=120, color=None):
    if color:
        content = f"{color}{content}{bcolors.ENDC}"
    if isinstance(content, str):
        print("\n".join(textwrap.fill(line, width=width) for line in content.splitlines()))
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

def _verbosity() -> str:
    mode = os.getenv("AGENTOMICS_VERBOSITY", "summary").strip().lower()
    return mode if mode in {"summary", "full"} else "summary"

def _args_dict(tool_call_part: ToolCallPart) -> dict:
    try:
        return tool_call_part.args_as_dict()
    except ValueError:
        return {}

def _tool_call_summary(part: ToolCallPart) -> str:
    tool = part.tool_name
    args = _args_dict(part)

    if tool == "bash":
        return f"{bcolors.OKCYAN}agent runs bash command{bcolors.ENDC}"

    if tool == "write_python":
        file_path = args.get("file_path")
        if file_path:
            return f"{bcolors.OKCYAN}agent writes python file{bcolors.ENDC} {os.path.basename(str(file_path))}"
        return f"{bcolors.OKCYAN}agent writes python file{bcolors.ENDC}"

    if tool == "run_python":
        python_file_path = args.get("python_file_path")
        if python_file_path:
            return f"{bcolors.OKCYAN}agent runs python file{bcolors.ENDC} {os.path.basename(str(python_file_path))}"
        return f"{bcolors.OKCYAN}agent runs python file{bcolors.ENDC}"

    if tool == "replace":
        file_path = args.get("file_path")
        if file_path:
            return f"{bcolors.OKCYAN}agent edits file{bcolors.ENDC} {os.path.basename(str(file_path))}"
        return f"{bcolors.OKCYAN}agent edits file{bcolors.ENDC}"

    return f"{bcolors.OKCYAN}agent calls tool {tool}{bcolors.ENDC}"

def _tool_return_summary(part: ToolReturnPart) -> tuple[str, bool]:
    tool = part.tool_name
    content_str = str(part.content or "")
    lower = content_str.lower()
    is_error = (
        "traceback" in lower
        or "error:" in lower
        or "exception" in lower
        or "command failed" in lower
        or "timed out" in lower
    )
    if is_error:
        first_line = next((ln.strip() for ln in content_str.splitlines() if ln.strip()), "")
        if len(first_line) > 90:
            first_line = first_line[:90] + "..."
        msg = f"agent tool {tool} error"
        if first_line:
            msg += f": {first_line}"
        return msg, True
    return "", False

def pretty_print_node(node):
    mode = _verbosity()
    if mode == "summary":
        if isinstance(node, CallToolsNode):
            for part in node.model_response.parts:
                if isinstance(part, TextPart):
                    if str(part.content).strip():
                        pretty_print(part.content)
                elif isinstance(part, ToolCallPart):
                    if part.tool_name == "final_result":
                        continue
                    # Print without wrapping so ANSI color codes don't confuse text wrapping.
                    print(_tool_call_summary(part))
        elif isinstance(node, ModelRequestNode):
            for part in node.request.parts:
                if isinstance(part, ToolReturnPart):
                    if part.tool_name == "final_result":
                        continue
                    msg, is_error = _tool_return_summary(part)
                    if is_error:
                        pretty_print(msg, color=bcolors.FAIL)
                elif isinstance(part, RetryPromptPart):
                    pretty_print("agent output validation failed; retrying", color=bcolors.WARNING)
        elif isinstance(node, UserPromptNode):
            pass
        elif isinstance(node, End):
            output = node.data.output
            for attr, value in output.__dict__.items():
                if attr not in ["tool_name", "tool_call_id"]:
                    pretty_print(attr, color=bcolors.BOLD + bcolors.ORANGE)
                    pretty_print(value)
        return

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
                    try:
                        args_dict = part.args_as_dict()
                        pretty_print_code(args_dict.get('code'))
                        pretty_print("file_path: "+ args_dict.get('file_path', 'Empty'), color=bcolors.OKCYAN)
                    except Exception:
                        pretty_print(part.args, color=bcolors.OKCYAN)
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

def print_phase(title: str):
    print(f"\n{bcolors.BOLD}{bcolors.PRETTY_BLUE}■ {title}{bcolors.ENDC}\n")


def print_best_iteration_metrics(config: Config) -> None:
    # Lazy imports: agents.agent_utils imports printing_utils, so top-level imports here would be circular.
    from agentomics.agents.steps.validation_evaluation import ValidationEvaluationStep
    from agentomics.runtime.step_outputs import load_step_output

    metric_name = config.val_metric
    snapshot_dir = config.best_iteration_snapshot_dir

    train_val: float | None = None
    val_val: float | None = None
    test_val: float | None = None

    validation_output = load_step_output(config, ValidationEvaluationStep.step_id, iteration_dir=snapshot_dir)
    if validation_output is not None:
        metrics: dict = getattr(validation_output, "metrics", {}) or {}
        train_val = metrics.get(f"train/{metric_name}")
        val_val = metrics.get(f"validation/{metric_name}")

    test_metrics_path = snapshot_dir / "test_metrics.json"
    if test_metrics_path.exists():
        try:
            test_val = json.loads(test_metrics_path.read_text()).get(metric_name)
        except Exception:
            pass

    if train_val is None and val_val is None and test_val is None:
        return

    col_w = max(len(metric_name), 6)
    header = f"  {'Metric':<{col_w}}  {'Train':>10}  {'Validation':>10}  {'Test':>10}"
    sep = f"  {'-' * col_w}  {'-' * 10}  {'-' * 10}  {'-' * 10}"

    print_phase("Best iteration metrics summary")
    print(header)
    print(sep)
    train_str = f"{train_val:.4f}" if train_val is not None else "—"
    val_str = f"{val_val:.4f}" if val_val is not None else "—"
    test_str = f"{test_val:.4f}" if test_val is not None else "—"
    print(f"  {metric_name:<{col_w}}  {train_str:>10}  {val_str:>10}  {test_str:>10}")
    print()
