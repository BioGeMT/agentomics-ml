from rich.prompt import Prompt
from rich.console import Console

def get_user_input_for_int(prompt_text: str, default: int = None, valid_options: list[int] = None, color: str = "green") -> int:
    """
    Safely prompt for input with TTY validation.
    If valid_options are not provided, any integer input is accepted.
    If default is provided, pressing ENTER returns the default value.
    """
    console = Console()
    if(valid_options):
        prompt_text = f"{prompt_text} (options: {valid_options})"
    if(default is None):
        default = valid_options[0] if valid_options else 0
    prompt_text = f"{prompt_text} Press ENTER for default value ({default})"
    if(color):
        prompt_text = f"[{color}]{prompt_text}[/{color}]"
    while True:
        choice = Prompt.ask(prompt_text, default=None)
        if choice is None and default:
            return default
        if valid_options and (not choice or choice not in [str(option) for option in valid_options]):
            console.print(f"Please enter a number from this list: {valid_options}", style="red")
            continue
        if not choice.isdigit():
            console.print("Please enter a valid integer.", style="red")
            continue
        return int(choice)