import sys
from rich.prompt import Prompt
from rich.console import Console



def get_user_input_for_int(prompt_text, valid_options):
    """Safely prompt for input with TTY validation."""
    console = Console()
    while True:
        try:
            choice = Prompt.ask(prompt_text, default=None)
            if not choice or choice not in [str(option) for option in valid_options]:
                console.print(f"Please enter a number from this list: {valid_options}", style="red")
                continue
            return int(choice)

        except (KeyboardInterrupt, ValueError):
            console.print("\nInput cancelled", style="red")
            return None
    