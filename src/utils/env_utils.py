import os

def is_openrouter_key_available():
    if not os.getenv("OPENROUTER_API_KEY"):
        return False
    return True

def is_wandb_key_available():
    if not os.getenv("WANDB_API_KEY"):
        return False
    return True