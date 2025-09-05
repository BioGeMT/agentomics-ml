import os

def is_wandb_key_available():
    if not os.getenv("WANDB_API_KEY"):
        return False
    return True