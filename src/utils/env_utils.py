import os

def are_wandb_vars_available():
    wandb_vars_needed = [        
        "WANDB_PROJECT_NAME",
        "WANDB_ENTITY",
        "WANDB_API_KEY",
    ]
    for var in wandb_vars_needed:
        if not os.getenv(var):
            return False

    return True