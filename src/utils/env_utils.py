#!/usr/bin/env python3
"""
Shared environment loading utilities to avoid duplication.
"""

import os
import dotenv
from rich.console import Console

def load_environment():
    """
    Load environment variables and return common configuration.
    
    Returns:
        dict: Common environment variables used across the application
    """
    dotenv.load_dotenv()
    
    return {
        'openrouter_api_key': os.getenv("OPENROUTER_API_KEY"),
        'wandb_api_key': os.getenv("WANDB_API_KEY"),
        'proxy_url': os.getenv("HTTP_PROXY"),
        'console': Console()
    }

def get_api_key(key_name: str = "OPENROUTER_API_KEY") -> str:
    """
    Get API key with environment loading if needed.
    
    Args:
        key_name: Name of the environment variable
        
    Returns:
        API key value or None
    """
    dotenv.load_dotenv()
    return os.getenv(key_name)

def ensure_dotenv_loaded():
    """Ensure dotenv is loaded (idempotent)."""
    dotenv.load_dotenv()
