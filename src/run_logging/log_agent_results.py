import json
from pathlib import Path
from typing import Any

def log_agent_step_result_to_file(step_name: str, result: Any, iteration: int, config):
    """
    Log the complete agent step result to a JSON file for inspection.

    Args:
        step_name: Name of the step (e.g., 'data_exploration', 'model_training')
        result: The AgentRunResult object from run_agent
        iteration: The current iteration number
        config: The Config object with run directory info
    """
    logs_dir = config.runs_dir / config.agent_id / "agent_logs"
    logs_dir.mkdir(exist_ok=True, parents=True)

    filename = f"iteration_{iteration}_step_{step_name}.json"
    filepath = logs_dir / filename

    usage_data = result.usage() if callable(result.usage) else result.usage

    new_messages_bytes = result.new_messages_json()
    new_messages = json.loads(new_messages_bytes.decode('utf-8')) if isinstance(new_messages_bytes, bytes) else new_messages_bytes

    log_data = {
        "step_name": step_name,
        "iteration": iteration,
        "messages": new_messages,
        "usage": {
            "requests": usage_data.requests if hasattr(usage_data, 'requests') else None,
            "request_tokens": usage_data.request_tokens if hasattr(usage_data, 'request_tokens') else None,
            "response_tokens": usage_data.response_tokens if hasattr(usage_data, 'response_tokens') else None,
            "total_tokens": usage_data.total_tokens if hasattr(usage_data, 'total_tokens') else None,
        },
        "output_data": str(result.data),
    }

    with open(filepath, 'w') as f:
        json.dump(log_data, f, indent=2)

    return log_data
