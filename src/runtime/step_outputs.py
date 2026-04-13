from __future__ import annotations

import json
from pathlib import Path
from typing import Any

from utils.config import Config

def _deserialize_step_output_record(record: dict[str, Any]) -> Any:
    payload = record["payload"]
    step_id = record.get("step_id")
    output_type = None
    if step_id is not None:
        from agents.steps.step_registry import get_step_class
        try:
            output_type = getattr(get_step_class(step_id), "output_type", None)
        except KeyError:
            pass
    if output_type is None or not isinstance(payload, dict) or output_type is dict:
        return payload
    try:
        return output_type.model_validate(payload) if hasattr(output_type, "model_validate") else output_type(**payload)
    except Exception:
        return payload

def _load_step_output_file(output_path: Path) -> Any:
    if not output_path.exists():
        return None
    return _deserialize_step_output_record(json.loads(output_path.read_text(encoding="utf-8")))

def save_step_output(config: Config, step_id: str, output: Any) -> None:
    output_path = config.current_step_dir / Config.STEP_OUTPUT_FILENAME
    output_path.parent.mkdir(parents=True, exist_ok=True)
    if output_path.exists():
        raise FileExistsError(
            f"Step output for '{step_id}' already exists at {output_path}. "
            "Each step_id may only be written once per iteration."
        )
    payload = output.model_dump() if hasattr(output, "model_dump") else output
    output_path.write_text(json.dumps({"step_id": step_id, "model_type": type(output).__name__, "payload": payload}, indent=2), encoding="utf-8")

def load_step_output(config: Config, step_id: str, iteration_dir: Path) -> Any:
    return _load_step_output_file(iteration_dir / step_id / Config.STEP_OUTPUT_FILENAME)

def require_step_output(config: Config, step_id: str, iteration_dir: Path) -> Any:
    output = load_step_output(config, step_id, iteration_dir)
    if output is None:
        raise KeyError(f"Required step output '{step_id}' is missing from {iteration_dir / step_id}.")
    return output

def load_step_outputs(config: Config, iteration_dir: Path | None = None, step_sequence: list[str] | None = None) -> list[Any]:
    base_dir = iteration_dir or config.current_iteration_dir
    if not base_dir.exists():
        return []
    sequence = step_sequence or [str(s) for s in config.step_sequence]
    return [
        _load_step_output_file(base_dir / step_id / Config.STEP_OUTPUT_FILENAME)
        for step_id in sequence
        if (base_dir / step_id / Config.STEP_OUTPUT_FILENAME).exists()
    ]
