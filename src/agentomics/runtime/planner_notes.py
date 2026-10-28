from __future__ import annotations

from agentomics.runtime.read_write_utils import get_archived_iterations
from agentomics.utils.config import Config

PLANNER_NOTES_FILENAME = "planner_notes.md"
ITERATION_PLAN_STEP_ID = "iteration_plan"


def load_current_planner_notes(config: Config) -> str:
    """Return the notes of the latest archived iteration that has any; empty if none."""
    for iteration in reversed(get_archived_iterations(config)):
        notes_path = config.iteration_dir(iteration) / ITERATION_PLAN_STEP_ID / PLANNER_NOTES_FILENAME
        if notes_path.is_file():
            return notes_path.read_text(encoding="utf-8")
    return ""


def save_planner_notes(config: Config, notes: str) -> None:
    """Write this iteration's notes next to the planner step output."""
    (config.current_step_dir / PLANNER_NOTES_FILENAME).write_text(notes, encoding="utf-8")
