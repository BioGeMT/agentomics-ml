from __future__ import annotations

import traceback
from collections.abc import Callable
from typing import Any

from pydantic_ai import Agent
from pydantic_ai.messages import (
    ModelMessage,
    RetryPromptPart,
    TextPart,
    ToolCallPart,
    ToolReturnPart,
)
from pydantic_ai.models import Model
from pydantic_ai.settings import ModelSettings

from agentomics.agents.agent_utils import run_agent
from agentomics.utils.config import Config
from agentomics.utils.exceptions import IterationRunFailed

ITERATIONS_LESSONS_AGENT_MAX_REQUESTS = 5
NO_NEW_ITERATIONS_LESSONS = "NO NEW LESSONS"
MAX_PART_CHARS = 1500
MAX_TRANSCRIPT_CHARS = 60000

# Parts not listed here (system prompt, user prompt, thinking) are left out of the transcript
# Only the text needed for lessons is taken from each listed part
_PART_CONTENT: dict[type, Callable[[Any], str]] = {
    ToolCallPart: lambda part: f"{part.tool_name} {part.args_as_json_str()}",
    ToolReturnPart: lambda part: f"{part.tool_name} {part.model_response_str()}",
    RetryPromptPart: lambda part: part.model_response(),
    TextPart: lambda part: part.content,
}

ITERATIONS_LESSONS_SYSTEM_PROMPT = """
You maintain iterations_lessons.md, the memory of mistakes made by an autonomous ML agent that runs the same sequence of steps over several iterations on one dataset.
After every step, successful or failed, you receive the current file and the transcript of that step.

A lesson is a mistake the agent made and what to do instead, such as:
- an error or dead end and its root cause
- a validator complaint and how it was satisfied
- an environment problem, such as a missing package, a timeout, or a wrong path
- a workflow problem, such as repeating the same tool call or not finishing the step
Don't record findings about the data or the model unless they caused a mistake, good practices, plans, or generic ML advice. Never invent anything the transcript doesn't show.

Write each lesson as an instruction for the next agent, under about 25 words:
- <Step or "Any step">: <what to do or avoid>. <short reason>.
For example:
- Training: `xgboost` is not installed. Use `sklearn.ensemble.HistGradientBoostingClassifier` instead.
- Any step: `write_python` rejects paths outside the current step folder. Save new scripts there.
Keep the names the next agent needs, such as columns, packages, and functions. Leave out full error messages, absolute paths, script names, names of functions or variables the agent wrote itself, and the story of how the fix was found.

Return the complete new file as plain markdown, without code fences or commentary:
- Keep every existing lesson unless the transcript shows it is wrong. In that case, correct or remove it.
- Write one lesson per root cause, not one per symptom. If several errors come from the same mistake, describe it once.
- If the file already states the same mistake in other words, merge the new detail into that bullet instead of adding a near-duplicate.
- Keep the file under 60 bullets.
- If the step made no new mistakes, return exactly: NO NEW LESSONS
"""


def read_iterations_lessons(config: Config) -> str:
    if not config.iterations_lessons_path.exists():
        return ""
    return config.iterations_lessons_path.read_text(encoding="utf-8").strip()


async def update_iterations_lessons(
    config: Config,
    model: Model,
    step_name: str,
    messages: list[ModelMessage],
    error: Exception | None = None,
) -> None:
    try:
        _, iterations_lessons = await run_agent(
            agent=_create_iterations_lessons_agent(config, model),
            user_prompt=_build_user_prompt(config, step_name, messages, error),
            max_steps=ITERATIONS_LESSONS_AGENT_MAX_REQUESTS,
            message_history=None,
            verbose=False,
        )
    except Exception:
        print(f"WARNING: {Config.ITERATIONS_LESSONS_FILENAME} was not updated after step {step_name}.")
        return
    # todo: test the approach with structured lessons (a DTO instead of a plain string) and compare after observability metrics are introduced
    if iterations_lessons.strip() != NO_NEW_ITERATIONS_LESSONS:
        config.iterations_lessons_path.write_text(f"{iterations_lessons.strip()}\n", encoding="utf-8")


def _build_user_prompt(config: Config, step_name: str, messages: list[ModelMessage], error: Exception | None) -> str:
    result = "It succeeded." if error is None else f"It failed with:\n{_format_error(error)}"
    return f"""
Current {Config.ITERATIONS_LESSONS_FILENAME}:
{read_iterations_lessons(config)}

Step {step_name} just ended. Below is the transcript of its agent, for you to analyze:
{_format_transcript(messages)}

{result}
"""


def _format_transcript(messages: list[ModelMessage]) -> str:
    parts = [part for message in messages for part in message.parts if type(part) in _PART_CONTENT]
    contents = [(type(part).__name__, _PART_CONTENT[type(part)](part)) for part in parts]
    lines = [f"[{name}] {content[:MAX_PART_CHARS]}" for name, content in contents if content.strip()]
    return "\n".join(lines)[-MAX_TRANSCRIPT_CHARS:]


def _format_error(error: Exception) -> str:
    trace = (
        error.exception_trace if isinstance(error, IterationRunFailed) else "".join(traceback.format_exception(error))
    )
    return trace[-MAX_PART_CHARS:]


def _create_iterations_lessons_agent(config: Config, model: Model) -> Agent[None, str]:
    return Agent(
        model=model,
        system_prompt=ITERATIONS_LESSONS_SYSTEM_PROMPT,
        model_settings=ModelSettings(temperature=config.temperature),
        output_type=str,
        retries=config.max_validation_retries,
    )
