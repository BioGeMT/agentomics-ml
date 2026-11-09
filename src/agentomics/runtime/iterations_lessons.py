from __future__ import annotations

from dataclasses import replace

from pydantic_ai import Agent
from pydantic_ai.messages import ModelMessage, ModelRequest, SystemPromptPart, UserPromptPart
from pydantic_ai.models import Model
from pydantic_ai.settings import ModelSettings

from agentomics.agents.agent_utils import run_agent
from agentomics.utils.config import Config
from agentomics.utils.exceptions import IterationRunFailed

ITERATIONS_LESSONS_AGENT_MAX_REQUESTS = 5
NO_NEW_ITERATIONS_LESSONS = "NO NEW LESSONS"
MAX_ERROR_CHARS = 1500

ITERATIONS_LESSONS_INSTRUCTIONS = """
Maintain iterations_lessons.md: a list of lessons that help agents avoid mistakes and wasted work when they work in the future.

You receive:
- the current contents of iterations_lessons.md
- the trace of one agent step: its whole context, with its iteration number and step ID

Record a lesson when the trace shows:
- a crash, an incorrect result, substantial wasted work (for example, repeated failed attempts), or context bloat (for example, huge outputs read into the conversation), or
- a problem that was caught before it caused damage, but that another agent could easily miss and that would have caused one of the above.
The cause may be the agent, the plan it was given, the dataset, or the environment. Do not record every precaution or minor correction.

Write each lesson as a short bullet that says what to do or avoid, and why. Describe how to avoid the problem in future work, not how to fix one script. Keep the conditions under which the problem occurs (dataset property, library, model, input size). Drop incidental details such as variable names, script names and values that only made sense for that script.

Examples of when to include a lesson:
- The trace shows a crash because sequences in the dataset have different lengths.
  Lesson: "Sequences have unequal lengths. Pad them or process them one at a time before batching."
- The trace shows the agent repeatedly trying to install packages that are already installed.
  Lesson: "Conda packages stay installed for all later steps. Check with `conda list` before installing."
- The trace shows a GPU out-of-memory error that crashed a long training run and wasted its time.
  Lesson: "Fine-tuning a 650M-parameter model with batch size 64 on 512-token inputs ran out of GPU memory. Before a long GPU training run, run a short trial and check peak memory."
- The trace shows the agent repeatedly reading the full contents of large files, bloating its context.
  Lesson: "Search large files for the needed section instead of reading the whole file."

Examples of what to exclude:
- "Rename `lenght` to `length`." (one typo in one script)
- "Random forests scored worse." or "Try a transformer next." (experiment results and plans)
- Anything already stated in the prompts the agent was given, runtime bugs, provider outages, and other failures the agent cannot prevent.

Only record what the trace shows. Do not say that something worked unless the trace shows it working.

Update the existing file. Keep existing lessons unless the trace contradicts them: a step in which the problem did not occur is not evidence that the lesson is wrong. Remove or correct a lesson only if the trace shows the agent doing what the lesson warns against with no problem, or shows that its advice did not work. Merge lessons that describe the same problem into one.

Reference a file only if it contains working code that helps apply the lesson. Write the reference as: iteration number, step ID, and filename relative to the step folder. Never reference temporary files or use a reference just to record where something happened. The lesson must make sense without opening the file.
Example: "Tokenize with a fixed maximum length; unbounded inputs exhausted memory. (Working code: iteration 2, step training, `train.py`.)"

Return the complete updated iterations_lessons.md as plain Markdown. Do not add commentary or wrap the output in ``` markers. If nothing needs changing, return exactly:
NO NEW LESSONS
"""


def initialize_iterations_lessons(config: Config) -> None:
    config.iterations_lessons_path.touch(exist_ok=True)


def read_iterations_lessons(config: Config) -> str:
    return config.iterations_lessons_path.read_text(encoding="utf-8").strip()


async def update_iterations_lessons(
    config: Config,
    model: Model,
    iteration: int,
    step_id: str,
    messages: list[ModelMessage],
    error: IterationRunFailed | None = None,
) -> None:
    try:
        _, iterations_lessons = await run_agent(
            agent=_create_iterations_lessons_agent(config, model),
            user_prompt=_build_user_prompt(config, iteration, step_id, error),
            max_steps=ITERATIONS_LESSONS_AGENT_MAX_REQUESTS,
            message_history=_quote_system_prompts(messages),
            verbose=False,
        )
    except Exception:
        print(f"WARNING: {Config.ITERATIONS_LESSONS_FILENAME} was not updated after iteration {iteration} step {step_id}.")
        return
    # todo: test the approach with structured lessons (a DTO instead of a plain string) and compare after observability metrics are introduced
    if iterations_lessons.strip() != NO_NEW_ITERATIONS_LESSONS:
        config.iterations_lessons_path.write_text(f"{iterations_lessons.strip()}\n", encoding="utf-8")


def _build_user_prompt(config: Config, iteration: int, step_id: str, error: IterationRunFailed | None) -> str:
    result = "It succeeded." if error is None else f"It failed with:\n{_format_error(error)}"
    return f"""
Current {Config.ITERATIONS_LESSONS_FILENAME}:
{read_iterations_lessons(config)}

Iteration {iteration}, step {step_id} just ended. The messages above are the whole context of its agent, for you to analyze.

{result}
"""


def _quote_system_prompts(messages: list[ModelMessage]) -> list[ModelMessage]:
    return [
        replace(message, parts=[_quote_system_prompt(part) for part in message.parts])
        if isinstance(message, ModelRequest)
        else message
        for message in messages
    ]


def _quote_system_prompt(part):
    if not isinstance(part, SystemPromptPart):
        return part
    return UserPromptPart(content=f"<agent_system_prompt>\n{part.content}\n</agent_system_prompt>")


def _format_error(error: IterationRunFailed) -> str:
    return error.exception_trace[-MAX_ERROR_CHARS:]


def _create_iterations_lessons_agent(config: Config, model: Model) -> Agent[None, str]:
    return Agent(
        model=model,
        # instructions instead of system_prompt: pydantic-ai drops system_prompt when message_history is passed
        instructions=ITERATIONS_LESSONS_INSTRUCTIONS,
        model_settings=ModelSettings(temperature=config.temperature),
        output_type=str,
        retries=config.max_validation_retries,
    )
