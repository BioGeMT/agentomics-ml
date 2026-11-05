"""Optional event log of a run, in the AG-UI protocol, for a UI to follow.

Enabled by setting AGENTOMICS_UI_EVENTS=1. Events are appended to <workspace>/logs/ui_events.jsonl;
a UI follows the run by reading that file as it grows. When the variable is not set, every function
here is a no-op and the AG-UI dependencies are never imported.
"""
import os
import time
from contextlib import contextmanager
from pathlib import Path
from typing import Any

from pydantic import BaseModel, ConfigDict

UI_EVENTS_ENV = "AGENTOMICS_UI_EVENTS"
EVENTS_FILENAME = "ui_events.jsonl"
USAGE_EVENT = "usage"


class Usage(BaseModel):
    """The value of the CUSTOM "usage" event: one model response's tokens and price.

    cost_usd comes from pydantic-ai's model price list at the provider's per-token rates; it is
    None when the model is not in the list. For a subscription such as ChatGPT (Codex) it is what
    the same tokens would cost on the API, not what the user is billed.
    """

    model_config = ConfigDict(extra="forbid")

    model: str | None
    provider: str | None
    input_tokens: int
    output_tokens: int
    cache_read_tokens: int
    cost_usd: float | None

_events_path: Path | None = None


def publish(event: Any) -> None:
    """Append an ag_ui.core event to the run's event log."""
    if _events_path is None:
        return
    if getattr(event, "timestamp", None) is None:
        event.timestamp = int(time.time() * 1000)  # lets a UI show when each step ran, including on replay
    line = event.model_dump_json(by_alias=True, exclude_none=True)
    with _events_path.open("a", encoding="utf-8") as events_file:
        events_file.write(line + "\n")


@contextmanager
def run_stream(workspace_dir: Path, thread_id: str):
    """Log the run's events for the duration of the workflow, bracketed by RUN_STARTED/RUN_FINISHED.

    Yields a dict whose contents become RUN_FINISHED's result.
    """
    global _events_path
    result: dict[str, Any] = {}
    if os.getenv(UI_EVENTS_ENV) != "1":
        yield result
        return

    from ag_ui.core import EventType, RunErrorEvent, RunFinishedEvent, RunStartedEvent

    _events_path = Path(workspace_dir) / "logs" / EVENTS_FILENAME
    _events_path.parent.mkdir(parents=True, exist_ok=True)
    _events_path.write_text("", encoding="utf-8")
    publish(RunStartedEvent(type=EventType.RUN_STARTED, thread_id=thread_id, run_id=thread_id))
    try:
        yield result
    except BaseException as error:
        message = f"{type(error).__name__}: {error}" if str(error) else type(error).__name__
        if isinstance(error, SystemExit) and error.code in (0, None):
            publish(RunFinishedEvent(type=EventType.RUN_FINISHED, thread_id=thread_id, run_id=thread_id, result=result))
        else:
            publish(RunErrorEvent(type=EventType.RUN_ERROR, message=message))
        raise
    else:
        publish(RunFinishedEvent(type=EventType.RUN_FINISHED, thread_id=thread_id, run_id=thread_id, result=result))
    finally:
        _events_path = None


@contextmanager
def step(config: Any, display_name: str):
    """Bracket a workflow step with STEP_STARTED/STEP_FINISHED, including when the step fails."""
    if _events_path is None:
        yield
        return
    from ag_ui.core import EventType, StepFinishedEvent, StepStartedEvent
    from agentomics.runtime.read_write_utils import load_current_iteration_index

    name = f"Iteration {load_current_iteration_index(config)}: {display_name}"
    publish(StepStartedEvent(type=EventType.STEP_STARTED, step_name=name))
    try:
        yield
    finally:
        publish(StepFinishedEvent(type=EventType.STEP_FINISHED, step_name=name))


class AgentEvents:
    """Converts completed pydantic-ai agent nodes into AG-UI events.

    Works from finished nodes rather than streaming model responses, so model calls are made
    exactly as they are without the UI: tool calls and results appear as soon as they happen,
    and text arrives as whole messages.
    """

    def __init__(self) -> None:
        from pydantic_ai.ui.ag_ui import AGUIEventStream

        self._stream = AGUIEventStream()

    async def publish_node(self, node: Any) -> None:
        from pydantic_ai import CallToolsNode

        if isinstance(node, CallToolsNode):
            # Each model response is a new assistant message. transform_stream() would do this on
            # its turn transitions; converting nodes one by one, we do it here.
            self._stream.new_message_id()
        async for native_event in _native_events(node):
            async for event in self._stream.handle_event(native_event):
                publish(event)
        if isinstance(node, CallToolsNode):
            _publish_usage(node.model_response)


def _publish_usage(response: Any) -> None:
    """One model response's Usage, as a CUSTOM "usage" event inside the current step."""
    from ag_ui.core import CustomEvent

    try:
        cost = float(response.cost().total_price)
    except Exception:
        cost = None
    usage = Usage(
        model=response.model_name,
        provider=response.provider_name,
        input_tokens=response.usage.input_tokens,
        output_tokens=response.usage.output_tokens,
        cache_read_tokens=response.usage.cache_read_tokens,
        cost_usd=cost,
    )
    publish(CustomEvent(name=USAGE_EVENT, value=usage.model_dump()))


class _DisabledAgentEvents:
    async def publish_node(self, node: Any) -> None:
        return None


def agent_events() -> AgentEvents | _DisabledAgentEvents:
    return _DisabledAgentEvents() if _events_path is None else AgentEvents()


async def _native_events(node: Any):
    from pydantic_ai import CallToolsNode, ModelRequestNode
    from pydantic_ai.messages import (
        FunctionToolCallEvent,
        FunctionToolResultEvent,
        PartEndEvent,
        PartStartEvent,
        RetryPromptPart,
        ToolCallPart,
        ToolReturnPart,
    )

    if isinstance(node, CallToolsNode):
        for index, part in enumerate(node.model_response.parts):
            yield PartStartEvent(index=index, part=part)
            yield PartEndEvent(index=index, part=part)
            if isinstance(part, ToolCallPart):
                yield FunctionToolCallEvent(part=part)
    elif isinstance(node, ModelRequestNode):
        for part in node.request.parts:
            if isinstance(part, (ToolReturnPart, RetryPromptPart)) and part.tool_call_id:
                yield FunctionToolResultEvent(part=part)

