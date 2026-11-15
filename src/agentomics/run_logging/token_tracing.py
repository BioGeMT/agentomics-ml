from collections import Counter, defaultdict

from opentelemetry.sdk.trace import ReadableSpan, SpanProcessor, TracerProvider
from pydantic_ai import Agent, InstrumentationSettings


class TokenUsageProcessor(SpanProcessor):
    def __init__(self) -> None:
        self.current_step: str | None = None
        self._by_step: dict[str, Counter] = defaultdict(Counter)

    def on_start(self, span, parent_context=None) -> None:
        if self.current_step:
            span.set_attribute("agentomics.step_id", self.current_step)

    def on_end(self, span: ReadableSpan) -> None:
        attributes = span.attributes or {}
        if attributes.get("gen_ai.operation.name") != "chat":
            return
        step_id = attributes.get("agentomics.step_id", "outside_steps")
        self._by_step[step_id]["input_tokens"] += attributes.get("gen_ai.usage.input_tokens", 0)
        self._by_step[step_id]["output_tokens"] += attributes.get("gen_ai.usage.output_tokens", 0)

    def take(self) -> dict[str, Counter]:
        usage, self._by_step = self._by_step, defaultdict(Counter)
        return usage


token_usage = TokenUsageProcessor()


def enable_token_tracing() -> None:
    tracer_provider = TracerProvider()
    tracer_provider.add_span_processor(token_usage)
    Agent.instrument_all(InstrumentationSettings(tracer_provider=tracer_provider))
