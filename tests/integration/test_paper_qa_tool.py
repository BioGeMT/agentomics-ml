import asyncio

import pytest
from pydantic_ai import Agent
from pydantic_ai.messages import ModelMessagesTypeAdapter, ModelResponse, TextPart, ToolCallPart, ToolReturnPart
from pydantic_ai.models.function import FunctionModel
from pydantic_ai.models.test import TestModel
from reportlab.pdfgen import canvas

from agentomics.runtime.run_lifecycle import run_agentomics
from agentomics.tools.tool_registry import create_tool


def test_paper_qa_answers_from_text_and_images_without_forwarding_reader_context(default_agent_config, tmp_path):
    config = default_agent_config
    config.fetched_papers_dir.mkdir(parents=True)
    pdf = canvas.Canvas(str(config.fetched_papers_dir / "example.pdf"))
    pdf.drawString(72, 720, "The classifier uses a convolutional network.")
    pdf.setFillColorRGB(0, 0, 1)
    pdf.rect(72, 600, 100, 100, fill=1)
    pdf.save()
    outside = tmp_path / "secret.md"
    outside.write_text("PRIVATE_SENTINEL")
    config.processed_papers_dir.mkdir(parents=True)
    (config.processed_papers_dir / "escape.md").symlink_to(outside)
    parent_requests = []
    reader_requests = []
    reader_calls = iter([
        ("search_files", {"pattern": "convolutional"}),
        ("read_file", {"path": "escape.md"}),
        ("read_file", "document"),
        ("read_file", "image"),
    ])

    def respond(messages, info):
        if any(tool.name == "paper_qa" for tool in info.function_tools):
            parent_requests.append(list(messages))
            if len(parent_requests) == 1:
                return ModelResponse(parts=[ToolCallPart("paper_qa", {"question": "What model and figure does the paper show?"})])
            return ModelResponse(parts=[TextPart("The paper uses a CNN and shows a blue square.")])

        reader_requests.append(list(messages))
        document = next(config.processed_papers_dir.glob("*/document.md"))
        call = next(reader_calls, None)
        if call is not None:
            name, args = call
            if args == "document":
                args = {"path": str(document.relative_to(config.processed_papers_dir))}
            elif args == "image":
                args = {"path": str((document.parent / "pages" / "page_0001.png").resolve())}
            return ModelResponse(parts=[ToolCallPart(name, args)])
        return ModelResponse(parts=[ToolCallPart(info.output_tools[0].name, {
            "answer": "The classifier is a CNN; its figure shows a blue square.",
            "sources": [{"path": str(document.relative_to(config.processed_papers_dir)), "line": 9}],
        })])

    parent = Agent(FunctionModel(respond), tools=[create_tool(config, "paper_qa")], instructions="PRIVATE_PARENT_CONTEXT")
    asyncio.run(parent.run("Read the downloaded paper."))

    payload = next(
        part.content for message in parent_requests[-1] for part in message.parts
        if isinstance(part, ToolReturnPart) and part.tool_name == "paper_qa"
    )
    assert payload.answer.startswith("The classifier is a CNN")
    assert (config.processed_papers_dir / payload.sources[0].path).is_file()
    reader_json = ModelMessagesTypeAdapter.dump_json(reader_requests[-1])
    assert b"image/png" in reader_json
    assert b"PRIVATE_SENTINEL" not in reader_json
    assert b"PRIVATE_PARENT_CONTEXT" not in ModelMessagesTypeAdapter.dump_json(reader_requests[0])
    assert b"image/png" not in ModelMessagesTypeAdapter.dump_json(parent_requests[-1])


def test_run_rejects_text_only_models_with_paper_qa(default_agent_config):
    default_agent_config.tool_ids = ["paper_qa"]
    model = TestModel(profile={"supports_image_input": False})
    with pytest.raises(ValueError, match="must support image input"):
        asyncio.run(run_agentomics(default_agent_config, model, model, model, None))
