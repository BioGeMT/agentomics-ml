from pathlib import Path
from typing import Any

from pydantic_ai import ModelRetry, Tool
from pydantic_ai.messages import BinaryContent, ToolReturn

from agentomics.utils.config import Config


IMAGE_MEDIA_TYPES = {
    ".gif": "image/gif",
    ".jpeg": "image/jpeg",
    ".jpg": "image/jpeg",
    ".png": "image/png",
    ".webp": "image/webp",
}
MAX_IMAGE_BYTES = 20 * 1024 * 1024


def create_inspect_image_tool(config: Config) -> Tool[Any]:
    def _inspect_image(image_path: str, query: str) -> ToolReturn:
        """Send an image to the configured model, which must support image input.

        Args:
            image_path: Image inside the current step or processed papers directory.
            query: Question to answer using the image.
        """
        path = Path(image_path).resolve()
        allowed_roots = (
            config.current_step_dir.resolve(),
            config.processed_papers_dir.resolve(),
        )
        if not any(path.is_relative_to(root) for root in allowed_roots):
            raise ModelRetry("Image must be inside the current step or processed papers directory")

        media_type = IMAGE_MEDIA_TYPES.get(path.suffix.lower())
        if not path.is_file() or media_type is None:
            raise ModelRetry(f"Unsupported or missing image: {image_path}")
        if path.stat().st_size > MAX_IMAGE_BYTES:
            raise ModelRetry(f"Image exceeds {MAX_IMAGE_BYTES // (1024 * 1024)} MB: {image_path}")

        return ToolReturn(
            return_value=f"Loaded {path}",
            content=[
                query,
                BinaryContent(data=path.read_bytes(), media_type=media_type),
            ],
        )

    return Tool(
        _inspect_image,
        name="inspect_image",
        max_retries=config.max_tool_retries,
        require_parameter_descriptions=True,
    )
