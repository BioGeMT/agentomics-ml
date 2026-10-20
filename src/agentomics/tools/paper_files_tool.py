from __future__ import annotations

import re
from pathlib import Path
from typing import Annotated, Any

from pydantic import Field
from pydantic_ai import ModelRetry, Tool

from agentomics.runtime.paper_retrieval import prepare_papers, render_pdf_page
from agentomics.utils.config import Config


def _paper_path(config: Config, path: str) -> Path:
    root = prepare_papers(config).resolve()
    resolved = (root / path).resolve()
    if not resolved.is_relative_to(root):
        raise ModelRetry(f"Path must be inside the processed papers directory: {root}")
    return resolved


def create_ls_tool(config: Config) -> Tool[Any]:
    def ls(
        path: str = ".",
        max_results: Annotated[int, Field(ge=1, le=100)] = 100,
    ) -> list[dict[str, str | int]]:
        """List files and directories containing processed scientific papers.

        Args:
            path: Directory path relative to the processed papers directory.
            max_results: Maximum number of entries to return.
        """
        directory = _paper_path(config, path)
        if not directory.is_dir():
            raise ModelRetry(f"Directory not found: {path}")
        root = config.processed_papers_dir.resolve()
        return [
            {
                "path": str(child.relative_to(root)),
                "type": "directory" if child.is_dir() else "file",
                "size": child.stat().st_size,
            }
            for child in sorted(directory.iterdir())[:max_results]
        ]

    return _tool(config, ls, "ls")


def create_glob_tool(config: Config) -> Tool[Any]:
    def glob(
        pattern: str,
        max_results: Annotated[int, Field(ge=1, le=100)] = 100,
    ) -> list[str]:
        """Find processed paper files matching a glob pattern.

        Args:
            pattern: Pattern relative to the processed papers directory, such as **/*.md.
            max_results: Maximum number of paths to return.
        """
        if Path(pattern).is_absolute() or ".." in Path(pattern).parts:
            raise ModelRetry("Glob pattern must stay inside the processed papers directory")
        root = prepare_papers(config).resolve()
        return [
            str(path.relative_to(root))
            for path in sorted(root.glob(pattern))[:max_results]
            if path.resolve().is_relative_to(root)
        ]

    return _tool(config, glob, "glob")


def create_grep_tool(config: Config) -> Tool[Any]:
    def grep(
        pattern: str,
        path: str = ".",
        max_results: Annotated[int, Field(ge=1, le=100)] = 50,
    ) -> list[dict[str, str | int]]:
        """Search processed paper text with a regular expression.

        Args:
            pattern: Case-insensitive regular expression to search for.
            path: File or directory path relative to the processed papers directory.
            max_results: Maximum number of matching lines to return.
        """
        try:
            expression = re.compile(pattern, re.IGNORECASE)
        except re.error as error:
            raise ModelRetry(f"Invalid regular expression: {error}") from error

        target = _paper_path(config, path)
        if not target.exists():
            raise ModelRetry(f"Path not found: {path}")
        root = config.processed_papers_dir.resolve()
        files = [target] if target.is_file() else sorted(target.rglob("*.md"))
        matches = []
        for file_path in files:
            for line_number, line in enumerate(
                file_path.read_text(encoding="utf-8", errors="replace").splitlines(),
                start=1,
            ):
                if expression.search(line):
                    matches.append(
                        {
                            "path": str(file_path.relative_to(root)),
                            "line": line_number,
                            "text": line[:1000],
                        }
                    )
                    if len(matches) == max_results:
                        return matches
        return matches

    return _tool(config, grep, "grep")


def create_read_file_tool(config: Config) -> Tool[Any]:
    def read_file(
        path: str,
        offset: Annotated[int, Field(ge=1)] = 1,
        limit: Annotated[int, Field(ge=1, le=500)] = 200,
    ) -> str:
        """Read numbered lines from a processed scientific paper.

        Args:
            path: File path relative to the processed papers directory.
            offset: First line to read, starting at 1.
            limit: Maximum number of lines to return.
        """
        file_path = _paper_path(config, path)
        if not file_path.is_file():
            raise ModelRetry(f"File not found: {path}")
        lines = file_path.read_text(encoding="utf-8", errors="replace").splitlines()
        return "\n".join(
            f"{line_number}: {lines[line_number - 1]}"
            for line_number in range(offset, min(offset + limit, len(lines) + 1))
        )

    return _tool(config, read_file, "read_file")


def create_render_paper_page_tool(config: Config) -> Tool[Any]:
    def render_paper_page(
        pdf_path: str,
        page_number: Annotated[int, Field(ge=1)],
    ) -> str:
        """Render one paper page as an image for inspection.

        Args:
            pdf_path: Source PDF path shown at the top of its processed Markdown file.
            page_number: One-based page number to render.
        """
        try:
            return str(render_pdf_page(config, pdf_path, page_number))
        except ValueError as error:
            raise ModelRetry(str(error)) from error

    return _tool(config, render_paper_page, "render_paper_page")


def _tool(config: Config, function: Any, name: str) -> Tool[Any]:
    return Tool(
        function,
        name=name,
        max_retries=config.max_tool_retries,
        require_parameter_descriptions=True,
        sequential=True,
    )
