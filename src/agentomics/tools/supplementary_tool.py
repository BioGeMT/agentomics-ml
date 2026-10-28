"""Read-only, confined tools for browsing a dataset's supplementary folder (iteration planner only).

Not in TOOL_FACTORY_REGISTRY.
"""

import functools
import subprocess
from pathlib import Path

from pydantic_ai import Tool

from agentomics.datasets.data_contract import SUPPLEMENTARY_DIR_NAME
from agentomics.utils.config import Config

MAX_OUTPUT_CHARS = 5000


def create_supplementary_tools(config: Config) -> list[Tool]:
    root = (config.dataset_dir / SUPPLEMENTARY_DIR_NAME).resolve()

    def tool(fn):
        """Return errors as messages and keep the head of over-long output, followed by fn's hint."""

        @functools.wraps(fn)
        def wrapper(*args, **kwargs) -> str:
            try:
                text, hint = fn(*args, **kwargs)
            except (ValueError, OSError) as e:
                return f"Error: {e}"
            if len(text) > MAX_OUTPUT_CHARS:
                head = text[: MAX_OUTPUT_CHARS - 100]  # leave room for the hint
                text = head + "\n" + hint(max(head.count("\n"), 1))
            return text

        return wrapper

    def resolve(path: str) -> Path:
        target = (root / path).resolve()
        if not target.is_relative_to(root):
            raise ValueError(f"path {path!r} is outside the supplementary folder")
        if not target.exists():
            raise ValueError(f"path {path!r} does not exist")
        return target

    def run(*command: str) -> str:
        done = subprocess.run(command, cwd=root, capture_output=True, text=True, timeout=30)
        return done.stdout or done.stderr

    narrow = lambda _: "[truncated; narrow the request]"

    @tool
    def ls(path: str = ""):
        """
        List a directory of the supplementary folder (papers, notes, code, docs) with file sizes.
        Binary files such as PDFs are listed but cannot be read.

        Args:
            path: Directory relative to the supplementary folder. Empty means the folder itself.
        """
        return run("ls", "-lAp", "--", str(resolve(path).relative_to(root))), narrow

    @tool
    def grep(pattern: str, path: str = "", case_sensitive: bool = False, context_lines: int = 0):
        """
        Search the text files of the supplementary folder recursively with a regular expression (binary files are skipped).

        Args:
            pattern: Extended regular expression to look for.
            path: File or directory relative to the supplementary folder. Empty means the folder itself.
            case_sensitive: Whether matching is case sensitive.
            context_lines: Lines of context to show around each match.
        """
        target = str(resolve(path).relative_to(root))
        flags = "-rInE" + ("" if case_sensitive else "i")
        return run("grep", flags, f"-C{max(context_lines, 0)}", "-e", pattern, "--", target) or "No matches.", narrow

    @tool
    def read_file(path: str, start_line: int = 1, max_lines: int = 200):
        """
        Read lines of one text file in the supplementary folder (binary files such as PDFs cannot be read).

        Args:
            path: File relative to the supplementary folder.
            start_line: First line to return, 1-based.
            max_lines: Maximum number of lines to return.
        """
        target, start = resolve(path), max(start_line, 1)
        try:
            text = target.read_text(encoding="utf-8")
        except UnicodeDecodeError:
            text = "\0"
        if "\0" in text:
            raise ValueError(f"{path!r} is a binary file and cannot be read")
        lines = text.splitlines()[start - 1 : start - 1 + max_lines]
        return "\n".join(f"{start + i}: {line}" for i, line in enumerate(lines)), lambda kept: f"[truncated; continue with start_line={start + kept}]"

    options = dict(takes_ctx=False, max_retries=config.max_tool_retries, require_parameter_descriptions=True, sequential=True)
    return [Tool(fn, **options) for fn in (ls, grep, read_file)]
