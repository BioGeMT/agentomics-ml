"""Read-only, confined tools for browsing a dataset's supplementary folder (iteration planner only).

Not in TOOL_FACTORY_REGISTRY; each planner run gets fresh tools and therefore a fresh shared budget.
"""

import functools
import re
from pathlib import Path

from pydantic_ai import Tool

from agentomics.datasets.data_contract import SUPPLEMENTARY_DIR_NAME
from agentomics.utils.config import Config

BUDGET_MESSAGE = (
    "Supplementary tool budget is spent ({reason}). Do not call this tool again: "
    "write your notes and give your plan with what you have."
)


def create_supplementary_tools(config: Config) -> list[Tool]:
    root = (config.dataset_dir / SUPPLEMENTARY_DIR_NAME).resolve()
    used = {"calls": 0, "chars": 0}

    def tool(fn):
        """Confine paths, share one call/character budget across tools, and cap the output (never raises)."""

        @functools.wraps(fn)
        def wrapper(*args, **kwargs) -> str:
            remaining = config.supplementary_tool_max_total_chars - used["chars"]
            if used["calls"] >= config.supplementary_tool_max_calls:
                return BUDGET_MESSAGE.format(reason="call limit reached")
            if remaining <= 0:
                return BUDGET_MESSAGE.format(reason="output limit reached")
            used["calls"] += 1
            limit = min(config.supplementary_tool_max_call_chars, remaining)
            try:
                text, hint = fn(*args, **kwargs)
            except (ValueError, OSError, re.error) as e:
                return f"Error: {e}"
            if len(text) > limit:
                head = text[: max(limit - 100, 0)]  # leave room for the hint
                text = head + "\n" + hint(max(head.count("\n"), 1))
            used["chars"] += len(text)
            return text

        return wrapper

    def resolve(path: str) -> Path:
        target = (root / path).resolve()
        if not target.is_relative_to(root):
            raise ValueError(f"path {path!r} is outside the supplementary folder")
        if not target.exists():
            raise ValueError(f"path {path!r} does not exist")
        return target

    def text_lines(file: Path) -> list[str]:
        try:
            text = file.read_text(encoding="utf-8")
        except UnicodeDecodeError:
            text = "\0"
        if "\0" in text:
            raise ValueError(f"{file.relative_to(root)} is a binary file and cannot be read")
        return text.splitlines()

    narrow = lambda _: "[truncated; narrow the request]"

    @tool
    def ls(path: str = ""):
        """
        List a directory of the supplementary folder (papers, notes, code, docs) with file sizes.
        Binary files such as PDFs are listed but cannot be read.

        Args:
            path: Directory relative to the supplementary folder. Empty means the folder itself.
        """
        entries = sorted(e for e in resolve(path).iterdir() if e.resolve().is_relative_to(root))
        return "\n".join(f"{e.relative_to(root)}{'/' if e.is_dir() else f'  ({e.stat().st_size} B)'}" for e in entries), narrow

    @tool
    def grep(
        pattern: str,
        path: str = "",
        case_sensitive: bool = False,
        context_lines: int = 0,
        max_matches: int = 50,
        recursive: bool = True,
    ):
        """
        Search the text files of the supplementary folder with a regular expression (binary files are skipped).

        Args:
            pattern: Regular expression to look for.
            path: File or directory relative to the supplementary folder. Empty means the folder itself.
            case_sensitive: Whether matching is case sensitive.
            context_lines: Lines of context to show around each match.
            max_matches: Maximum number of matches to return.
            recursive: Whether to search subdirectories when path is a directory.
        """
        regex = re.compile(pattern, 0 if case_sensitive else re.IGNORECASE)
        target = resolve(path)
        files = [target] if target.is_file() else sorted(target.rglob("*") if recursive else target.iterdir())
        out, matches = [], 0
        for file in files:
            if not file.resolve().is_relative_to(root) or not file.is_file():
                continue
            try:
                lines = text_lines(file)
            except ValueError:
                continue
            for i, line in enumerate(lines):
                if regex.search(line) and matches < max_matches:
                    matches += 1
                    for j in range(max(i - context_lines, 0), min(i + context_lines + 1, len(lines))):
                        out.append(f"{file.relative_to(root)}{':' if j == i else '-'}{j + 1}: {lines[j]}")
        if matches >= max_matches:
            out.append(f"[stopped at max_matches={max_matches}; narrow the pattern to see more]")
        return "\n".join(out) or "No matches.", narrow

    @tool
    def read_file(path: str, start_line: int = 1, max_lines: int = 200):
        """
        Read lines of one text file in the supplementary folder (binary files such as PDFs cannot be read).

        Args:
            path: File relative to the supplementary folder.
            start_line: First line to return, 1-based.
            max_lines: Maximum number of lines to return.
        """
        start = max(start_line, 1)
        lines = text_lines(resolve(path))[start - 1 : start - 1 + max_lines]
        return "\n".join(f"{start + i}: {line}" for i, line in enumerate(lines)), lambda kept: f"[truncated; continue with start_line={start + kept}]"

    options = dict(takes_ctx=False, max_retries=config.max_tool_retries, require_parameter_descriptions=True, sequential=True)
    return [Tool(fn, **options) for fn in (ls, grep, read_file)]
