"""Read-only, confined tools for browsing a dataset's supplementary folder (iteration planner only).

Not in TOOL_FACTORY_REGISTRY; each planner run gets fresh tools and therefore a fresh shared budget.
"""

import functools
import re
from collections.abc import Callable
from pathlib import Path

from pydantic_ai import Tool

from agentomics.datasets.data_contract import SUPPLEMENTARY_DIR_NAME
from agentomics.utils.config import Config

BINARY_SNIFF_BYTES = 8192
BUDGET_MESSAGE = (
    "Supplementary tool budget is spent ({reason}). Do not call this tool again: "
    "write your notes and give your plan with what you have."
)


def _is_binary(path: Path) -> bool:
    with path.open("rb") as f:
        return b"\x00" in f.read(BINARY_SNIFF_BYTES)


def _read_lines(path: Path) -> list[str]:
    return path.read_text(encoding="utf-8", errors="replace").splitlines()


def _head(lines: list[str], limit: int, hint: Callable[[int], str]) -> str:
    """Join lines; if over limit keep the head at a line boundary and append hint(lines_kept)."""
    text = "\n".join(lines)
    if len(text) <= limit:
        return text
    head = text[: max(limit - len(hint(len(lines))) - 1, 0)]
    if "\n" in head:  # cut mid-line only when the first line alone is too long
        head = head.rsplit("\n", 1)[0]
    return head + "\n" + hint(head.count("\n") + 1)


def create_supplementary_tools(config: Config) -> list[Tool]:
    root = (config.dataset_dir / SUPPLEMENTARY_DIR_NAME).resolve()
    state = {"calls": 0, "chars": 0, "limit": 0}
    narrow_hint = lambda _: "[truncated; narrow the pattern or lower the match limit]"

    def budgeted(fn):
        """Share one call/character budget across tools; never raises, returns a message when spent."""

        @functools.wraps(fn)
        def wrapper(*args, **kwargs) -> str:
            remaining = config.supplementary_tool_max_total_chars - state["chars"]
            if state["calls"] >= config.supplementary_tool_max_calls:
                return BUDGET_MESSAGE.format(reason="call limit reached")
            if remaining <= 0:
                return BUDGET_MESSAGE.format(reason="output limit reached")
            state["calls"] += 1
            state["limit"] = min(config.supplementary_tool_max_call_chars, remaining)
            text = fn(*args, **kwargs)[: state["limit"]]
            state["chars"] += len(text)
            return text

        return wrapper

    def rel(path: Path) -> str:
        return path.relative_to(root).as_posix() or "."

    def resolve(path: str, kind: str) -> Path | str:
        """The confined target, or an error message."""
        target = (root / path).resolve()
        if not target.is_relative_to(root):
            return f"Error: path {path!r} is outside the supplementary folder."
        if not target.exists():
            return f"Error: path {path!r} does not exist."
        if kind == "dir" and not target.is_dir():
            return f"Error: {path!r} is not a directory."
        if kind == "file" and not target.is_file():
            return f"Error: {path!r} is not a file."
        if kind == "file" and _is_binary(target):
            return f"Error: {path!r} is a binary file and cannot be read."
        return target

    @budgeted
    def ls(path: str = "") -> str:
        """
        List a directory of the supplementary folder (papers, notes, code, docs) with file sizes.
        Binary files such as PDFs are listed but cannot be read.

        Args:
            path: Directory relative to the supplementary folder. Empty means the folder itself.
        """
        target = resolve(path, "dir")
        if isinstance(target, str):
            return target
        lines = []
        for entry in sorted(target.iterdir()):
            real = entry.resolve()
            if not real.is_relative_to(root):
                lines.append(f"{rel(entry)}  (symlink pointing outside the folder; rejected)")
            elif real.is_dir():
                lines.append(f"{rel(entry)}/")
            else:
                binary = ", binary (not readable)" if _is_binary(real) else ""
                lines.append(f"{rel(entry)}  ({real.stat().st_size} B{binary})")
        return _head(lines or ["(empty directory)"], state["limit"], lambda _: "[truncated; list a subdirectory]")

    @budgeted
    def grep(
        pattern: str,
        path: str = "",
        case_sensitive: bool = False,
        context_lines: int = 0,
        max_matches: int = 50,
        recursive: bool = True,
    ) -> str:
        """
        Search text files in the supplementary folder with a regular expression (binary files are skipped).

        Args:
            pattern: Regular expression to look for.
            path: File or directory relative to the supplementary folder. Empty means the folder itself.
            case_sensitive: Whether matching is case sensitive.
            context_lines: Lines of context to show around each match.
            max_matches: Maximum number of matches to return.
            recursive: Whether to search subdirectories when path is a directory.
        """
        target = resolve(path, "any")
        if isinstance(target, str):
            return target
        try:
            regex = re.compile(pattern, 0 if case_sensitive else re.IGNORECASE)
        except re.error as e:
            return f"Invalid pattern: {e}"
        context_lines, max_matches = max(context_lines, 0), max(max_matches, 1)
        files = [target] if target.is_file() else sorted(target.rglob("*") if recursive else target.iterdir())
        out: list[str] = []
        matches = 0
        for file in files:
            real = file.resolve()
            if not (real.is_file() and real.is_relative_to(root)) or _is_binary(real):
                continue
            text = _read_lines(real)
            for i in (i for i, line in enumerate(text) if regex.search(line)):
                if matches >= max_matches:
                    out.append(f"[more matches exist beyond max_matches={max_matches}]")
                    return _head(out, state["limit"], narrow_hint)
                matches += 1
                lo, hi = max(0, i - context_lines), min(len(text) - 1, i + context_lines)
                out += ["--"] if out and context_lines else []
                out += [f"{rel(file)}{':' if j == i else '-'}{j + 1}: {text[j]}" for j in range(lo, hi + 1)]
        return _head(out or ["No matches."], state["limit"], narrow_hint)

    @budgeted
    def read_file(path: str, start_line: int = 1, max_lines: int = 200) -> str:
        """
        Read lines of one text file in the supplementary folder (binary files such as PDFs cannot be read).

        Args:
            path: File relative to the supplementary folder.
            start_line: First line to return, 1-based.
            max_lines: Maximum number of lines to return.
        """
        target = resolve(path, "file")
        if isinstance(target, str):
            return target
        start_line, max_lines = max(start_line, 1), max(max_lines, 1)
        file_lines = _read_lines(target)
        window = file_lines[start_line - 1 : start_line - 1 + max_lines]
        if not window:
            return f"(no lines at start_line={start_line}; file has {len(file_lines)} lines)"
        lines = [f"{start_line + i}: {text}" for i, text in enumerate(window)]
        next_line = start_line + len(window)
        if next_line <= len(file_lines):
            lines.append(f"[file continues; next start_line={next_line}]")
        return _head(lines, state["limit"], lambda kept: f"[truncated; continue with start_line={start_line + kept}]")

    options = dict(takes_ctx=False, max_retries=config.max_tool_retries, require_parameter_descriptions=True, sequential=True)
    return [Tool(fn, **options) for fn in (ls, grep, read_file)]
