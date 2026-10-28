"""Read-only, confined tool for browsing a dataset's supplementary folder.

Not registered in TOOL_FACTORY_REGISTRY: it is given only to the iteration planner,
with a fresh instance (and therefore a fresh budget) per planner run.
"""

import re
from pathlib import Path
from typing import Literal

from pydantic_ai import Tool

from agentomics.datasets.data_contract import SUPPLEMENTARY_DIR_NAME
from agentomics.utils.config import Config

BINARY_SNIFF_BYTES = 8192
HINT_RESERVE = 100  # room kept free in every call's output for a truncation hint
BUDGET_MESSAGE = (
    "Supplementary tool budget is spent ({reason}). Do not call this tool again: "
    "write your notes and give your plan with what you have."
)


def _is_binary(path: Path) -> bool:
    """Null byte in the first chunk (enough to tell text from PDFs, images, archives)."""
    with path.open("rb") as f:
        return b"\x00" in f.read(BINARY_SNIFF_BYTES)


def _read_lines(path: Path) -> list[str]:
    return path.read_text(encoding="utf-8", errors="replace").splitlines()


def create_supplementary_tool(config: Config):
    root = (config.dataset_dir / SUPPLEMENTARY_DIR_NAME).resolve()
    state = {"calls": 0, "chars": 0}

    def _rel(path: Path) -> str:
        return path.relative_to(root).as_posix() or "."

    def _list(target: Path) -> list[str]:
        if not target.is_dir():
            return [f"Not a directory: {_rel(target)}"]
        lines = []
        for entry in sorted(target.iterdir()):
            real = entry.resolve()
            if not real.is_relative_to(root):
                lines.append(f"{_rel(entry)}  (symlink pointing outside the supplementary folder; rejected)")
            elif real.is_dir():
                lines.append(f"{_rel(entry)}/")
            else:
                binary = ", binary (not readable)" if _is_binary(real) else ""
                lines.append(f"{_rel(entry)}  ({real.stat().st_size} B{binary})")
        return lines or ["(empty directory)"]

    def _search(target: Path, regex: re.Pattern, context_lines: int, max_matches: int, recursive: bool) -> list[str]:
        if target.is_file():
            files = [target]
        else:
            files = sorted(target.rglob("*") if recursive else target.iterdir())
        out: list[str] = []
        matches = 0
        for file in files:
            real = file.resolve()
            if not (real.is_file() and real.is_relative_to(root)) or _is_binary(real):
                continue
            text = _read_lines(real)
            hits = [i for i, line in enumerate(text) if regex.search(line)]
            shown_until = -1
            for i in hits:
                if matches >= max_matches:
                    return out + [f"[more matches exist beyond max_matches={max_matches}]"]
                matches += 1
                lo, hi = max(0, i - context_lines), min(len(text) - 1, i + context_lines)
                if out and lo > shown_until + 1:
                    out.append("--")
                for j in range(max(lo, shown_until + 1), hi + 1):
                    sep = ":" if j == i else "-"
                    out.append(f"{_rel(file)}{sep}{j + 1}{sep} {text[j]}")
                shown_until = hi
        return out or ["No matches."]

    def _supplementary(
        operation: Literal["list", "search", "read"],
        path: str = "",
        pattern: str = "",
        case_sensitive: bool = False,
        context_lines: int = 0,
        max_matches: int = 50,
        recursive: bool = True,
        start_line: int = 1,
        max_lines: int = 200,
    ) -> str:
        """
        Read-only access to the dataset's supplementary folder (papers, notes, code, docs).
        Paths are relative to the supplementary folder. Binary files (e.g. PDFs) are listed but cannot be read.
        Output per call and the number of calls are limited; when a limit is hit, write your notes and give your plan.

        Args:
            operation: "list" shows a directory's entries with sizes; "search" finds regex matches in text files; "read" returns lines of one text file.
            path: File or directory relative to the supplementary folder. Empty means the folder itself.
            pattern: (search) Regular expression to look for.
            case_sensitive: (search) Whether matching is case sensitive.
            context_lines: (search) Lines of context to show around each match.
            max_matches: (search) Maximum number of matches to return.
            recursive: (search) Whether to search subdirectories when path is a directory.
            start_line: (read) First line to return, 1-based.
            max_lines: (read) Maximum number of lines to return.
        """
        if state["calls"] >= config.supplementary_tool_max_calls:
            return BUDGET_MESSAGE.format(reason="call limit reached")
        remaining = config.supplementary_tool_max_total_chars - state["chars"]
        if remaining <= 0:
            return BUDGET_MESSAGE.format(reason="output limit reached")
        state["calls"] += 1
        limit = min(config.supplementary_tool_max_call_chars, remaining)

        target = (root / path).resolve()
        if not target.is_relative_to(root):
            return f"Error: path {path!r} is outside the supplementary folder."
        if not target.exists():
            return f"Error: path {path!r} does not exist."

        start_line, max_lines = max(start_line, 1), max(max_lines, 1)
        truncation_hint = "[truncated; list a subdirectory to see fewer entries]"
        if operation == "list":
            lines = _list(target)
        elif operation == "search":
            try:
                regex = re.compile(pattern, 0 if case_sensitive else re.IGNORECASE)
            except re.error as e:
                return f"Invalid pattern: {e}"
            lines = _search(target, regex, max(context_lines, 0), max(max_matches, 1), recursive)
            truncation_hint = "[truncated; narrow the pattern or lower the match limit]"
        elif operation == "read":
            if not target.is_file():
                return f"Error: {path!r} is not a file."
            if _is_binary(target):
                return f"Error: {path!r} is a binary file and cannot be read."
            file_lines = _read_lines(target)
            window = file_lines[start_line - 1 : start_line - 1 + max_lines]
            if not window:
                return f"(no lines at start_line={start_line}; file has {len(file_lines)} lines)"
            lines = [f"{start_line + i}: {text}" for i, text in enumerate(window)]
        else:
            return "Error: operation must be one of list, search, read."

        text = "\n".join(lines)
        cut = limit - min(HINT_RESERVE, limit // 2)
        if len(text) > cut:
            # Keep the head, cutting at a line boundary (or mid-line if the first line alone is too long).
            text = text[:cut]
            text = text.rsplit("\n", 1)[0] if "\n" in text else text
            if operation == "read":
                kept = text.count("\n") + 1
                truncation_hint = f"[truncated; continue with start_line={start_line + kept}]"
            text += "\n" + truncation_hint
        elif operation == "read" and start_line - 1 + max_lines < len(file_lines):
            text += f"\n[file continues; next start_line={start_line + len(window)}]"
        text = text[:limit]
        state["chars"] += len(text)
        return text

    return Tool(
        function=_supplementary,
        takes_ctx=False,
        max_retries=config.max_tool_retries,
        require_parameter_descriptions=True,
        name="supplementary",
        sequential=True,
    )
