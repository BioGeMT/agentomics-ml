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
BUDGET_MESSAGE = (
    "Supplementary tool budget is spent ({reason}). Do not call this tool again: "
    "write your notes and give your plan with what you have."
)


def _is_binary(path: Path) -> bool:
    """Null byte in the first chunk (enough to tell text from PDFs, images, archives)."""
    with path.open("rb") as f:
        return b"\x00" in f.read(BINARY_SNIFF_BYTES)


def _human_size(size: int) -> str:
    return f"{size} B" if size < 1024 else f"{size / 1024:.1f} KiB"


def _head(lines: list[str], limit: int) -> tuple[str, int, bool]:
    """Join whole leading lines that fit in `limit` chars.

    Returns the text, the number of whole lines kept, and whether the first line alone did not
    fit and was cut instead (then 0 whole lines are kept but a prefix of line 1 is shown).
    """
    kept: list[str] = []
    total = 0
    for line in lines:
        cost = len(line) + 1
        if total + cost > limit:
            break
        kept.append(line)
        total += cost
    if not kept and lines:
        return lines[0][: max(limit, 0)], 0, True
    return "\n".join(kept), len(kept), False


def _fit(lines: list[str], limit: int, tail_for) -> tuple[str, bool]:
    """Head of `lines` plus the tail text, together at most `limit` chars.

    `tail_for(kept, partial, truncated)` builds the hint or note (may be empty).
    """
    budget = limit
    while True:
        body, kept, partial = _head(lines, budget)
        truncated = partial or kept < len(lines)
        tail = tail_for(kept, partial, truncated)
        text = body + ("\n" + tail if tail else "")
        if len(text) <= limit:
            return text, truncated
        if budget <= 0:
            return text[:limit], truncated
        budget -= len(text) - limit


def create_supplementary_tool(config: Config):
    root = (config.dataset_dir / SUPPLEMENTARY_DIR_NAME).resolve()
    state = {"calls": 0, "chars": 0}

    def _inside(real: Path) -> bool:
        return real == root or root in real.parents

    def _rel(path: Path) -> str:
        return path.relative_to(root).as_posix() or "."

    def _list(target: Path) -> list[str]:
        if not target.is_dir():
            return [f"Not a directory: {_rel(target)}"]
        lines = []
        for entry in sorted(target.iterdir()):
            real = entry.resolve()
            name = entry.relative_to(root).as_posix()
            if not _inside(real):
                lines.append(f"{name}  (symlink pointing outside the supplementary folder; rejected, not accessible)")
                continue
            if real.is_dir():
                lines.append(f"{name}/")
            else:
                note = ", binary (not readable)" if _is_binary(real) else ""
                lines.append(f"{name}  ({_human_size(real.stat().st_size)}{note})")
        return lines or ["(empty directory)"]

    def _search(target, pattern, case_sensitive, context_lines, max_matches, recursive) -> tuple[list[str], str]:
        try:
            regex = re.compile(pattern, 0 if case_sensitive else re.IGNORECASE)
        except re.error as e:
            return [f"Invalid pattern: {e}"], ""
        skipped_outside = 0
        skipped_binary = 0
        if target.is_file():
            candidates = [target]
        elif recursive:
            candidates = sorted(target.rglob("*"))
        else:
            candidates = sorted(target.iterdir())
        files = []
        for candidate in candidates:
            if candidate.is_symlink() and not _inside(candidate.resolve()):
                skipped_outside += 1
            elif candidate.is_file():
                files.append(candidate)
        out: list[str] = []
        matches = 0
        more = False
        for file in files:
            real = file.resolve()
            if _is_binary(real):
                skipped_binary += 1
                continue
            text = real.read_text(encoding="utf-8", errors="replace").splitlines()
            shown_until = -1
            for i, line in enumerate(text):
                if not regex.search(line):
                    continue
                if matches >= max_matches:
                    more = True
                    break
                matches += 1
                lo, hi = max(0, i - context_lines), min(len(text) - 1, i + context_lines)
                if lo > shown_until + 1 and out:
                    out.append("--")
                for j in range(max(lo, shown_until + 1), hi + 1):
                    sep = ":" if j == i else "-"
                    out.append(f"{_rel(file)}{sep}{j + 1}{sep} {text[j]}")
                shown_until = hi
            if more:
                break
        notes = []
        if more:
            notes.append(f"[more matches exist beyond max_matches={max_matches}]")
        if skipped_outside:
            notes.append(f"[skipped {skipped_outside} symlink(s) pointing outside the supplementary folder]")
        if skipped_binary:
            notes.append(f"[skipped {skipped_binary} binary file(s)]")
        return out or ["No matches."], " ".join(notes)

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
        context_lines = max(context_lines, 0)
        max_matches = max(max_matches, 1)
        max_lines = max(max_lines, 1)
        start_line = max(start_line, 1)
        if state["calls"] >= config.supplementary_tool_max_calls:
            return BUDGET_MESSAGE.format(reason="call limit reached")
        remaining = config.supplementary_tool_max_total_chars - state["chars"]
        if remaining <= 0:
            return BUDGET_MESSAGE.format(reason="output limit reached")
        state["calls"] += 1
        limit = min(config.supplementary_tool_max_call_chars, remaining)

        target = (root / path).resolve()
        if not _inside(target):
            return f"Error: path {path!r} is outside the supplementary folder."
        if not target.exists():
            return f"Error: path {path!r} does not exist."

        note = ""
        if operation == "list":
            lines = _list(target)

            def tail_for(kept: int, partial: bool, truncated: bool) -> str:
                return "[truncated; list a subdirectory to see fewer entries]" if truncated else ""

        elif operation == "search":
            if not pattern:
                return "Error: pattern is required for search."
            lines, note = _search(target, pattern, case_sensitive, context_lines, max_matches, recursive)

            def tail_for(kept: int, partial: bool, truncated: bool) -> str:
                if truncated:
                    return "[truncated; narrow the pattern or lower the match limit]" + (f" {note}" if note else "")
                return note

        elif operation == "read":
            if not target.is_file():
                return f"Error: {path!r} is not a file."
            if _is_binary(target):
                return f"Error: {path!r} is a binary file and cannot be read."
            all_lines = target.read_text(encoding="utf-8", errors="replace").splitlines()
            start = start_line
            window = all_lines[start - 1 : start - 1 + max_lines]
            lines = [f"{start + i}: {text}" for i, text in enumerate(window)]
            continues = start - 1 + max_lines < len(all_lines)
            if not window:
                lines = [f"(no lines at start_line={start}; file has {len(all_lines)} lines)"]

            def tail_for(kept: int, partial: bool, truncated: bool) -> str:
                if partial:
                    return (
                        f"[line {start} is longer than the per-call limit and was cut; "
                        f"continue with start_line={start + 1} to skip the rest of it, "
                        "or use search with a pattern to find text inside it]"
                    )
                if truncated:
                    return f"[truncated; continue with start_line={start + kept}]"
                if continues:
                    return f"[file continues; next start_line={start + len(window)}]"
                return ""

        else:
            return "Error: operation must be one of list, search, read."

        text, _ = _fit(lines, limit, tail_for)
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
