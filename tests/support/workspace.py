from pathlib import Path

from agentomics.runtime.filesystem import get_file_hash


def snapshot_workspace(root: Path) -> dict[Path, tuple[int, str]]:
    """Record names, permissions, symlink targets, and file contents without following links."""
    snapshot = {}
    for path in root.rglob("*"):
        mode = path.lstat().st_mode
        if path.is_symlink():
            content = str(path.readlink())
        elif path.is_file():
            content = get_file_hash(path)
        else:
            content = ""
        snapshot[path.relative_to(root)] = (mode, content)
    return snapshot
