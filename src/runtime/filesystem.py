from __future__ import annotations

import os
import shutil
import subprocess
from pathlib import Path


def remove_path(path: Path) -> None:
    if path.is_symlink() or path.is_file():
        path.unlink(missing_ok=True)
        return
    if path.is_dir():
        shutil.rmtree(path)

def create_absolute_symlink(target: Path, link: Path) -> None:
    """Create a symlink at link pointing at the resolved absolute path of target."""
    os.symlink(Path(target).resolve(), link)

def find_symlinks_in_dir(root: Path, include_root: bool = False) -> list[Path]:
    root = Path(root)
    paths = [p for p in root.rglob("*") if p.is_symlink()]
    if include_root and root.is_symlink():
        paths.append(root)
    return paths

def rewrite_symlinks_to_absolute(root: Path) -> None:
    symlinks = find_symlinks_in_dir(root)
    for path in symlinks:
        absolute_target = path.resolve()
        path.unlink()
        os.symlink(absolute_target, path)

def validate_symlinks_targets_in(root: Path, allowed_parent: Path) -> None:
    """Raise ValueError if any symlink under root resolves outside allowed_parent or its symlinked contents."""
    allowed_resolved_roots = [allowed_parent.resolve()]
    allowed_resolved_roots.extend(
        path.resolve() for path in find_symlinks_in_dir(allowed_parent)
    )
    bad = [
        path for path in find_symlinks_in_dir(root)
        if not any(path.resolve().is_relative_to(allowed) for allowed in allowed_resolved_roots)
    ]
    if bad:
        relative = [str(p.relative_to(root)) for p in bad[:10]]
        raise ValueError(
            f"Symlinks in {root.name}/ must point to files inside {allowed_parent}. "
            f"Up to first 10 invalid symlinks: {relative}"
        )

def chown_tree_to_root(path: Path) -> None:
    """Transfer ownership of a directory tree to root and normalize permissions.

    Used after archiving step/iteration outputs to prevent the agent user from
    modifying files it previously created. Root can later delete the tree without
    needing to first restore write permission.

    Normalizing permissions (u=rwX,go=rX) strips any world-writable bits the agent
    may have set, so ownership change alone is sufficient to block agent writes.
    """
    subprocess.run(["chown", "-R", "root:root", str(path)], check=True)
    subprocess.run(["chmod", "-R", "u=rwX,go=rX", str(path)], check=True)
