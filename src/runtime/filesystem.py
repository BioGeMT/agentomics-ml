from __future__ import annotations

import os
import shutil
import stat
import subprocess
from pathlib import Path


def remove_path(path: Path) -> None:
    if path.is_symlink() or path.is_file():
        path.unlink(missing_ok=True)
        return
    if path.is_dir():
        shutil.rmtree(path)

def copy_workspace_tree(
    source_dir: Path,
    destination_dir: Path,
) -> None:
    excluded_names = {
        ".conda",
        "__pycache__",
        ".cache",
    }

    def ignore(directory: str, names: list[str]) -> set[str]:
        ignored = set()
        if Path(directory).resolve() == source_dir.resolve():
            ignored.update(name for name in names if name in excluded_names)
        return ignored

    shutil.copytree(
        source_dir,
        destination_dir,
        symlinks=False,
        copy_function=shutil.copy2,
        ignore=ignore,
        dirs_exist_ok=False,
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

def make_tree_writable(path: Path) -> None:
    if not path.exists():
        return

    write_bits = stat.S_IWUSR | stat.S_IWGRP | stat.S_IWOTH
    for root, dirs, files in os.walk(path):
        root_path = Path(root)
        root_path.chmod(root_path.stat().st_mode | write_bits)
        for directory_name in dirs:
            directory_path = root_path / directory_name
            if not directory_path.is_symlink():
                directory_path.chmod(directory_path.stat().st_mode | write_bits)
        for file_name in files:
            file_path = root_path / file_name
            if not file_path.is_symlink():
                file_path.chmod(file_path.stat().st_mode | write_bits)
