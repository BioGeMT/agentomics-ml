#!/usr/bin/env python3
"""Validate and return the version of a published Agentomics Release."""

import re
import sys
import tomllib
from pathlib import Path


STABLE_VERSION_PATTERN = re.compile(
    r"^(0|[1-9][0-9]*)\."
    r"(0|[1-9][0-9]*)\."
    r"(0|[1-9][0-9]*)$"
)


def main() -> int:
    if len(sys.argv) != 2:
        print(
            "Usage: validate_release_version.py <release-tag>",
            file=sys.stderr,
        )
        return 2

    release_tag = sys.argv[1]
    with Path("pyproject.toml").open("rb") as pyproject_file:
        project_version = tomllib.load(pyproject_file)["project"]["version"]

    if (
        not STABLE_VERSION_PATTERN.fullmatch(project_version)
        or release_tag != f"v{project_version}"
    ):
        print(
            f"Release tag {release_tag} does not match stable "
            f"project.version {project_version}",
            file=sys.stderr,
        )
        return 1

    print(project_version)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
