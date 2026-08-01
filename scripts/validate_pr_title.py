#!/usr/bin/env python3
"""Validate the Release Impact declared by a pull-request title."""

import re
import sys


TITLE_PATTERN = re.compile(
    r"^(?P<type>[a-z]+)(?P<breaking>!)?: "
    r"(?P<description>\S(?:.*\S)?)$"
)
RELEASE_IMPACT_BY_TYPE = {
    "feat": "minor",
    "fix": "patch",
    "perf": "patch",
}
NON_RELEASING_TYPES = {"chore", "ci", "docs", "refactor", "test"}


def main() -> int:
    title = sys.argv[1] if len(sys.argv) == 2 else ""
    match = TITLE_PATTERN.fullmatch(title)
    if match and match.group("type") in RELEASE_IMPACT_BY_TYPE:
        impact = (
            "major"
            if match.group("breaking")
            else RELEASE_IMPACT_BY_TYPE[match.group("type")]
        )
        print(f"Valid PR title. Release Impact: {impact}")
        return 0
    if (
        match
        and match.group("type") in NON_RELEASING_TYPES
        and not match.group("breaking")
    ):
        print("Valid PR title. Release Impact: none")
        return 0

    print(
        """Invalid PR title. Expected: <type>[!]: <description>
See CONTRIBUTING.md for accepted types and examples.""",
        file=sys.stderr,
    )
    return 1


if __name__ == "__main__":
    raise SystemExit(main())
