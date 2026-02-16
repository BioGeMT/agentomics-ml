#!/usr/bin/env python3
from __future__ import annotations

import argparse
import re
from pathlib import Path

def load_tasks(config_path: Path) -> list[str]:
    lines = config_path.read_text().splitlines()
    datasets: list[str] = []
    in_datasets = False
    item_pattern = re.compile(r"^\s*-\s*(.+?)\s*$")

    for line in lines:
        if not in_datasets:
            if line.strip() == "datasets:":
                in_datasets = True
            continue

        if line and not line.startswith((" ", "\t", "-")):
            break

        item_match = item_pattern.match(line)
        if item_match is None:
            continue
        value = item_match.group(1).split("#", 1)[0].strip().strip("'").strip('"')
        if "/" in value:
            datasets.append(value)

    return datasets


def filter_tasks(tasks: list[str], mode: str) -> list[str]:
    if mode == "all":
        return tasks
    if mode == "proteingym":
        return [task for task in tasks if task.startswith("proteingym-dms/")]
    if mode == "polaris":
        return [task for task in tasks if task.startswith("polarishub/")]
    raise ValueError(f"Unsupported mode: {mode}")


def main() -> None:
    parser = argparse.ArgumentParser(description="List BioMLBench task IDs from competitors/config.yaml")
    parser.add_argument(
        "--config",
        default=str(Path(__file__).resolve().parents[1] / "config.yaml"),
        help="Path to competitors config.yaml",
    )
    parser.add_argument(
        "--mode",
        choices=["all", "proteingym", "polaris"],
        default="all",
        help="Filter task group",
    )
    args = parser.parse_args()

    tasks = load_tasks(Path(args.config).resolve())
    tasks = filter_tasks(tasks, args.mode)
    for task in tasks:
        print(task)


if __name__ == "__main__":
    main()
