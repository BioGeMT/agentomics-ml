#!/usr/bin/env python3
"""Run Agentomics variants side by side."""

import argparse
import os
import subprocess
import sys
import tomllib
from datetime import datetime
from pathlib import Path

from agentomics.cli.docker_utils import resolve_proxy_env_vars


def prepare_commit(name: str, experiment_dir: Path) -> str:
    commit = subprocess.run(
        ["git", "rev-parse", "--short=12", "--verify", f"{name}^{{commit}}"],
        stdout=subprocess.PIPE,
        text=True,
        check=True,
    ).stdout.strip()
    source_dir = experiment_dir / "source" / commit
    source_dir.mkdir(parents=True, exist_ok=True)
    archive = subprocess.run(["git", "archive", commit], stdout=subprocess.PIPE, check=True).stdout
    subprocess.run(["tar", "-x", "-C", source_dir], input=archive, check=True)

    image = f"agentomics:ab-{commit}"
    if subprocess.run(["docker", "image", "inspect", image], capture_output=True).returncode:
        proxy_arguments = [
            f"--build-arg={variable}={value}"
            for variable, value in resolve_proxy_env_vars().items()
        ]
        subprocess.run(
            ["docker", "build", *proxy_arguments, "--build-arg", "REPOSITORY_SOURCE=.",
             "-t", image, source_dir],
            check=True,
        )
    return commit


def option_arguments(options: dict) -> list[str]:
    arguments = []
    for name, value in options.items():
        if value is True:
            arguments.append(f"--{name}")
        elif value is not False:
            arguments += [f"--{name}", str(value)]
    return arguments


def run(workspace: Path, index: int, commit: str, arguments: list[str]) -> None:
    print(f"Running {workspace.name}", flush=True)
    with open(f"{workspace}.log", "w") as log:
        result = subprocess.run(
            [sys.executable, "-m", "agentomics.cli.run",
             "--image", f"agentomics:ab-{commit}",
             "--workspace-dir", str(workspace),
             *arguments,
             "--tags", f"variant:{index}", f"commit:{commit}"],
            stdout=log,
            stderr=subprocess.STDOUT,
            env=os.environ | {
                "PYTHONUNBUFFERED": "1",
                "PYTHONPATH": str(workspace.parent / "source" / commit / "src"),
                "WANDB_PROJECT_NAME": f"ab-{workspace.parent.name}",
                "WANDB_RUN_GROUP": f"variant-{index}",
            },
        )
    if result.returncode:
        print(f"  Run failed; see {workspace}.log", flush=True)


def main() -> int:
    parser = argparse.ArgumentParser(description="Run Agentomics variants side by side.")
    parser.add_argument("config", type=Path, help="A/B test config file (TOML)")
    config_path = parser.parse_args().config
    config = tomllib.loads(config_path.read_text())
    experiment_dir = Path("outputs/ab_tests", f"{config_path.stem}-{datetime.now():%Y%m%d-%H%M%S}")
    experiment_dir.mkdir(parents=True)
    print(f"Writing runs to {experiment_dir} (W&B project ab-{experiment_dir.name})", flush=True)

    variants = []
    for variant in config["variants"]:
        # Shared options the variant doesn't set are added; the variant's own options win.
        options = config.get("shared", {}) | variant
        commit = prepare_commit(options.pop("commit"), experiment_dir)
        variants.append((commit, option_arguments(options)))

    for replicate in range(1, config.get("replicates", 1) + 1):
        for index, (commit, arguments) in enumerate(variants):
            run(experiment_dir / f"variant_{index}_replicate_{replicate}", index, commit, arguments)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
