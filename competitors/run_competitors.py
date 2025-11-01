
import argparse
import os
import subprocess
import sys
from pathlib import Path

import yaml
from rich import box
from rich.console import Console
from rich.table import Table

HERE = Path(__file__).resolve().parent
CONFIG_PATH = HERE / "config.yaml"
CLONE_DIR = HERE / "biomlbench"
RESULTS_DIR = HERE / "results"
DATA_DIR = HERE / "data"


def load_config():
    with open(CONFIG_PATH, "r") as fh:
        return yaml.safe_load(fh)


def build_env(base, config, agent):
    env = base.copy()
    env.update(
        {
            "OPENROUTER_API_KEY": config["openrouter_key"],
            "OPENROUTER_BASE_URL": config["openrouter_base_url"],
            "OPENROUTER_MODEL": config["model"],
            "BMLB_TIME_LIMIT_SECS": str(config["time_limit_secs"]),
            "BMLB_STEP_LIMIT": str(config["step_limit"]),
        }
    )
    if agent == "biomni":
        env.setdefault("LLM_SOURCE", "Custom")
        env.setdefault("CUSTOM_MODEL_BASE_URL", config["openrouter_base_url"])
        env.setdefault("CUSTOM_MODEL_API_KEY", config["openrouter_key"])
    return env


def run_agent(config, agent, dataset):
    env = build_env(os.environ, config, agent)
    output_subdir = RESULTS_DIR / f"{dataset}_{agent}"
    output_subdir.mkdir(parents=True, exist_ok=True)
    log_file = output_subdir / "run.log"

    cmd = [
        "biomlbench",
        "run-agent",
        "--agent",
        agent,
        "--task-id",
        f"agentomics/{dataset}",
        "--output-dir",
        str(output_subdir),
        "--data-dir",
        str(DATA_DIR),
    ]

    # Stream output live while also saving to log file
    with open(log_file, "w") as f:
        process = subprocess.Popen(
            cmd,
            cwd=CLONE_DIR,
            env=env,
            stdout=subprocess.PIPE,
            stderr=subprocess.STDOUT,
            text=True,
            bufsize=1
        )

        for line in process.stdout:
            print(line, end='')
            f.write(line)

        process.wait()

    # Return a result-like object for compatibility
    class Result:
        def __init__(self, returncode):
            self.returncode = returncode
            self.stdout = ""
            self.stderr = ""

    return Result(process.returncode)


def main():
    parser = argparse.ArgumentParser(description="Run BioMLBench agents on Agentomics datasets")
    parser.add_argument("--agents", nargs="+", help="Agents to run (filters config)")
    parser.add_argument("--datasets", nargs="+", help="Datasets to run (filters config)")
    args = parser.parse_args()

    config = load_config()
    console = Console()
    RESULTS_DIR.mkdir(exist_ok=True)

    agents = [a for a in config["agents"] if not args.agents or a in args.agents]
    datasets = [d for d in config["datasets"] if not args.datasets or d in args.datasets]

    summary = []
    for dataset in datasets:
        for agent in agents:
            console.rule(f"{agent} on {dataset}")
            res = run_agent(config, agent, dataset)
            summary.append((dataset, agent, "success" if res.returncode == 0 else "failed"))

    table = Table(title="Benchmark Summary", box=box.SIMPLE_HEAVY)
    table.add_column("Dataset")
    table.add_column("Agent")
    table.add_column("Status")
    for row in summary:
        table.add_row(*row)
    console.print(table)

    return 0


if __name__ == "__main__":
    sys.exit(main())
