import argparse
import json
import os
import shutil
import subprocess
import sys
import time
from pathlib import Path
from typing import Iterator

import wandb
import yaml
from dotenv import load_dotenv
from rich import box
from rich.console import Console
from rich.table import Table

os.environ["PROTOCOL_BUFFERS_PYTHON_IMPLEMENTATION"] = "python"
PROJECT_ROOT = Path(__file__).resolve().parents[1]
SRC_DIR = PROJECT_ROOT / "src"
sys.path.insert(0, str(SRC_DIR))

load_dotenv(PROJECT_ROOT / ".env")

from evaluation import INFERENCE_STAGE, evaluate_submission, rerun_inference
from utils.metrics import get_task_to_metrics_names  

HERE = Path(__file__).resolve().parent
CONFIG_PATH = HERE / "config.yaml"
CLONE_DIR = HERE / "biomlbench"
RESULTS_DIR = HERE / "results"
DATA_DIR = HERE / "data"


def load_config() -> dict:
    with open(CONFIG_PATH, "r") as fh:
        return yaml.safe_load(fh)


def build_env(base: dict, config: dict, agent: str) -> dict:
    env = base.copy()
    agent_config = config["agents"][agent]
    env.update(
        {
            "OPENROUTER_API_KEY": config["openrouter_key"],
            "OPENROUTER_BASE_URL": config["openrouter_base_url"],
            "OPENROUTER_MODEL": agent_config["model"],
            "BMLB_TIME_LIMIT_SECS": str(config["time_limit_secs"]),
            "BMLB_STEP_LIMIT": str(config["step_limit"]),
        }
    )
    if agent == "biomni":
        env["LLM_SOURCE"] = "Custom"
        env["CUSTOM_MODEL_BASE_URL"] = config["openrouter_base_url"]
        env["CUSTOM_MODEL_API_KEY"] = config["openrouter_key"]
        env["BIOMNI_SELF_CRITIC"] = str(agent_config["self_critic"]).lower()
        env["BIOMNI_ITERATIONS"] = str(agent_config["iterations"])
    return env


def run_agent(config: dict, agent: str, dataset: str) -> Path:
    env = build_env(os.environ, config, agent)
    timestamp = time.strftime("%Y-%m-%dT%H-%M-%S-%Z", time.gmtime())
    output_subdir = RESULTS_DIR / f"{dataset}_{agent}_{timestamp}"
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

    with open(log_file, "w") as f:
        result = subprocess.run(
            cmd,
            cwd=CLONE_DIR,
            env=env,
            stdout=f,
            stderr=subprocess.STDOUT,
            text=True,
            check=False,
        )

    if result.returncode != 0:
        raise RuntimeError(f"Agent {agent} failed on dataset {dataset} with exit code {result.returncode}")

    return copy_run_artifacts(agent, dataset, output_subdir)


def copy_run_artifacts(agent: str, dataset: str, output_subdir: Path) -> Path:
    runs_root = CLONE_DIR / "runs"
    dataset_full_id = f"agentomics/{dataset}"
    pattern = f"*run-group_{agent}"
    candidates = sorted(
        runs_root.glob(pattern),
        key=lambda p: p.stat().st_mtime,
        reverse=True,
    )

    for candidate in candidates:
        metadata = json.loads((candidate / "metadata.json").read_text())
        if dataset_full_id in metadata["task_ids"]:
            artifact_dir = output_subdir / "run_artifacts"
            if artifact_dir.exists():
                shutil.rmtree(artifact_dir)
            shutil.copytree(candidate, artifact_dir)
            return artifact_dir

    raise FileNotFoundError(f"No run artifacts found for {agent} on {dataset}")


def highlight_metric(metrics: dict[str, float], task_type: str) -> str:
    ordered = get_task_to_metrics_names()[task_type]
    primary = ordered[0]
    return f"{primary}: {metrics[primary]:.4f}"


def iterate_targets(config: dict, args: argparse.Namespace) -> Iterator[tuple[str, str]]:
    agents = [a for a in config["agents"].keys() if not args.agents or a in args.agents]
    datasets = [d for d in config["datasets"] if not args.datasets or d in args.datasets]
    for dataset in datasets:
        for agent in agents:
            yield dataset, agent


def main() -> int:
    parser = argparse.ArgumentParser(description="Run BioMLBench agents on Agentomics datasets")
    parser.add_argument("--agents", nargs="+", help="Agents to run (filters config)")
    parser.add_argument("--datasets", nargs="+", help="Datasets to run (filters config)")
    args = parser.parse_args()

    config = load_config()
    RESULTS_DIR.mkdir(exist_ok=True)

    wandb.login(key=os.environ["WANDB_API_KEY"], anonymous="allow", timeout=5)

    console = Console()
    summary: list[tuple[str, str, str]] = []

    for dataset, agent in iterate_targets(config, args):
        console.rule(f"{agent} on {dataset}")
        try:
            artifact_dir = run_agent(config, agent, dataset)
            output_subdir = RESULTS_DIR / f"{dataset}_{agent}"

            metrics, task_type = evaluate_submission(
                dataset=dataset,
                artifact_root=artifact_dir,
                data_dir=DATA_DIR,
                output_dir=output_subdir,
            )

            inference_stage = rerun_inference(
                dataset=dataset,
                artifact_root=artifact_dir,
                data_dir=DATA_DIR,
                output_dir=output_subdir,
            )
            (output_subdir / "inference_stage.json").write_text(
                json.dumps(
                    {
                        "inference_stage": inference_stage,
                        "inference_stage_id": INFERENCE_STAGE[inference_stage],
                    },
                    indent=2,
                )
            )

            wandb.init(
                project=os.environ["WANDB_PROJECT_NAME"],
                entity=os.environ["WANDB_ENTITY"],
                name=f"{dataset}-{agent}-{json.loads((artifact_dir / 'metadata.json').read_text())['created_at']}",
                config={
                    "dataset": dataset,
                    "agent": agent,
                    "task_type": task_type,
                    "model": config["agents"][agent]["model"],
                },
            )
            payload = {name: float(value) for name, value in metrics.items()}
            payload["inference_stage_id"] = INFERENCE_STAGE[inference_stage]
            wandb.log(payload)
            wandb.finish()

            console.print(f"Metrics: {json.dumps(metrics, indent=2)}")
            console.print(f"Inference stage: {inference_stage}")
            summary.append((dataset, agent, highlight_metric(metrics, task_type)))
        except Exception as e:
            console.print(f"[red]FAILED: {e}[/red]")
            summary.append((dataset, agent, f"FAILED: {str(e)}"))

    table = Table(title="Benchmark Summary", box=box.SIMPLE_HEAVY)
    table.add_column("Dataset")
    table.add_column("Agent")
    table.add_column("Metric")
    for row in summary:
        table.add_row(*row)
    console.print(table)

    return 0


if __name__ == "__main__":
    sys.exit(main())
