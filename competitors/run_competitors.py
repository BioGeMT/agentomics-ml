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

HERE = Path(__file__).resolve().parent
CONFIG_PATH = HERE / "config.yaml"
with open(CONFIG_PATH, "r") as fh:
    _config = yaml.safe_load(fh)
    if _config.get("enable_cost_tracking") and _config.get("provisioning_key"):
        os.environ["PROVISIONING_OPENROUTER_API_KEY"] = _config["provisioning_key"]

from evaluation import (
    INFERENCE_STAGE,
    evaluate_classification_submission,
    evaluate_proteingym_submission,
    get_submission_dir_from_artifacts,
    rerun_inference,
)
from utils.api_keys import create_new_api_key, delete_api_key, get_api_key_usage
from utils.metrics import get_task_to_metrics_names

CLONE_DIR = HERE / "biomlbench"
RESULTS_DIR = HERE / "results"
DATA_DIR = HERE / "data"


def resolve_task_id(dataset: str) -> str:
    return dataset if "/" in dataset else f"agentomics/{dataset}"


def is_agentomics_task(task_id: str) -> bool:
    return task_id.startswith("agentomics/")


def is_proteingym_task(task_id: str) -> bool:
    return task_id.startswith("proteingym-dms/")


def sanitize_dataset_for_path(dataset: str) -> str:
    return dataset.replace("/", "__")


def run_proteingym_postprocess(artifact_dir: Path, task_id: str, data_dir: Path) -> None:
    script_path = HERE / "scripts" / "proteingym_postprocess.py"
    result = subprocess.run(
        [
            sys.executable,
            str(script_path),
            "--artifact-root",
            str(artifact_dir),
            "--task-id",
            task_id,
            "--data-dir",
            str(data_dir),
        ],
        capture_output=True,
        text=True,
        check=False,
    )
    if result.returncode != 0:
        raise RuntimeError(
            f"ProteinGym postprocess failed for task {task_id}\n"
            f"STDOUT:\n{result.stdout}\n"
            f"STDERR:\n{result.stderr}"
        )


def grade_biomlbench_submission(
    task_id: str,
    artifact_dir: Path,
    output_subdir: Path,
    data_dir: Path,
) -> dict:
    submission_dir = get_submission_dir_from_artifacts(artifact_dir)
    submission_path = submission_dir / "submission.csv"
    assert submission_path.is_file(), f"Submission file not found for grading: {submission_path}"

    result = subprocess.run(
        ["biomlbench", "grade-sample", str(submission_path), task_id, "--data-dir", str(data_dir)],
        cwd=CLONE_DIR,
        capture_output=True,
        text=True,
        check=False,
    )
    if result.returncode != 0:
        raise RuntimeError(
            f"biomlbench grade-sample failed for {task_id}\n"
            f"STDOUT:\n{result.stdout}\n"
            f"STDERR:\n{result.stderr}"
        )

    grade_output = result.stdout if result.stdout.strip() else result.stderr
    start = grade_output.find("{")
    end = grade_output.rfind("}")
    assert start != -1 and end != -1 and end > start, (
        "Could not parse JSON payload from biomlbench grade-sample output.\n"
        f"Output:\n{grade_output}"
    )
    grade_dict = json.loads(grade_output[start:end + 1])
    (output_subdir / "grade.json").write_text(json.dumps(grade_dict, indent=2))
    return grade_dict


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


def run_agent(config: dict, agent: str, dataset: str, cpu_only: bool = False):
    timestamp = time.strftime("%Y-%m-%dT%H-%M-%S-%Z", time.gmtime())
    task_id = resolve_task_id(dataset)
    dataset_slug = sanitize_dataset_for_path(dataset)

    key_hash = None
    if config.get("enable_cost_tracking", False):
        config = config.copy()
        key_name = f"{agent}_{dataset_slug}_{timestamp}"
        key_result = create_new_api_key(key_name, config["spending_limit_per_run"])
        key_hash = key_result['hash']
        config["openrouter_key"] = key_result['key']
    else:
        config = config.copy()
        config["openrouter_key"] = config.get("openrouter_key")

    try:
        start_time = time.time()
        env = build_env(os.environ, config, agent)
        output_subdir = RESULTS_DIR / f"{dataset_slug}_{agent}_{timestamp}"
        output_subdir.mkdir(parents=True, exist_ok=True)
        log_file = output_subdir / "run.log"

        cmd = [
            "biomlbench",
            "run-agent",
            "--agent",
            agent,
            "--task-id",
            task_id,
            "--output-dir",
            str(output_subdir),
            "--data-dir",
            str(DATA_DIR),
        ]
        if cpu_only:
            cmd.append("--cpu-only")

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

        duration_seconds = time.time() - start_time

        if result.returncode != 0:
            print("Zero-shot agent failed")
            return None, False, task_id

        (output_subdir / "duration.json").write_text(json.dumps({"duration_seconds": duration_seconds}, indent=2))

        if key_hash:
            usage_data = get_api_key_usage(key_hash)
            (output_subdir / "cost.json").write_text(json.dumps({"cost_usd": usage_data['usage']}, indent=2))

        return copy_run_artifacts(agent, task_id, output_subdir), True, task_id

    finally:
        # ALWAYS cleanup the provisioned key, even on failure
        if key_hash:
            delete_api_key(key_hash)


def copy_run_artifacts(agent: str, task_id: str, output_subdir: Path) -> Path:
    runs_root = CLONE_DIR / "runs"
    pattern = f"*run-group_{agent}"
    candidates = sorted(
        runs_root.glob(pattern),
        key=lambda p: p.stat().st_mtime,
        reverse=True,
    )

    for candidate in candidates:
        metadata = json.loads((candidate / "metadata.json").read_text())
        if task_id in metadata["task_ids"]:
            artifact_dir = output_subdir / "run_artifacts"
            if artifact_dir.exists():
                shutil.rmtree(artifact_dir)
            shutil.copytree(candidate, artifact_dir)
            return artifact_dir

    raise FileNotFoundError(f"No run artifacts found for {agent} on task {task_id}")


def highlight_metric(metrics: dict[str, float], task_type: str) -> str:
    if "biomlbench_score" in metrics and metrics["biomlbench_score"] is not None:
        return f"biomlbench_score: {metrics['biomlbench_score']:.4f}"

    if task_type not in get_task_to_metrics_names():
        first_key = next(iter(metrics.keys()))
        return f"{first_key}: {metrics[first_key]:.4f}"

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
    parser.add_argument("--cpu-only", action="store_true", help="Use CPU-only mode (no GPU)")
    parser.add_argument("--tags", nargs="+", help="Tags to attach to wandb runs")
    args = parser.parse_args()

    config = load_config()
    RESULTS_DIR.mkdir(exist_ok=True)

    wandb.login(key=os.environ["WANDB_API_KEY"], anonymous="allow", timeout=5)

    console = Console()
    summary: list[tuple[str, str, str]] = []

    for dataset, agent in iterate_targets(config, args):
        console.rule(f"{agent} on {dataset}")
        try:
            artifact_dir, success, task_id = run_agent(config, agent, dataset, cpu_only=args.cpu_only)

            if success: 
                output_subdir = artifact_dir.parent  # Use timestamped directory
                grade_dict = None

                if is_agentomics_task(task_id):
                    metrics, task_type = evaluate_classification_submission(
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
                        agent=agent,
                    )
                else:
                    if is_proteingym_task(task_id):
                        if agent == "zeroshot":
                            run_proteingym_postprocess(
                                artifact_dir=artifact_dir,
                                task_id=task_id,
                                data_dir=DATA_DIR,
                            )
                        metrics, task_type = evaluate_proteingym_submission(
                            task_id=task_id,
                            artifact_root=artifact_dir,
                            output_dir=output_subdir,
                            data_dir=DATA_DIR,
                        )
                    else:
                        metrics = {}
                        task_type = "biomlbench"

                    grade_dict = grade_biomlbench_submission(
                        task_id=task_id,
                        artifact_dir=artifact_dir,
                        output_subdir=output_subdir,
                        data_dir=DATA_DIR,
                    )
                    assert "score" in grade_dict, f"Missing score in biomlbench grade output: {grade_dict}"
                    metrics["biomlbench_score"] = float(grade_dict["score"])
                    inference_stage = "matches" if agent == "zeroshot" else "exists"

                (output_subdir / "inference_stage.json").write_text(
                    json.dumps(
                        {
                            "inference_stage": inference_stage,
                            "inference_stage_id": INFERENCE_STAGE[inference_stage],
                        },
                        indent=2,
                    )
                )

                # Load cost data if available
                cost_file = artifact_dir.parent / "cost.json"
                cost_usd = None
                if cost_file.exists():
                    cost_data = json.loads(cost_file.read_text())
                    cost_usd = cost_data.get("cost_usd")

                duration_file = artifact_dir.parent / "duration.json"
                duration_seconds = None
                if duration_file.exists():
                    duration_data = json.loads(duration_file.read_text())
                    duration_seconds = duration_data.get("duration_seconds")

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
                    tags=args.tags or []
                )
                payload = {name: float(value) for name, value in metrics.items()}
                payload["inference_stage_id"] = INFERENCE_STAGE[inference_stage]
                if grade_dict is not None:
                    for key, value in grade_dict.items():
                        if isinstance(value, (int, float)):
                            payload[f"biomlbench/{key}"] = float(value)
                if cost_usd is not None:
                    payload["cost_usd"] = cost_usd
                if duration_seconds is not None:
                    payload["duration_seconds"] = duration_seconds
                wandb.log(payload)
                wandb.finish()

                console.print(f"Metrics: {json.dumps(metrics, indent=2)}")
                console.print(f"Inference stage: {inference_stage}")
                if grade_dict is not None:
                    console.print(f"BioMLBench grade: {json.dumps(grade_dict, indent=2)}")
                summary.append((dataset, agent, highlight_metric(metrics, task_type)))
            else:
                wandb.init(
                    project=os.environ["WANDB_PROJECT_NAME"],
                    entity=os.environ["WANDB_ENTITY"],
                    name=f"{dataset}-{agent}-{time.strftime('%Y-%m-%dT%H-%M-%S-%Z', time.gmtime())}",
                    config={
                        "dataset": dataset,
                        "agent": agent,
                        "task_type": "classification",
                        "model": config["agents"][agent]["model"],
                    },
                    tags=args.tags or []
                )
                if is_agentomics_task(task_id):
                    failure_metrics = {
                        "ACC": None,
                        "AUPRC": None,
                        "AUROC": None,
                        "F1": None,
                        "LOG_LOSS": None,
                        "MCC": None,
                    }
                else:
                    failure_metrics = {"biomlbench_score": None}

                wandb.log(failure_metrics)
                wandb.finish()
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
