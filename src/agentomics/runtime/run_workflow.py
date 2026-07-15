from __future__ import annotations

import argparse
import os
import shutil
import subprocess
import sys
import time
from pathlib import Path

import wandb

from agentomics.cli.run_arguments import add_run_arguments
from agentomics.run_agent_interactive import run_agent_interactive
from agentomics.run_logging.wandb_setup import resume_wandb_run
from agentomics.runtime.filesystem import restore_host_ownership
from agentomics.runtime.read_write_utils import load_config
from agentomics.runtime.setup_fork import fork_run
from agentomics.utils.agent_id import create_agent_id
from agentomics.utils.api_keys import create_new_api_key, delete_api_key, get_api_key_usage
from agentomics.utils.config import Config


def _prepare_codex_credentials() -> None:
    source_directory = Path("/mnt/codex-host")
    if not source_directory.is_dir():
        return
    target_directory = Path("/tmp/codex")
    target_directory.mkdir(parents=True, exist_ok=True)
    for filename in ("auth.json", "models_cache.json"):
        source_path = source_directory / filename
        if source_path.is_file():
            shutil.copy2(source_path, target_directory / filename)
    os.environ["CODEX_AUTH_FILE"] = str(target_directory / "auth.json")
    os.environ["CODEX_MODELS_CACHE_FILE"] = str(
        target_directory / "models_cache.json"
    )


def _resolve_agent_id() -> str:
    return os.environ.get("AGENT_ID") or create_agent_id()


def _configure_runtime_environment(
    arguments: argparse.Namespace,
    agent_id: str,
) -> None:
    os.environ["AGENT_ID"] = agent_id
    os.environ["AGENTOMICS_VERBOSITY"] = arguments.verbosity
    if arguments.cpu_only:
        os.environ["CUDA_VISIBLE_DEVICES"] = ""


def _prepare_fork(
    arguments: argparse.Namespace,
    agent_id: str,
    workspace_directory: Path,
) -> None:
    if arguments.fork_from_run is not None:
        fork_run(
            source_workspace_dir=arguments.fork_from_run,
            target_agent_id=agent_id,
            target_workspace_dir=workspace_directory,
            fork_from_step=arguments.fork_from_step,
            fork_from_iteration=arguments.fork_from_iteration,
        )


def _create_temporary_api_key(spend_limit: int) -> str:
    print(f"Creating temporary API key with spend limit: {spend_limit}")
    temporary_key = create_new_api_key(
        f"agentomics_run_{int(time.time())}",
        spend_limit,
    )
    os.environ["OPENROUTER_API_KEY"] = temporary_key["key"]
    return temporary_key["hash"]


def _finalize_temporary_api_key(
    temporary_key_hash: str,
    workspace_directory: Path,
) -> None:
    config_path = (
        workspace_directory
        / Config.RUN_DIRNAME
        / Config.SHARED_DIRNAME
        / Config.CONFIG_FILENAME
    )
    if config_path.is_file():
        try:
            config = load_config(config_path)
            resume_wandb_run(config, dir="./cleanup_wandb_logs")
            usage = get_api_key_usage(temporary_key_hash)
            wandb.log(
                {
                    "api_usage/limit": usage["limit"],
                    "api_usage/usage": usage["usage"],
                }
            )
            print(
                f"Logged API usage: limit={usage['limit']}, "
                f"usage={usage['usage']}"
            )
        except Exception as error:
            print(
                f"Warning: Failed to log temporary API key usage: {error}",
                file=sys.stderr,
            )
    else:
        print(
            f"Warning: Cannot log temporary API key usage; config not found: "
            f"{config_path}",
            file=sys.stderr,
        )
    try:
        delete_api_key(temporary_key_hash)
    except Exception as error:
        print(
            f"Warning: Failed to delete temporary API key "
            f"{temporary_key_hash}: {error}",
            file=sys.stderr,
        )


def _execute_workflow(
    arguments: argparse.Namespace,
    agent_id: str,
    workspace_directory: Path,
) -> int:
    if arguments.test:
        result = subprocess.run(
            [
                sys.executable,
                "-m",
                "test.run_all_tests",
                "--workspace-dir",
                str(workspace_directory),
            ]
        )
        return result.returncode

    list_mode = (
        arguments.list_models
        or arguments.list_datasets
        or arguments.list_metrics
    )
    exit_code = run_agent_interactive(arguments)

    if list_mode:
        return exit_code

    snapshot_metadata = (
        workspace_directory
        / Config.BEST_ITERATION_SNAPSHOT_DIRNAME
        / Config.RUNTIME_INFO_DIRNAME
        / Config.ITERATION_METADATA_FILENAME
    )
    run_succeeded = snapshot_metadata.is_file()

    if run_succeeded:
        print(f"Run finished. Files can be found in {workspace_directory}")
        print(
            "To run inference, use: "
            f"agentomics-inference --agent-dir {workspace_directory} "
            "--input <input> --output <predictions.csv>"
        )
    else:
        print(
            "Warning: Agent did not produce a valid best iteration snapshot. "
            f"Run artifacts are in {workspace_directory}.",
            file=sys.stderr,
        )
    if exit_code != 0:
        return exit_code
    return 0 if run_succeeded else 1


def run_workflow(arguments: argparse.Namespace) -> int:
    workspace_directory = arguments.workspace_dir
    temporary_key_hash = None
    try:
        agent_id = _resolve_agent_id()
        _configure_runtime_environment(arguments, agent_id)
        _prepare_codex_credentials()
        _prepare_fork(arguments, agent_id, workspace_directory)
        if arguments.use_provisioning_key and not arguments.test:
            temporary_key_hash = _create_temporary_api_key(arguments.spend_limit)
        return _execute_workflow(arguments, agent_id, workspace_directory)
    finally:
        if temporary_key_hash is not None:
            _finalize_temporary_api_key(
                temporary_key_hash,
                workspace_directory,
            )
        restore_host_ownership(workspace_directory)


def main() -> int:
    parser = argparse.ArgumentParser(
        description="Run the Agentomics model-development workflow.",
        allow_abbrev=False,
        formatter_class=argparse.ArgumentDefaultsHelpFormatter,
    )
    add_run_arguments(parser)
    return run_workflow(parser.parse_args())


if __name__ == "__main__":
    sys.exit(main())
