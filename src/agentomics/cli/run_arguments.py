from __future__ import annotations

import argparse
from pathlib import Path


def create_run_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        description="Run the Agentomics model-development workflow.",
        allow_abbrev=False,
        formatter_class=argparse.ArgumentDefaultsHelpFormatter,
    )
    run_arguments = parser.add_argument_group("run configuration")
    run_arguments.add_argument(
        "--model",
        help="LLM model used by the agent (for example, 'openai/gpt-5.4')",
    )
    run_arguments.add_argument(
        "--iteration-plan-model",
        help="LLM model used to generate the iteration plan; defaults to --model",
    )
    run_arguments.add_argument(
        "--provider",
        help="Provider override when multiple API-key providers are configured",
    )
    run_arguments.add_argument(
        "--dataset",
        help=(
            "Dataset identifier (for example, 'breast_cancer'); may be omitted "
            "when --fork-from-run supplies the dataset"
        ),
    )
    run_arguments.add_argument(
        "--task-type",
        choices=("classification", "regression"),
        help=(
            "Task type used to prepare the selected dataset; when omitted, use "
            "dataset metadata or prompt interactively"
        ),
    )
    run_arguments.add_argument(
        "--val-metric",
        help=(
            "Validation metric to optimize; defaults to AUROC for classification "
            "and MAE for regression"
        ),
    )
    run_arguments.add_argument(
        "--iterations",
        type=int,
        help=(
            "Maximum agent iterations; with --fork-from-run, the value is the "
            "number of additional iterations from the fork point"
        ),
    )
    run_arguments.add_argument(
        "--timeout",
        type=int,
        help=(
            "Maximum run time in seconds; the run stops when this timeout or the "
            "iteration limit is reached"
        ),
    )
    run_arguments.add_argument(
        "--split-timeout",
        type=int,
        help="Seconds after which the agent may no longer change the data split",
    )
    run_arguments.add_argument(
        "--run-python-timeout",
        type=int,
        help=(
            "Timeout in seconds for each run_python tool execution, which limits "
            "individual training runs (workflow default: 21600)"
        ),
    )
    run_arguments.add_argument(
        "--split-allowed-iterations",
        type=int,
        help=(
            "Initial iterations allowed to change the train/validation split; "
            "with --fork-from-run, the value is additional iterations"
        ),
    )
    run_arguments.add_argument(
        "--exploration-iterations",
        type=int,
        help=(
            "Initial iterations focused on baseline and exploratory models; with "
            "--fork-from-run, the value is additional iterations"
        ),
    )
    run_arguments.add_argument(
        "--user-prompt",
        help=(
            "Main goal for the agent; defaults to developing a model that "
            "generalizes well to unseen data"
        ),
    )
    run_arguments.add_argument(
        "--tags",
        nargs="*",
        help="Space-separated tags for Weights & Biases logging",
    )

    fork_arguments = parser.add_argument_group("forking")
    fork_arguments.add_argument(
        "--fork-from-run",
        type=Path,
        help=(
            "Source run workspace to branch from; omitted run arguments inherit "
            "values from the source configuration"
        ),
    )
    fork_arguments.add_argument(
        "--fork-from-step",
        help=(
            "Checkpoint step to use with --fork-from-run; defaults to the latest "
            "completed step or iteration-end checkpoint"
        ),
    )
    fork_arguments.add_argument(
        "--fork-from-iteration",
        type=int,
        help=(
            "Checkpoint iteration to use with --fork-from-run; defaults to the "
            "latest iteration containing the selected checkpoint"
        ),
    )

    operational_arguments = parser.add_argument_group("operational options")
    operational_arguments.add_argument(
        "--use-provisioning-key",
        action="store_true",
        help=(
            "Create a temporary OpenRouter API key from the provisioning key and "
            "log its cost"
        ),
    )
    operational_arguments.add_argument(
        "--spend-limit",
        type=int,
        default=10,
        help="Spend limit for the temporary OpenRouter key",
    )
    operational_arguments.add_argument(
        "--test",
        action="store_true",
        help="Run the project's integrated test suite instead of an agent run",
    )
    operational_arguments.add_argument(
        "--cpu-only",
        action="store_true",
        help="Disable GPU access",
    )
    operational_arguments.add_argument(
        "--disable-training-reporting",
        action="store_true",
        help="Disable structured best-effort updates from TrainingReporter",
    )
    operational_arguments.add_argument(
        "--conda-export-mode",
        choices=("full", "yaml"),
        default="full",
        help=(
            "Best-iteration environment format: 'full' copies the environment for "
            "fast reuse; 'yaml' stores a portable environment definition"
        ),
    )
    operational_arguments.add_argument(
        "--verbosity",
        choices=("summary", "full"),
        default="full",
        help="Amount of agent interaction detail printed during the run",
    )
    operational_arguments.add_argument(
        "--workspace-dir",
        type=Path,
        help="Workspace directory for this run",
    )
    operational_arguments.add_argument(
        "--datasets-dir",
        type=Path,
        help="Directory containing available datasets",
    )

    listing_arguments = parser.add_argument_group("listing options")
    listing_arguments.add_argument(
        "--list-models",
        action="store_true",
        help="List models available through configured providers and exit",
    )
    listing_arguments.add_argument(
        "--list-datasets",
        action="store_true",
        help="List available datasets and exit",
    )
    listing_arguments.add_argument(
        "--list-metrics",
        action="store_true",
        help="List available validation metrics and exit",
    )
    parser.epilog = (
        "API keys configured for Agentomics must be available as environment "
        "variables or in a .env file. Codex authentication is read from "
        "~/.codex. By default, results are written under ./outputs and datasets "
        "are read from ./datasets."
    )
    return parser

