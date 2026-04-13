from __future__ import annotations

import time

import weave

from agents.prompts.prompts_utils import build_iteration_base_prompt
from run_logging.logging_helpers import log_files, log_iteration_duration
from runtime.best_run_snapshot import update_best_run_snapshot
from runtime.git_checkpoints import commit_iteration_end
from runtime.read_write_utils import (
    archive_current_iteration,
    get_next_iteration_index,
    initialize_current_iteration_workspace,
    initialize_current_iteration_metadata,
    initialize_current_iteration_state,
    load_iteration_state,
    update_current_iteration_state,
    write_current_iteration_base_prompt,
)
from agents.steps.data_split import DataSplitStep
from agents.steps.step_registry import get_step_sequence
from tools.tool_registry import create_tools
from utils.config import Config
from utils.exceptions import IterationRunFailed
from utils.providers.provider import Provider
from pydantic_ai.models import Model


@weave.op(call_display_name=lambda call: f"Agentomics run - agent_id: {call.inputs['config'].agent_id}")
async def run_agentomics(
    config: Config,
    default_model: Model,
    iteration_plan_model: Model,
    provider: Provider,
) -> None:
    total_iterations = config.iterations
    tools = create_tools(config, config.tool_ids)
    print(f"Starting training loop with {total_iterations} iterations") #TODO for run continuing this is wrong?

    while True:
        iteration = get_next_iteration_index(config)
        if iteration >= total_iterations:
            return
        print(f"\n=== ITERATION {iteration} / {total_iterations - 1} ===")
        _prepare_iteration_workspace(
            config=config,
            iteration=iteration,
        )
        steps = [step_cls(config, default_model, iteration_plan_model, provider, tools) for step_cls in get_step_sequence(config)]
        try:
            for step in steps:
                await step.run()
            update_current_iteration_state(config, ended_at=time.time(), status="success")
        except IterationRunFailed:
            for step in steps:
                step.on_iteration_fail(iteration)
            update_current_iteration_state(config, ended_at=time.time(), status="failed")

        for step in steps:
            step.on_iteration_end(iteration)

        iteration_state = load_iteration_state(config.current_iteration_dir)
        iteration_duration = float(iteration_state["ended_at"]) - float(iteration_state["started_at"])
        log_iteration_duration(iteration=iteration, duration=iteration_duration)
        log_files(config, iteration=iteration)

        archive_current_iteration(config, iteration)
        if iteration_state["status"] == "success":
            update_best_run_snapshot(config=config, iteration=iteration)

        commit_iteration_end(config, iteration=iteration)
        _enforce_time_deadline(config)

def _prepare_iteration_workspace(config, iteration: int) -> None:
    started_at = time.time()
    initialize_current_iteration_workspace(config)
    initialize_current_iteration_metadata(
        config,
        iteration=iteration,
        split_allowed_at_start=DataSplitStep.is_split_allowed(config, iteration),
    )
    initialize_current_iteration_state(config, started_at=started_at)
    write_current_iteration_base_prompt(config, build_iteration_base_prompt(config, iteration))

def _enforce_time_deadline(config) -> None:
    time_deadline = config.time_deadline
    if time_deadline is not None and time.time() >= time_deadline:
        print("Internal timeout reached, stopping the run.")
        raise SystemExit(0)
