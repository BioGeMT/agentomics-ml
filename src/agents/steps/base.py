from __future__ import annotations

import shutil
import subprocess
from abc import ABC, abstractmethod
from datetime import datetime
from pathlib import Path
from typing import Any

from pydantic import BaseModel, Field
from pydantic.json_schema import SkipJsonSchema
from pydantic_ai import Agent, ModelRetry, RunContext
from pydantic_ai.models import Model

from agents.agent_utils import (
    fabricate_final_result_messages,
    fabricate_initial_prompt_messages,
    run_agent,
)
from rich.console import Console
from runtime.conda_utils import export_shared_environment_descriptor
from runtime.filesystem import chown_tree_to_root, rewrite_symlinks_to_absolute
from runtime.git_checkpoints import commit_step_checkpoint
from runtime.read_write_utils import (
    get_new_rundir_files,
    load_current_iteration_base_prompt,
    load_current_iteration_index,
    load_current_iteration_system_prompt,
)
from runtime.step_outputs import load_step_outputs, save_step_output
from utils.config import Config
from utils.providers.provider import Provider


_console = Console()

class RuntimeStep(ABC):
    step_id: str
    display_name: str
    output_type: Any = None

    def __init__(self, config: Config, model: Model, iteration_plan_model: Model, provider: Provider, tools: list) -> None:
        self.config = config
        self.model = model
        self.iteration_plan_model = iteration_plan_model
        self.provider = provider
        self.tools = tools

    async def run(self) -> None:
        if not self.should_run():
            return
        if self.should_be_simulated():
            self.on_step_success(self.build_simulated_output())
            return
        self.on_step_start()
        output = await self._execute()
        self.on_step_success(output)

    def should_run(self) -> bool:
        return not (self.config.current_iteration_dir / self.step_id / Config.STEP_OUTPUT_FILENAME).exists()

    def should_be_simulated(self) -> bool:
        return False

    def build_simulated_output(self) -> Any:
        raise RuntimeError(f"Step '{self.step_id}' does not support being simulated.")

    @abstractmethod
    async def _execute(self) -> Any:
        raise NotImplementedError

    def on_step_start(self) -> None:
        iteration = load_current_iteration_index(self.config)
        _console.print(f"[bold purple]ITERATION {iteration} | {self.display_name} STEP[/bold purple]")
        self.config.current_step_dir.mkdir(exist_ok=True)
        if self.config.agent_user:
            subprocess.run(
                ["chown", self.config.agent_user, str(self.config.current_step_dir)],
                check=True,
            )

    def on_step_success(self, output: Any) -> None:
        save_step_output(self.config, self.step_id, output)
        export_shared_environment_descriptor(self.config)
        self._archive_step_folder()
        commit_step_checkpoint(self.config, iteration=load_current_iteration_index(self.config), step_id=self.step_id)

    def on_iteration_start(self, iteration: int) -> None:
        return None

    def on_iteration_fail(self, iteration: int) -> None:
        return None

    def on_iteration_end(self, iteration: int) -> None:
        return None
    
    def _archive_step_folder(self) -> None:
        step_dir = self.config.current_step_dir
        archived_dir = self.config.archived_step_dir(self.step_id)
        rewrite_symlinks_to_absolute(step_dir)
        step_dir.rename(archived_dir)
        self._rewrite_paths_in_step_output(archived_dir, str(step_dir), str(archived_dir))
        if self.config.agent_user:
            chown_tree_to_root(archived_dir)

    def _rewrite_paths_in_step_output(self, directory: Path, old: str, new: str) -> None:
        output_file = directory / Config.STEP_OUTPUT_FILENAME
        if output_file.exists():
            content = output_file.read_text(encoding="utf-8")
            output_file.write_text(content.replace(old, new), encoding="utf-8")

class AgenticStepOutput(BaseModel):
    files_created: SkipJsonSchema[list[str]] = Field(
        default_factory=list,
        description="List of files created during the step. Populated programmatically.",
    )

class AgenticStep(RuntimeStep):
    output_type: type[AgenticStepOutput]

    def on_step_start(self) -> None:
        super().on_step_start()
        self._setup_injected_scripts()

    def injected_scripts(self) -> list[Path]:
        return []

    def _setup_injected_scripts(self) -> None:
        scripts = self.injected_scripts()
        if not scripts:
            return
        helpers_dir = self.config.current_step_dir / "helpers"
        helpers_dir.mkdir(exist_ok=True)
        (helpers_dir / "__init__.py").touch()
        # In the Docker/agent_user setup, the step directory itself is handed to
        # the agent user, but injected helpers are created here by the runtime
        # afterward and are never chowned to that agent user. That leaves
        # helpers/ importable but non-modifiable for the isolated agent process,
        # so steps can safely depend on helper behavior.
        for source_path in scripts:
            shutil.copy2(source_path, helpers_dir / source_path.name)

    def create_agent(self) -> Agent[dict, AgenticStepOutput]:
        agent = Agent(
            model=self.model,
            system_prompt=self.get_system_prompt(),
            tools=self.tools,
            model_settings=self.provider.get_reasoning_model_settings(
                kwargs={"temperature": self.config.temperature}
            ),
            output_type=self.output_type,
            retries=self.config.max_validation_retries,
            deps_type=dict,
        )
        self.attach_output_validator(agent)
        self._attach_common_validators(agent)
        return agent

    def _attach_common_validators(self, agent: Agent[dict, AgenticStepOutput]) -> None:
        @agent.output_validator
        async def _no_reserved_runtime_files(ctx: RunContext[dict], result: AgenticStepOutput) -> AgenticStepOutput:
            if (self.config.current_step_dir / Config.STEP_OUTPUT_FILENAME).exists():
                raise ModelRetry(f"The current step directory contains a forbidden file '{Config.STEP_OUTPUT_FILENAME}' likely copied from another step. Rename or delete it and retry.")
            return result

        @agent.output_validator
        async def _no_symlinks(ctx: RunContext[dict], result: AgenticStepOutput) -> AgenticStepOutput:
            new_files = get_new_rundir_files(self.config, ctx.deps["start_time"])
            symlinks = [f for f in new_files if (self.config.current_iteration_dir / f).is_symlink()]
            if symlinks:
                raise ModelRetry(
                    f"Your step created symlink(s), which are not allowed: {symlinks}. "
                    "Use actual file copies instead of symbolic links."
                )
            return result

    def get_system_prompt(self) -> str:
        return load_current_iteration_system_prompt(self.config)

    def attach_output_validator(self, agent: Agent[dict, AgenticStepOutput]) -> None:
        return None

    def step_prompt(self) -> str:
        raise NotImplementedError(f"{type(self).__name__} must define step_prompt() or override build_user_prompt().")

    def build_user_prompt(self) -> str:
        prompt = self.step_prompt()
        if self.injected_scripts():
            prompt += "\nYour current working directory contains a helpers/ package of read-only modules you can import."
        prompt += "\nSummarized outputs from your previous steps are in previous messages."
        return prompt

    def build_deps(self, step_started_at: datetime) -> dict[str, Any]:
        return {"start_time": step_started_at}

    def _populate_files_created(self, output: AgenticStepOutput, step_started_at: datetime) -> AgenticStepOutput:
        return output.model_copy(
            update={
                "files_created": get_new_rundir_files(
                    self.config,
                    since_timestamp=step_started_at,
                )
            }
        )

    def get_message_history(self):
        base_prompt = load_current_iteration_base_prompt(self.config)
        history = fabricate_initial_prompt_messages(
            system_prompt=self.get_system_prompt(),
            user_prompt=base_prompt,
        )
        for output in load_step_outputs(self.config):
            history.extend(
                fabricate_final_result_messages(
                    output,
                    model_name=self.config.model_name,
                )
            )
        return history

    async def _execute(self) -> AgenticStepOutput:
        step_started_at = datetime.now()
        message_history = self.get_message_history()
        user_prompt = self.build_user_prompt()
        agent = self.create_agent()
        _, output = await run_agent(
            agent=agent,
            user_prompt=user_prompt,
            max_steps=self.config.max_steps,
            message_history=message_history,
            deps=self.build_deps(step_started_at),
        )
        return self._populate_files_created(output, step_started_at)
