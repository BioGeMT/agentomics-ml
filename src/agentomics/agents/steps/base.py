from __future__ import annotations

import os
import shutil
import subprocess
from abc import ABC, abstractmethod
from datetime import datetime
from pathlib import Path
from collections.abc import Callable
from typing import Any

from pydantic import BaseModel, Field
from pydantic.json_schema import SkipJsonSchema
from pydantic_ai import Agent, ModelRetry, RunContext
from pydantic_ai.messages import ModelMessage
from pydantic_ai.models import Model
from pydantic_ai.settings import ModelSettings

from agentomics.agents.agent_utils import (
    run_agent,
    fabricate_final_result_messages,
    fabricate_initial_prompt_messages,
)
from rich.console import Console
from agentomics.runtime.conda_utils import export_shared_environment_descriptor
from agentomics.runtime.filesystem import chown_tree_to_root, find_new_files_in_dir, rewrite_symlinks_to_absolute
from agentomics.runtime.git_checkpoints import commit_step_checkpoint
from agentomics.runtime.iterations_lessons import read_iterations_lessons, update_iterations_lessons
from agentomics.runtime.read_write_utils import (
    load_current_iteration_base_prompt,
    load_current_iteration_index,
    load_current_iteration_system_prompt,
)
from agentomics.runtime.step_outputs import load_step_outputs, save_step_output
from agentomics.utils.config import Config
from agentomics.utils.exceptions import AgentRunFailed, StepFailed
from agentomics.utils.providers.provider import Provider


_console = Console()

class RuntimeStep(ABC):
    step_id: str
    display_name: str
    output_type: Any = None
    output_filename: str = "output.json"
    failure_is_fatal: bool = True

    @property
    @abstractmethod
    def step_dir(self) -> Path:
        raise NotImplementedError

    async def run(self) -> None:
        if not self.should_run():
            return
        if self.should_be_simulated():
            self.on_step_success(self.build_simulated_output())
            return
        self.on_step_start()
        try:
            output = await self._execute()
        except StepFailed as error:
            await self.on_step_end(error=error)
            if self.failure_is_fatal:
                raise
            _console.print(f"{self.display_name} failed, keeping the iteration")
            output = self.build_failure_output()
        else:
            await self.on_step_end()
        self.on_step_success(output)

    def should_run(self) -> bool:
        return not (self.step_dir / self.output_filename).exists()

    def should_be_simulated(self) -> bool:
        return False

    def build_simulated_output(self) -> Any:
        raise RuntimeError(f"Step '{self.step_id}' does not support being simulated.")

    def build_failure_output(self) -> Any:
        raise RuntimeError(f"Step '{self.step_id}' does not support a failure output.")

    @abstractmethod
    async def _execute(self) -> Any:
        raise NotImplementedError

    def on_step_start(self) -> None:
        return None

    async def on_step_end(self, error: StepFailed | None = None) -> None:
        return None

    def on_step_success(self, output: Any) -> None:
        save_step_output(self.step_dir, self.step_id, output, filename=self.output_filename)

class AgenticStepOutput(BaseModel):
    files_created: SkipJsonSchema[list[str]] = Field(
        default_factory=list,
        description="List of files created during the step. Populated programmatically.",
    )

class AgenticStepMixin(ABC):
    # Provided by the step class that uses this mixin
    step_id: str
    output_type: type[AgenticStepOutput]
    output_filename: str

    @property
    @abstractmethod
    def step_dir(self) -> Path:
        raise NotImplementedError

    def __init__(self, *args: Any, **kwargs: Any) -> None:
        super().__init__(*args, **kwargs)
        self._step_messages: list[ModelMessage] = []

    def injected_scripts(self) -> list[Path]:
        return []

    def setup_injected_scripts(self) -> None:
        scripts = self.injected_scripts()
        if not scripts:
            return
        helpers_dir = self.step_dir / "helpers"
        helpers_dir.mkdir(exist_ok=True)
        (helpers_dir / "__init__.py").touch()
        # In the Docker/agent_user setup, the step directory itself is handed to
        # the agent user, but injected helpers are created here by the runtime
        # afterward and are never chowned to that agent user. That leaves
        # helpers/ importable but non-modifiable for the isolated agent process,
        # so steps can safely depend on helper behavior.
        for source_path in scripts:
            shutil.copy2(source_path, helpers_dir / source_path.name)

    def step_validators(self) -> list[Callable]:
        """Output validators specific to this step.

        A validator must not create, move, or rewrite files, because a later validator can still reject the output.
        Do that in finalize_output().
        """
        return []

    def symlink_allowed_dirs(self, output: AgenticStepOutput) -> list[Path]:
        """Directories in which the agent may create symlinks."""
        return []

    def finalize_output(self, output: AgenticStepOutput) -> AgenticStepOutput:
        """Runs once, on the output that passed all validators. The place to create, move, or rewrite files."""
        return output

    def build_validators(self) -> list[Callable]:
        async def _no_reserved_runtime_files(ctx: RunContext[Any], result: AgenticStepOutput) -> AgenticStepOutput:
            if (self.step_dir / self.output_filename).exists():
                raise ModelRetry(f"The current step directory contains a forbidden file '{self.output_filename}' likely copied from another step. Rename or delete it and retry.")
            return result

        async def _no_symlinks(ctx: RunContext[Any], result: AgenticStepOutput) -> AgenticStepOutput:
            # The agent may return a path with '..' in it. normpath, not resolve(): the paths compared below are intentionally unresolved.
            allowed_dirs = [Path(os.path.normpath(d)) for d in self.symlink_allowed_dirs(result)]
            new_files = find_new_files_in_dir(self.step_dir, self._step_started_at)
            symlinks = [
                f for f in new_files
                if (self.step_dir / f).is_symlink()
                and not any(allowed_dir in (self.step_dir / f).parents for allowed_dir in allowed_dirs)
            ]
            if symlinks:
                raise ModelRetry(
                    f"Your step created symlink(s), which are not allowed: {symlinks}. "
                    "Use actual file copies instead of symbolic links."
                )
            return result
        return [*self.step_validators(), _no_reserved_runtime_files, _no_symlinks]

    def build_deps(self) -> Any:
        return {}

    def create_agent(
        self,
        model: Model,
        model_settings: ModelSettings,
        max_validation_retries: int,
        max_tool_retries: int | None,
        tools: list,
        capabilities: list,
        deps_type: type,
        system_prompt: str | None,
        instructions: str | None,
    ) -> Agent[Any, AgenticStepOutput]:
        agent = Agent(
            model=model,
            system_prompt=system_prompt or (),
            instructions=instructions,
            tools=tools,
            capabilities=capabilities,
            model_settings=model_settings,
            output_type=self.output_type,
            retries=max_validation_retries if max_tool_retries is None else {"output": max_validation_retries, "tools": max_tool_retries},
            deps_type=deps_type,
        )
        for validator in self.build_validators():
            agent.output_validator(validator)
        return agent

    async def run_agent_step(
        self,
        user_prompt: str,
        model: Model,
        model_settings: ModelSettings,
        max_steps: int,
        max_validation_retries: int = 1,
        max_tool_retries: int | None = None,
        tools: list | None = None,
        capabilities: list | None = None,
        deps_type: type = dict,
        message_history: list[ModelMessage] | None = None,
        system_prompt: str | None = None,
        instructions: str | None = None,
    ) -> AgenticStepOutput:
        if system_prompt and message_history:
            raise ValueError("system_prompt is ignored when message_history is non-empty. Set it to None and ensure your message_history contains it, or use instructions.")
        self._step_started_at = datetime.now()
        self._step_messages = []
        agent = self.create_agent(
            model=model,
            model_settings=model_settings,
            max_validation_retries=max_validation_retries,
            max_tool_retries=max_tool_retries,
            tools=tools or [],
            capabilities=capabilities or [],
            deps_type=deps_type,
            system_prompt=system_prompt,
            instructions=instructions,
        )
        try:
            self._step_messages, output = await run_agent(
                agent=agent,
                user_prompt=user_prompt,
                max_steps=max_steps,
                message_history=message_history,
                deps=self.build_deps(),
            )
        except AgentRunFailed as error:
            self._step_messages = error.context_messages
            raise StepFailed(
                message=error.message,
                exception_trace=error.exception_trace,
            ) from error
        output = self.finalize_output(output)
        return output.model_copy(
            update={"files_created": find_new_files_in_dir(self.step_dir, since_timestamp=self._step_started_at)}
        )

class AgentomicsStep(RuntimeStep):
    output_filename = Config.STEP_OUTPUT_FILENAME

    def __init__(
        self,
        config: Config,
        model: Model,
        iteration_plan_model: Model,
        trace_analysis_model: Model,
        provider: Provider,
        tools: list,
    ) -> None:
        super().__init__()
        self.config = config
        self.model = model
        self.iteration_plan_model = iteration_plan_model
        self.trace_analysis_model = trace_analysis_model
        self.provider = provider
        self.tools = tools

    @property
    def step_dir(self) -> Path:
        return self.config.current_step_dir

    @property
    def archive_dir(self) -> Path:
        return self.config.archived_step_dir(self.step_id)

    def should_run(self) -> bool:
        return not (self.archive_dir / self.output_filename).exists()

    def on_step_start(self) -> None:
        super().on_step_start()
        iteration = load_current_iteration_index(self.config)
        _console.print(f"[bold purple]ITERATION {iteration} | {self.display_name} STEP[/bold purple]")
        self.step_dir.mkdir(exist_ok=True)
        subprocess.run(
            ["chown", self.config.AGENT_USER, str(self.step_dir)],
            check=True,
        )

    async def on_step_end(self, error: StepFailed | None = None) -> None:
        await super().on_step_end(error)
        iteration = load_current_iteration_index(self.config)
        status = "FAILED" if error else "SUCCEEDED"
        _console.print(f"[bold purple]ITERATION {iteration} | {self.display_name} STEP {status}[/bold purple]")

    def on_step_success(self, output: Any) -> None:
        super().on_step_success(output)
        export_shared_environment_descriptor(self.config)
        self.archive_step_folder()
        commit_step_checkpoint(self.config, iteration=load_current_iteration_index(self.config), step_id=self.step_id)

    def on_iteration_start(self, iteration: int) -> None:
        return None

    def on_iteration_fail(self, iteration: int) -> None:
        return None

    def on_iteration_end(self, iteration: int) -> None:
        return None
    
    def archive_step_folder(self) -> None:
        step_dir = self.step_dir
        archive_dir = self.archive_dir
        rewrite_symlinks_to_absolute(step_dir)
        step_dir.rename(archive_dir)
        self._rewrite_paths_in_step_output(archive_dir, str(step_dir), str(archive_dir))
        chown_tree_to_root(archive_dir)

    def _rewrite_paths_in_step_output(self, directory: Path, old: str, new: str) -> None:
        output_file = directory / self.output_filename
        if output_file.exists():
            content = output_file.read_text(encoding="utf-8")
            output_file.write_text(content.replace(old, new), encoding="utf-8")

class AgentomicsAgenticStep(AgentomicsStep, AgenticStepMixin):
    def on_step_start(self) -> None:
        super().on_step_start()
        self.setup_injected_scripts()

    async def on_step_end(self, error: StepFailed | None = None) -> None:
        await super().on_step_end(error)
        await update_iterations_lessons(
            config=self.config,
            model=self.trace_analysis_model,
            iteration=load_current_iteration_index(self.config),
            step_id=self.step_id,
            messages=self._step_messages,
            error=error,
        )

    def get_system_prompt(self) -> str:
        return load_current_iteration_system_prompt(self.config)

    def step_prompt(self) -> str:
        raise NotImplementedError(f"{type(self).__name__} must define step_prompt() or override build_user_prompt().")

    def build_user_prompt(self) -> str:
        prompt = self.step_prompt()
        if self.injected_scripts():
            prompt += "\nYour current working directory contains a helpers/ package of read-only modules you can import."
        prompt += "\nSummarized outputs from your previous steps are in previous messages."
        prompt += self._build_iterations_lessons()
        return prompt

    def _build_iterations_lessons(self) -> str:
        iterations_lessons = read_iterations_lessons(self.config)
        if not iterations_lessons:
            return ""
        return (
            "\nLessons established so far in this run.\n"
            f"<iterations_lessons>\n{iterations_lessons}\n</iterations_lessons>"
        )

    def get_initial_message_history(self) -> list[ModelMessage]:
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
        return await self.run_agent_step(
            user_prompt=self.build_user_prompt(),
            model=self.model,
            model_settings=ModelSettings(temperature=self.config.temperature),
            max_steps=self.config.max_steps,
            max_validation_retries=self.config.max_validation_retries,
            tools=self.tools,
            message_history=self.get_initial_message_history(),
        )
