import datetime
import os
import re
import shlex
import subprocess
import time
import uuid
from pathlib import Path

from pydantic_ai import Tool


def _filter_agent_env_vars():
    env = {}
    for key, value in os.environ.items():
        if "API_KEY" in key:
            continue
        env[key] = value
    return env


def _sanitize_session_name(session_name: str) -> str:
    cleaned = re.sub(r"[^a-zA-Z0-9_-]+", "_", session_name.strip().lower())
    return cleaned or "default"


def _normalize_tools_list(tools_value: str) -> str:
    if not tools_value:
        return ""
    parts = [p.strip() for p in tools_value.split(",") if p.strip()]
    return ",".join(parts)


def create_claude_code_tool(agent_id, runs_dir, max_retries):
    def _claude_session(
        prompt: str,
        session_name: str,
        mode: str = "default",
        max_turns: int = 1,
        max_budget_usd: float = 0.10,
        output_format: str = "text",
        allowed_tools: str = "web_search",
        disallowed_tools: str = "",
        dangerously_skip_permissions: bool = False,
        timeout_seconds: int = 120,
        max_calls_per_step: int = 2,
    ):
        """
        Use Claude Code CLI for quick analysis or web search in a headless call.
        It can use file/command tools only if you explicitly allow them via allowed_tools
        and are not in plan mode. To allow running commands in headless mode, set
        dangerously_skip_permissions=True and include Bash/Edit/Read in allowed_tools.
        This tool is synchronous and should not be used for long-running tasks or background jobs.
        Keep prompts concise and request short outputs. If it suggests code, implement
        it yourself with write_python/run_python unless you explicitly allow file/exec tools.
        Set session_name to the current step (data_exploration, data_split, data_representation,
        model_architecture, model_training, model_inference, prediction_exploration) to keep one
        session per step. Web search is allowed by default.
        Maximum calls per step session are limited to max_calls_per_step (default 2).

        Args:
            prompt: The instruction for Claude Code. Keep it short and specific.
            session_name: A short name for the step session (one session per step).
            mode: "default" or "plan" (plan mode disables tool execution).
            max_turns: Maximum turns for the CLI (keep small for fast responses).
            max_budget_usd: Budget cap for this call.
            output_format: CLI output format, e.g. "text" or "json".
            allowed_tools: Comma-separated tool allowlist. Includes "web_search" by default.
            disallowed_tools: Comma-separated tool denylist (web_search will be removed).
            dangerously_skip_permissions: If True, bypasses Claude Code permission prompts (use with care).
            timeout_seconds: Hard timeout for the CLI call.
            max_calls_per_step: Maximum claude_session calls allowed per session_name.
        """
        start_time = time.time()
        run_dir = runs_dir / agent_id
        outputs_dir = run_dir / "claude_code_outputs"
        outputs_dir.mkdir(parents=True, exist_ok=True)

        home_dir = Path(os.path.expanduser("~"))
        debug_dir = home_dir / ".claude" / "debug"
        debug_dir.mkdir(parents=True, exist_ok=True)

        safe_session_name = _sanitize_session_name(session_name)
        count_path = outputs_dir / f"session_{safe_session_name}_count.txt"
        try:
            current_count = int(count_path.read_text().strip()) if count_path.exists() else 0
        except Exception:
            current_count = 0
        if max_calls_per_step is not None and current_count >= int(max_calls_per_step):
            return (
                "Error: claude_session call limit reached for this step "
                f"(max {int(max_calls_per_step)})."
            )
        count_path.write_text(str(current_count + 1) + "\n")
        session_file = outputs_dir / f"session_{safe_session_name}.txt"
        if session_file.exists():
            session_id = session_file.read_text().strip()
        else:
            session_id = ""
        if not session_id:
            session_id = str(uuid.uuid4())
            session_file.write_text(session_id + "\n")
            session_status = "started"
        else:
            session_status = "continued"

        timestamp = datetime.datetime.utcnow().strftime("%Y%m%d_%H%M%S")
        request_path = outputs_dir / f"{timestamp}_{safe_session_name}_request.md"
        response_path = outputs_dir / f"{timestamp}_{safe_session_name}_response.md"

        guardrail_prefix = (
            "You are a lightweight helper for quick analysis. "
            "Keep output concise. If you provide code, keep it short and avoid long-running tasks. "
            "If web search helps, use it.\n\n"
        )
        full_prompt = guardrail_prefix + prompt
        request_path.write_text(full_prompt)

        args = ["claude", "-p", full_prompt, "--session-id", session_id]
        mode_value = mode.strip().lower()
        if mode_value in {"plan", "acceptedits"}:
            args += ["--permission-mode", "plan" if mode_value == "plan" else "acceptEdits"]
        if max_turns and int(max_turns) > 0:
            args += ["--max-turns", str(int(max_turns))]
        if max_budget_usd is not None:
            args += ["--max-budget-usd", str(max_budget_usd)]
        if output_format:
            args += ["--output-format", output_format]
        if dangerously_skip_permissions:
            args += ["--dangerously-skip-permissions"]

        allowed_tools_value = _normalize_tools_list(allowed_tools)
        disallowed_tools_value = _normalize_tools_list(disallowed_tools)
        if disallowed_tools_value:
            disallowed_tools_list = [t for t in disallowed_tools_value.split(",") if t != "web_search"]
            disallowed_tools_value = ",".join(disallowed_tools_list)
        if allowed_tools_value:
            allowed_list = [t for t in allowed_tools_value.split(",") if t]
            if "web_search" not in allowed_list:
                allowed_list.append("web_search")
            allowed_tools_value = ",".join(allowed_list)

        if allowed_tools_value:
            args += ["--allowedTools", allowed_tools_value]
        if disallowed_tools_value:
            args += ["--disallowedTools", disallowed_tools_value]

        env_path = runs_dir / agent_id / ".conda" / "envs" / f"{agent_id}_env"
        cmd = " ".join(shlex.quote(arg) for arg in args)
        full_cmd = f"conda run -p {shlex.quote(str(env_path))} --no-capture-output {cmd}"

        env = _filter_agent_env_vars()
        try:
            result = subprocess.run(
                full_cmd,
                shell=True,
                executable="/bin/bash",
                cwd=str(run_dir),
                env=env,
                stdout=subprocess.PIPE,
                stderr=subprocess.STDOUT,
                text=True,
                timeout=int(timeout_seconds),
                errors="replace",
            )
            output = result.stdout or ""
            if result.returncode != 0:
                output = f"Command failed with error code {result.returncode}:\n{output}"
        except subprocess.TimeoutExpired:
            output = f"Command timed out after {int(timeout_seconds)} seconds."

        response_path.write_text(output)

        max_chars = 4000
        if len(output) > max_chars:
            output = output[:max_chars] + "\n... (truncated)"
        output_line = "claude_session_output_file: created"
        timer_msg = f"\n[Tool call took {time.time() - start_time:.1f} seconds]"
        return output_line + "\n" + output + timer_msg

    claude_tool = Tool(
        function=_claude_session,
        takes_ctx=False,
        max_retries=max_retries,
        require_parameter_descriptions=True,
        name="claude_session",
        sequential=True,
    )
    return claude_tool
