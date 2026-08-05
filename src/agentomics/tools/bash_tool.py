import subprocess
import re
import threading
import shlex
import os
import time

from pydantic_ai import Tool
from agentomics.runtime.conda_utils import get_shared_environment_path
from agentomics.utils.config import Config
from agentomics.utils.text_processing_utils import collapse_repeated_lines, concise_output


class BashProcess:
    def __init__(self, config: Config, autoconda=True, timeout=60):
        self.locked = threading.Lock()
        self.config = config
        self.autoconda = autoconda
        self.timeout = timeout
        self.agent_env = self.filter_agent_env_vars()

        if autoconda:
            self.create_preloaded_conda()
    
    def filter_agent_env_vars(self):
        agent_env = {}
        
        for key, value in os.environ.items():
            if "API_KEY" in key: # don't pass any API keys to the agent
                continue
            agent_env[key] = value

        return agent_env

    def create_preloaded_conda(self):
        conda_env_path = get_shared_environment_path(self.config)
        # Forked runs may already carry a fully initialized shared env.
        # In that case, preserve it instead of unpacking the base start env over it.
        if (conda_env_path / "bin" / "activate").exists():
            return
        start_env_pkg = os.getenv('START_ENV_PKG')
        shared_dir = str(self.config.shared_dir)
        for cmd in [
            f"mkdir -p {conda_env_path}",
            f"tar -xf {start_env_pkg} -C {conda_env_path}",
            f"source {conda_env_path}/bin/activate && conda-unpack",
        ]:
            subprocess.run(
                cmd,
                shell=True,
                executable="/bin/bash",
                cwd=shared_dir,
                env=self.agent_env,
                check=True,
            )
        if self.config.agent_user:
            subprocess.run(
                ["chown", "-R", self.config.agent_user, str(conda_env_path)],
                check=True,
            )

    def run(self, command: str):
        with self.locked: #exclusive bash access
            try:
                run_kwargs = {
                    "shell": True,
                    "executable": "/bin/bash",
                    "timeout": self.timeout,
                    "stdout": subprocess.PIPE,
                    "stderr": subprocess.STDOUT,
                    "text": True,
                    "env": self.agent_env,
                    "errors": "replace",  # handle invalid UTF-8 bytes
                    "cwd": str(self.config.current_step_dir)
                }

                result = subprocess.run(
                    command,
                    **run_kwargs
                )
                output = result.stdout
                if result.stderr:
                    output += result.stderr

                if result.returncode != 0:
                    output = collapse_repeated_lines(output)
                    output = concise_output(output)
                    return f"Command failed with error code {result.returncode}:\n{output}"

                return self.process_output(output, command)
            except subprocess.TimeoutExpired as e:
                msg = f"Command timed out after {self.timeout} seconds: {e}"
                if "python" in command and ".py" in command:
                    msg += "\nYou should use run_python_tool for running python scripts"
                return msg
    
    def process_output(self, output: str, command: str) -> str:
        """
        Remove the echoed command and return a concise version of the output.

        Args:
            output: a process' output string
            command: the executed command
        """
        pattern = re.escape(command) + r"\s*\n"
        output = re.sub(pattern, "", output, count=1)
        output = collapse_repeated_lines(output).strip()
        return concise_output(output)

def create_bash_tool(config: Config):
        bash = BashProcess(
            config=config,
            autoconda=True,
            timeout=config.bash_tool_timeout,
        )

        def _bash(command: str):
            """
            Run a bash command.
            Input should be a valid bash command.
            Do not use sudo commands, as you don't have sudo access.
            Do not use this tool to run python scripts, use the run_python tool instead.

            Examples:
            \"ls\"
            \"ls -la /workspace/shared/splits\"
            \"mkdir test\"
            \"echo "hello world" > test.txt\"
            \"conda install numpy=2.2.2 -y\"
            \"pip list | grep torch\"

            Args:
                command: A valid bash command.
            """  
            start_time = time.time()
            env_path = get_shared_environment_path(config)
            command_parsed = shlex.quote(command)
            command = f"conda run -p {env_path} --no-capture-output bash -c {command_parsed}"
            if config.agent_user:
                command = f"runuser -u {config.agent_user} -- {command}"
            out = bash.run(command)
            timer_msg = f"\n[Tool call took {time.time() - start_time:.1f} seconds]"
            return out + timer_msg
    
        bash_tool = Tool(
            function=_bash,
            takes_ctx=False,
            max_retries=config.max_tool_retries,
            # description=None, # Inferred from the function docstring
            require_parameter_descriptions=True,
            name="bash",
            sequential=True,
        )

        return bash_tool
