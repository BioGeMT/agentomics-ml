import subprocess
import re
import threading
import shlex    
import os
import time

from pydantic_ai import Tool

class BashProcess:
    def __init__(self, agent_id, runs_dir, autoconda=True, timeout=60, proxy=False):
        self.locked = threading.Lock()
        self.agent_id = agent_id
        self.runs_dir = runs_dir
        self.autoconda = autoconda
        self.timeout = timeout
        self.proxy = proxy
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
        conda_env_path = self.runs_dir / self.agent_id / ".conda" / "envs" / f"{self.agent_id}_env"

        self.run(f"mkdir -p {conda_env_path}")
        self.run(f"tar -xzf /opt/agent_start_env.tar.gz -C {conda_env_path}")
        self.run(f"source {conda_env_path}/bin/activate && conda-unpack")

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
                    "errors": "replace"  # handle invalid UTF-8 bytes
                }

                result = subprocess.run(
                    command,
                    **run_kwargs
                )
                output = result.stdout
                if result.stderr:
                    output += result.stderr

                if result.returncode != 0:
                    if(len(output) > 5000):
                        output = f"output truncated, too long, showing first 5000 and last 5000 characters. First 5000:\n{output[:5000]}\n...\nLast 5000:\n{output[-5000:]}"
                    return f"Command failed with error code {result.returncode}:\n{output}"

                return self.process_output(output, command)
            except subprocess.TimeoutExpired as e:
                return f"Command timed out after {self.timeout} seconds: {e}"
    
    def process_output(self, output: str, command: str) -> str:
        """
        Uses regex to remove the command from the output.
        Return only first 5000 output characters.

        Args:
            output: a process' output string
            command: the executed command
        """
        pattern = re.escape(command) + r"\s*\n"
        output = re.sub(pattern, "", output, count=1)
        if(len(output) > 5000):
            output = output[:5000]+"\n ... (output truncated, too long)"
        return output.strip()

def create_bash_tool(agent_id, runs_dir, timeout, max_retries, autoconda=True, proxy=False):
        bash = BashProcess(
            agent_id=agent_id,
            runs_dir=runs_dir,
            autoconda=autoconda,
            timeout=timeout,
            proxy=proxy
        )

        def _bash(command: str):
            """
            A persistent bash. 
            Use this to execute bash commands. 
            Input should be a valid bash command.
            Do not use sudo commands, as you don't have sudo access.
            Do not use this tool to run python scripts, use the run_python tool instead.

            Examples:
            \"ls\"
            \"cd /workspace\"
            \"mkdir test\"
            \"echo "hello world" > test.txt\"
            \"conda create -n my_env python=3.8 matplotlib -c conda-forge -y\"
            \"source activate my_env\"            

            Args:
                command: A valid bash command.
            """  
            start_time = time.time()
            env_path = runs_dir / agent_id / ".conda" / "envs" / f"{agent_id}_env"
            command_parsed = shlex.quote(command)
            command = f"conda run -p {env_path} --no-capture-output bash -c {command_parsed}"
            out = bash.run(command)
            timer_msg = f"\n[Tool call took {time.time() - start_time:.1f} seconds]"
            return out + timer_msg
    
        bash_tool = Tool(
            function=_bash,
            takes_ctx=False,
            max_retries=max_retries,
            # description=None, # Inferred from the function docstring
            require_parameter_descriptions=True,
            name="bash",
        )

        return bash_tool