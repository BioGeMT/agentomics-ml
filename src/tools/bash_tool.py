import subprocess
import re
import threading
import shlex

from pydantic_ai import Tool

class BashProcess:
    def __init__(self, agent_id, workspace_dir, autoconda=True, timeout=60, proxy=False):
        self.locked = threading.Lock()
        self.agent_id = agent_id
        self.workspace_dir = workspace_dir
        self.autoconda = autoconda
        self.timeout = timeout
        self.proxy = proxy

        if autoconda:
            self.create_conda_env()
    
    def create_conda_env(self):
        conda_env_path = self.workspace_dir / self.agent_id / ".conda" / "envs" / f"{self.agent_id}_env"
        self.run(
            f"conda create -p {conda_env_path} python=3.9 -y"
        )

    def run(self, command: str):
        with self.locked: #exclusive bash access
            try:
                result = subprocess.run(
                    command, 
                    shell=True,
                    executable="/bin/bash",
                    timeout=self.timeout,
                    stdout=subprocess.PIPE,
                    stderr=subprocess.STDOUT,
                    text=True,
                    errors="replace" # handle invalid UTF-8 bytes
                )
                output = result.stdout
                if result.stderr:
                    output += result.stderr

                if result.returncode != 0:
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

def create_bash_tool(agent_id, workspace_dir, timeout, max_retries, autoconda=True, proxy=False):
        bash = BashProcess(
            agent_id=agent_id,
            workspace_dir=workspace_dir,
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

            Examples:
            \"ls\"
            \"cd /workspace\"
            \"mkdir test\"
            \"echo "hello world" > test.txt\"
            \"conda create -n my_env python=3.8 matplotlib -c conda-forge -y\"
            \"source activate my_env\"            
            \"python /workspace/numpy_test.py\"

            Args:
                command: A valid bash command.
            """  
            env_path = workspace_dir / agent_id / ".conda" / "envs" / f"{agent_id}_env"
            command_parsed = shlex.quote(command)
            command = f"conda run -p {env_path} --no-capture-output bash -c {command_parsed}"
            out = bash.run(command)

            return out
    
        bash_tool = Tool(
            function=_bash,
            takes_ctx=False,
            max_retries=max_retries,
            # description=None, # Inferred from the function docstring
            require_parameter_descriptions=True,
            name="bash",
        )

        return bash_tool