import subprocess
import re
import threading
import shlex    
import os

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
            self.create_conda_env()
    
    def filter_agent_env_vars(self):
        agent_env = {}
        
        for key, value in os.environ.items():
            if "API_KEY" in key: # don't pass any API keys to the agent
                continue
            agent_env[key] = value

        return agent_env

    def create_conda_env(self):
        conda_env_path = self.runs_dir / self.agent_id / ".conda" / "envs" / f"{self.agent_id}_env"
        self.run(
            f"conda create -p {conda_env_path} python=3.9 -y"
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
            # Auto-fix GPU package installations
            original_command = command

            # TensorFlow GPU fix - match pip install with any flags, then tensorflow
            if re.search(r'\b(pip|conda)\s+install.*\btensorflow\b(?!\[and-cuda\])', command):
                # Replace tensorflow with tensorflow[and-cuda], preserving all flags for pip
                if 'pip install' in command:
                    command = re.sub(r'(\bpip\s+install(?:\s+[^\s]+)*\s+)tensorflow\b', r'\1tensorflow[and-cuda]', command)
                elif 'conda install' in command:
                    # For conda, replace entire command with pip
                    command = re.sub(r'conda\s+install.*tensorflow.*', 'pip install tensorflow[and-cuda]', command)

            # PyTorch GPU fix (if using pip/conda without CUDA specification)
            if re.search(r'\b(pip|conda)\s+install.*\btorch\b', command) and 'cu11' not in command and 'cu12' not in command and 'index-url' not in command:
                # For PyTorch, we need to replace the entire package list since we're adding index-url
                command = re.sub(r'(pip\s+install)(\s+[^\n]+)', r'\1 torch torchvision torchaudio --index-url https://download.pytorch.org/whl/cu118', command)
                if 'conda install' in command:
                    command = re.sub(r'conda\s+install.*torch.*', 'pip install torch torchvision torchaudio --index-url https://download.pytorch.org/whl/cu118', command)

            # JAX GPU fix - match pip install with any flags, then jax
            if re.search(r'\bpip\s+install.*\bjax\b(?!\[cuda)', command):
                command = re.sub(r'(\bpip\s+install(?:\s+[^\s]+)*\s+)jax\b', r'\1jax[cuda12]', command)

            if command != original_command:
                print(f"[Auto-fixed GPU package installation]\nOriginal: {original_command}\nFixed: {command}")

            env_path = runs_dir / agent_id / ".conda" / "envs" / f"{agent_id}_env"

            # Ensure conda environment exists (recreate if missing)
            if not env_path.exists():
                print(f"[Conda environment not found, recreating: {env_path}]")
                bash.create_conda_env()

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