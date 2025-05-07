# Adjusted version of this: https://python.langchain.com/api_reference/_modules/langchain_experimental/llm_bash/bash.html#BashProcess
"""Wrapper around subprocess to run commands."""

from __future__ import annotations

import os
import platform
import re
import subprocess
from typing import TYPE_CHECKING, List, Union
from uuid import uuid4


if TYPE_CHECKING:
    import pexpect



class BashProcess:
    """Wrapper for starting subprocesses.

    Uses the python built-in subprocesses.run()
    Persistent processes are **not** available
    on Windows systems, as pexpect makes use of
    Unix pseudoterminals (ptys). MacOS and Linux
    are okay.

    Example:
        .. code-block:: python

            from langchain_community.utilities.bash import BashProcess

            bash = BashProcess(
                strip_newlines = False,
                return_err_output = False,
                persistent = False
            )
            bash.run('echo \'hello world\'')

    """

    strip_newlines: bool = False
    """Whether or not to run .strip() on the output"""
    return_err_output: bool = False
    """Whether or not to return the output of a failed
    command, or just the error message and stacktrace"""
    persistent: bool = False
    """Whether or not to spawn a persistent session
    NOTE: Unavailable for Windows environments"""



    def __init__(
        self,
        agent_id,
        autoconda=True,
        strip_newlines: bool = False,
        return_err_output: bool = False,
        persistent: bool = False,
        timeout: int = 60,
        proxy: bool = False,
        auto_torch: bool = False
    ):
        """
        Initializes with default settings
        """
        self.strip_newlines = strip_newlines
        self.return_err_output = return_err_output
        self.prompt = ""
        self.process = None
        self.timeout = timeout
        self.agent_id = agent_id
        self.autoconda = autoconda
        self.proxy = proxy
        self.auto_torch = auto_torch
        if persistent:
            self.prompt = str(uuid4())
            self.process = self._initialize_persistent_process(self, self.prompt, agent_id)
            if self.proxy:
                self.proxy_setup()
            if(autoconda):
                self.create_conda_env()
                self.activate_conda_env()
            if auto_torch:
                self.install_torch()

    def custom_reset(self):
        self.prompt = str(uuid4())
        self.process = self._initialize_persistent_process(self, self.prompt, self.agent_id)
        if(self.autoconda):
            self.activate_conda_env()

    def create_conda_env(self):
        self.run(
            f"conda create -n {self.agent_id}_env -y"
        )

    def activate_conda_env(self):
        self.run(
            f"source activate {self.agent_id}_env"
        )

    def proxy_setup(self):
        # Add proxy configuration to the new bash environment
        http_proxy = os.environ.get("HTTP_PROXY", "")
        https_proxy = os.environ.get("HTTPS_PROXY", "")
        
        if http_proxy:
            self.run(f"export HTTP_PROXY={http_proxy}")
            self.run(f"export http_proxy={http_proxy}")
            self.run(f"conda config --set proxy_servers.http {http_proxy}")
    
        if https_proxy:
            self.run(f"export HTTPS_PROXY={https_proxy}")
            self.run(f"export https_proxy={https_proxy}")
            self.run(f"conda config --set proxy_servers.https {https_proxy}")

    def install_torch(self):
        self.run(
            "pip3 install torch torchvision torchaudio"
        )

        # check if installation went well
        self.run(
            "python -c \"import torch; print('PyTorch installed successfully. CUDA available:', torch.cuda.is_available())\""
        )

    @staticmethod
    def _lazy_import_pexpect() -> pexpect:
        """Import pexpect only when needed."""
        if platform.system() == "Windows":
            raise ValueError(
                "Persistent bash processes are not yet supported on Windows."
            )
        try:
            import pexpect

        except ImportError:
            raise ImportError(
                "pexpect required for persistent bash processes."
                " To install, run `pip install pexpect`."
            )
        return pexpect

    @staticmethod
    def _initialize_persistent_process(self: BashProcess, prompt: str, agent_id: str) -> pexpect.spawn:
        # Start bash in a clean environment
        # Doesn't work on windows
        """
        Initializes a persistent bash setting.
        NOTE: Unavailable on Windows

        Args:
            Prompt(str): the bash command to execute
        """
        pexpect = self._lazy_import_pexpect()
        process = pexpect.spawn(
            "sudo", ["-u", agent_id, "bash"], encoding="utf-8"
        )
        process.sendline("export PATH=/opt/conda/bin:$PATH")

        # Set the custom prompt
        process.sendline("PS1=" + prompt)
        process.expect_exact(prompt, timeout=self.timeout)
        return process


    def run(self, commands: Union[str, List[str]]) -> str:
        """
        Run commands in either an existing persistent
        subprocess or on in a new subprocess environment.

        Args:
            commands(List[str]): a list of commands to
                execute in the session
        """
        if isinstance(commands, str):
            commands = [commands]
        commands = ";".join(commands)
        if self.process is not None:
            return self._run_persistent(
                commands,
            )
        else:
            return self._run(commands)


    def _run(self, command: str) -> str:
        """
        Runs a command in a subprocess and returns
        the output.

        Args:
            command: The command to run
        """
        try:
            output = subprocess.run(
                command,
                shell=True,
                check=True,
                stdout=subprocess.PIPE,
                stderr=subprocess.STDOUT,
            ).stdout.decode()
        except subprocess.CalledProcessError as error:
            if self.return_err_output:
                return error.stdout.decode()
            return str(error)
        if self.strip_newlines:
            output = output.strip()
        return output


    def process_output(self, output: str, command: str) -> str:
        """
        Uses regex to remove the command from the output

        Args:
            output: a process' output string
            command: the executed command
        """
        pattern = re.escape(command) + r"\s*\n"
        output = re.sub(pattern, "", output, count=1)
        return output.strip()


    def _run_persistent(self, command: str) -> str:
        """
        Runs commands in a persistent environment
        and returns the output.

        Args:
            command: the command to execute
        """
        pexpect = self._lazy_import_pexpect()
        if self.process is None:
            raise ValueError("Process not initialized")
        self.process.sendline(command)

        # Clear the output with an empty string
        self.process.expect(self.prompt, timeout=self.timeout)
        self.process.sendline("")

        try:
            self.process.expect([self.prompt, pexpect.EOF], timeout=self.timeout)
        except pexpect.TIMEOUT:
            self.custom_reset()
            return f"Timeout error while executing command {command}. " + "Resetting bash to it's default state."
        if self.process.after == pexpect.EOF:
            return f"Exited with error status: {self.process.exitstatus}"
        output = self.process.before
        output = self.process_output(output, command)
        if self.strip_newlines:
            return output.strip()
        return output