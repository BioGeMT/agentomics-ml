import time

from pydantic_ai import Tool
from pathlib import Path
from .bash_tool import BashProcess

def create_run_python_tool(agent_id, runs_dir, timeout, max_retries, proxy):
    bash = BashProcess(
        agent_id=agent_id,
        runs_dir=runs_dir,
        autoconda=False,
        timeout=timeout,
        proxy = proxy
    )

    def _run_python(python_file_path: str, args: str = ""):
        """
        A tool used to run a python file with optional command line arguments.
        This tool can run long running python scripts.
        Returns the command line output of the run.
        When training a model, prefer using this tool over bash tool.

        Args:
            python_file_path: A full absolute path to the python file to run. Must be a path to an existing python file.
            args: Command line arguments to pass to the script. Use standard CLI format, e.g. "--epochs 10 --lr 0.001" or "" for no arguments.
        """
        start_time = time.time()
        # validate path is a file
        if not Path(python_file_path).is_file():
            return f"{python_file_path} is not a valid python file path"

        env_path = runs_dir / agent_id / ".conda" / "envs" / f"{agent_id}_env"
        if args and args.strip():
            command = f"conda run -p {env_path} --no-capture-output python {python_file_path} {args.strip()}"
        else:
            command = f"conda run -p {env_path} --no-capture-output python {python_file_path}"
        out = bash.run(command)
        timer_msg = f"\n[Tool call took {time.time() - start_time:.1f} seconds]"
        return out + timer_msg
    
    run_python_tool = Tool(
        function=_run_python, 
        takes_ctx=False, 
        max_retries=max_retries,
        require_parameter_descriptions=True,
        name="run_python",
        sequential=True,
    )
    return run_python_tool