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

    def _run_python(python_file_path: str, kwargs:dict=None):
        """
        A tool used to run a python file
        This tool can run long running python scripts
        Input must be a path to an existing python file
        Returns the command line output of the run
        
        Args:
            python_file_path: A full absolute path to the python file to run
            kwargs: A dictionary of arguments to pass to the python script as command line arguments (Optional). Example : {"--arg1": "value1", "--arg2": "value2"}
        """
        start_time = time.time()
        # validate path is a file
        if not Path(python_file_path).is_file():
            return "python_file_path is not a valid python file path"
        
        #TODO allow to accept arguments + validate they don't break the bash (requiring input etc)
        env_path = runs_dir / agent_id / ".conda" / "envs" / f"{agent_id}_env"
        if kwargs:
            args = " ".join([f"{key} {value}" for key, value in kwargs.items()])
            command = f"conda run -p {env_path} --no-capture-output python {python_file_path} {args}"
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
    )
    return run_python_tool