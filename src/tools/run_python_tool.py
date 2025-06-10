from pydantic_ai import Tool
from pathlib import Path
from .bash import ExclusiveBashProcess

def create_run_python_tool(agent_id, workspace_dir, run_mode, timeout, max_retries, proxy, conda_prefix=True):
    bash = ExclusiveBashProcess(
        agent_id=agent_id,
        workspace_dir=workspace_dir,
        run_mode=run_mode,
        autoconda=False,
        timeout=timeout,
        proxy = proxy,
        auto_torch=False,
    )

    def _run_python(python_file_path: str):
        """
        A tool used to run a python file
        This tool can run long running python scripts
        Input must be a path to an existing python file
        Returns the command line output of the run
        
        Args:
            python_file_path: A full absolute path to the python file to run
        """
        # validate path is a file
        if not Path(python_file_path).is_file():
            return "python_file_path is not a valid python file path"
        
        #TODO allow to accept arguments + validate they don't break the bash (requiring input etc)
        if(conda_prefix):
            env_path = workspace_dir / agent_id / ".conda" / "envs" / f"{agent_id}_env"
            command_prefix=f"source /opt/conda/etc/profile.d/conda.sh && conda activate {env_path} && "
            command = command_prefix + f"python {python_file_path}"
        out = bash.run(command)
        if(len(out) > 5000):
            out = out[:5000]+"\n ... (output truncated, too long)"
        return out
    
    run_python_tool = Tool(
        function=_run_python, 
        takes_ctx=False, 
        max_retries=max_retries,
        require_parameter_descriptions=True
    )
    return run_python_tool