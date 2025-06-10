import threading
from pydantic_ai import Tool
from .bash_helpers import BashProcess

class ExclusiveBashProcess:
    
    def __init__(self, agent_id, workspace_dir, run_mode, autoconda, timeout, proxy, auto_torch):
        self.locked = threading.Lock()

        self.bash = BashProcess(
            agent_id=agent_id,
            workspace_dir=workspace_dir,
            run_mode = run_mode,
            autoconda=autoconda,
            strip_newlines = False,
            return_err_output = True,
            persistent = True, # cd will change it for the next command etc... (better for the agent)
            timeout = timeout, #Seconds to wait for a command to finish
            proxy = proxy,
            auto_torch=auto_torch
        )

    def run(self, command: str):
        """"
        Run the bash unless it's already being run, wait until it's finished in that case.
        """
        with self.locked:
            return self.bash.run(command)

def create_bash_tool(agent_id, workspace_dir, run_mode, timeout, autoconda, max_retries, proxy = False, auto_torch=True, conda_prefix=True):
    bash = ExclusiveBashProcess(
        agent_id=agent_id,
        workspace_dir=workspace_dir,
        run_mode=run_mode,
        autoconda=autoconda,
        timeout = timeout, #Seconds to wait for a command to finish
        proxy = proxy,
        auto_torch=auto_torch
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

        if(conda_prefix):
            env_path = workspace_dir / agent_id / ".conda" / "envs" / f"{agent_id}_env"
            command_prefix=f"source /opt/conda/etc/profile.d/conda.sh && conda activate {env_path} && "
            command = command_prefix + command
        out = bash.run(command)
        if(len(out) > 5000):
            out = out[:5000]+"\n ... (output truncated, too long)"
        return out
  
    bash_tool = Tool(
        function=_bash, 
        takes_ctx=False, 
        max_retries=max_retries,
        # description=None, # Infered from the function docstring
        require_parameter_descriptions=True
    )
    return bash_tool


