from pydantic_ai import Tool
from .bash_helpers import BashProcess

def create_bash_tool(agent_id, timeout, autoconda, max_retries):
    bash = BashProcess(
        agent_id=agent_id,
        autoconda=autoconda,
        strip_newlines = False,
        return_err_output = True,
        persistent = True, # cd will change it for the next command etc... (better for the agent)
        timeout = timeout, #Seconds to wait for a command to finish
    )
    def _bash(command: str):
        """
        A persistent bash. 
        Use this to execute bash commands. 
        Input should be a valid bash command.

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
        return bash.run(command)
  
    bash_tool = Tool(
        function=_bash, 
        takes_ctx=False, 
        max_retries=max_retries,
        # description=None, # Infered from the function docstring
        require_parameter_descriptions=True
    )
    return bash_tool


