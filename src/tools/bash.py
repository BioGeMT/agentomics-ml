from smolagents import Tool
from .bash_helpers import BashProcess

bash_tool_desc = """
A persistent bash.
Use this to execute bash commands.
Input should be a valid bash command.

Examples:
- "ls"
- "cd /workspace"
- "mkdir test"
- "echo 'hello world' > test.txt"
- "conda create -n my_env python=3.8 matplotlib -c conda-forge -y"
- "source activate my_env"
- "python /workspace/numpy_test.py"
"""

class BashTool(Tool):
    name = "bash"
    description = bash_tool_desc
    inputs = {
        "command": {
            "type": "string",
            "description": "A valid bash command to execute."
        }
    }
    output_type = "string"

    def __init__(self, agent_id, timeout, autoconda):
        self.bash = BashProcess(
            agent_id=agent_id,
            autoconda=autoconda,
            strip_newlines=False,
            return_err_output=True,
            persistent=True, 
            timeout=timeout
        )
        self.args = {
            'timeout': timeout,
            'autoconda': autoconda,
        }
        # Calls the constructor of the parent to initialize the metadata
        super().__init__()

    def forward(self, command: str):
        return self.bash.run(command)