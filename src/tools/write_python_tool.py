from smolagents import Tool
from .bash_helpers import BashProcess

python_tool_desc = """
A tool to write python code into a single file.
Input must be a valid python code and name of the file.

Examples:
- "code": "import numpy as np\nx = np.linspace(0, 10, 100)\nprint('data:',x)"
- "file_path": "/workspace/runs/myname/numpy_test.py"
"""

class WritePythonTool(Tool):
    name = "write_python"
    description = python_tool_desc
    inputs = {
        "code": {
            "type": "string",
            "description": "A valid python code."
        },
        "file_path": {
            "type": "string",
            "description": f"A file path to write the code to."
        }
    }
    output_type = "string"

    def __init__(self, agent_id, timeout, add_code_to_response):
        self.bash = BashProcess(
            agent_id=agent_id,
            strip_newlines=False,
            return_err_output=True,
            persistent=True, 
            timeout=timeout,
        )
        self.agent_id = agent_id
        self.add_code_to_response = add_code_to_response
        self.args = {'timeout': timeout}
        super().__init__()

    def forward(self, code: str, file_path: str):
        code = code.replace('"','\\"')
        out_code =  self.bash.run(f"echo -e \"{code}\" > {file_path}")
        #Check for syntax errors
        out_syntax = self.bash.run(f"python -m py_compile {file_path}")

        if(self.add_code_to_response):
            return out_code + out_syntax
        return out_syntax
    
def get_write_python_tool(agent_id, timeout, add_code_to_response=False):
    return WritePythonTool(agent_id, timeout, add_code_to_response)

