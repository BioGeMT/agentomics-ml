from langchain_core.tools import Tool as LangchainTool
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
- "echo \\"hello world\\" > test.txt"
- "conda create -n my_env python=3.8 matplotlib -c conda-forge -y"
- "source activate my_env"
- "echo -e \\"import numpy as np\nx = np.linspace(0, 10, 100)\nprint('data:',x)\\" > /workspace/numpy_test.py"
    -  wrap your python code in double quotes prefixed with a backslash (\") to allow for correct bash interpretation.
    -  to print in python, never use double quotes (") as they will be interpreted by bash, use only single quotes (') instead.
- "python /workspace/numpy_test.py"
"""

bash = BashProcess(
    strip_newlines = False,
    return_err_output = True,
    persistent = True, # cd will change it for the next command etc... (better for the agent)
    timeout = 60*15, #Seconds to wait for a command to finish
)
#TODO remove langchain dependency
bash_langchain_tool = LangchainTool(
    name = "bash",
    description = bash_tool_desc,
    func = bash.run,
)
bash_tool = Tool.from_langchain(bash_langchain_tool)