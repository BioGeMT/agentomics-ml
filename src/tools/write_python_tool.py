from pydantic_ai import Tool
from .bash import ExclusiveBashProcess

def create_write_python_tool(agent_id, timeout, add_code_to_response, max_retries):
    bash = ExclusiveBashProcess(
        agent_id=agent_id,
        autoconda=True,
        timeout=timeout,
        proxy = False,
        auto_torch=False,
    )

    def _write_python(code: str, file_path: str):
        """
        A tool to write python code into a single file.
        Input must be a valid python code and name of the file.

        Examples:
        code : import numpy as np\\nx = np.linspace(0, 10, 100),
        file_path : /workspace/runs/myname/numpy_test.py,

        Args:
            code: A valid python code.
            file_path: A file path to write the code to.
        """
        code = code.replace('"','\\"')
        out_code =  bash.run(f"echo \"{code}\" > {file_path}")
        #Check for syntax errors
        out_syntax = bash.run(f"python -m py_compile {file_path}")

        if(add_code_to_response):
            return out_code + out_syntax
        return out_syntax
    
    write_python_tool = Tool(
        function=_write_python, 
        takes_ctx=False, 
        max_retries=max_retries,
        # description=None, # Infered from the function docstring
        require_parameter_descriptions=True
    )
    return write_python_tool