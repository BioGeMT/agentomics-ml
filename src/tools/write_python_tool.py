import traceback
from pydantic_ai import Tool
from pathlib import Path

def create_write_python_tool(agent_id, max_retries, runs_dir):
    def _write_python(code: str, file_path: str):
        """
        A tool to write python code into a single file.
        Input must be a valid python code and name of the file.

        Examples:
        code: "import numpy as np
        x = np.linspace(0, 10, 100)"
        file_path: "/workspace/runs/myname/numpy_test.py"

        Args:
            code: A valid python code.
            file_path: A file path to write the code to.
        """
        
        # Check if the file_path points to agents directory (we're not using bash, so we can't check privileges)
        necessary_prefix = str(runs_dir / agent_id)
        if not str(Path(file_path).resolve()).startswith(necessary_prefix):
            return f"Error: file_path must start with {necessary_prefix}. Provided: {file_path}"
        
        # Check syntax
        try:
            compile(code, "<string>", "exec") 
        except SyntaxError as e:
            return traceback.format_exc()

        # Write 
        with open(file_path, "w") as f:
            f.write(code)
            return f"Code syntax OK, written to {file_path}"

    write_python_tool = Tool(
        function=_write_python, 
        takes_ctx=False, 
        max_retries=max_retries,
        # description=None, # Infered from the function docstring
        require_parameter_descriptions=True,
        name="write_python",
    )
    return write_python_tool