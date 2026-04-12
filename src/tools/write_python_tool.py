import traceback
import time
from pydantic_ai import Tool
from pathlib import Path


def create_write_python_tool(config):
    def _write_python(code: str, file_path: str):
        """
        A tool to write python code into a single file.
        Input must be a valid python code and name of the file.
        This tool can be used to overwrite existing files.

        Examples:
        code: "import numpy as np
        x = np.linspace(0, 10, 100)"
        file_path: "/workspace/runs/myname/numpy_test.py"

        Args:
            code: A valid python code.
            file_path: A file path to write the code to.
        """
        start_time = time.time()
        
        # Check if the file_path points to the current step working directory.
        necessary_prefix = str(config.current_step_dir.resolve())
        if not str(Path(file_path).resolve()).startswith(necessary_prefix):
            return f"Error: file_path must start with {necessary_prefix}. Provided: {file_path}"
        
        # Check syntax
        try:
            compile(code, "<string>", "exec") 
        except SyntaxError as e:
            return traceback.format_exc()

        #Check the file_path doesnt need any mkdirs
        if not Path(file_path).parent.exists():
            return f"Error: Directory {Path(file_path).parent} does not exist."
        
        if len(str(file_path)) > 255:
            return f"Error: {file_path} is too long: length {len(str(file_path))} > 255 characters."

        # Write 
        with open(file_path, "w") as f:
            f.write(code)
            timer_msg = f"\n[Tool call took {time.time() - start_time:.1f} seconds]"
            return f"Code syntax OK, written to {file_path}" + timer_msg

    write_python_tool = Tool(
        function=_write_python, 
        takes_ctx=False, 
        max_retries=config.max_tool_retries,
        # description=None, # Infered from the function docstring
        require_parameter_descriptions=True,
        name="write_python",
        sequential=True,
    )
    return write_python_tool
