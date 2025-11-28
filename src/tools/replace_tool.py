import time
import traceback

from pydantic import BaseModel, Field
from pydantic_ai import Tool
from pathlib import Path

class Edit(BaseModel):
    old: str = Field(description="The old string to replace. Can be multi-line.")
    new: str = Field(description="The new string to replace with. Can be multi-line.")
    replace_all: bool = Field(description="Whether to replace all occurrences. If False, replaces only the first occurrence.", default=False)

def create_replace_tool(agent_id, runs_dir, max_retries):
    def _replace(file_path: str, edit: Edit|list[Edit]):
        """
        A tool used to replace specific text in a file
        Only use this tool on text files (e.g. .py, .txt, .md)
        Multi-line strings are supported
        Can specify a single edit or a list of edits in one call
        Prefer this tool over write_python_tool and shell sed command for editing files
        
        Args:
            file_path: A full absolute path to the file to edit
            edit: The edit(s) to apply to the file. You can provide a single edit or a list of edits.
        """
        start_time = time.time()

        if not Path(file_path).is_file():
            return f"File path {file_path} is not valid"

        necessary_prefix = str(runs_dir / agent_id)
        if not str(Path(file_path).resolve()).startswith(necessary_prefix):
            return f"Error: file_path must start with {necessary_prefix}. Provided: {file_path}"
        
        with open(file_path, "r") as f:
            content = f.read()

        edits = edit if isinstance(edit, list) else [edit]
        for i, single_edit in enumerate(edits):
            result = _apply_edit(content, single_edit)
            if result.startswith("Error:"):
                return f"Error on edit {i+1}/{len(edits)}: {result}"
            content = result
        
        if file_path.endswith('.py'):
            try:
                compile(content, file_path, "exec") 
            except SyntaxError as e:
                return f"Error: edits would create invalid Python syntax:\n{traceback.format_exc()}"

        with open(file_path, "w") as f:
            f.write(content)

        timer_msg = f"\n[Tool call took {time.time() - start_time:.1f} seconds]"
        return f"Successfully applied {len(edits)} edit(s) to {file_path}" + timer_msg

    def _apply_edit(content: str, edit: Edit) -> str:
        if edit.old not in content:
            return f"Error: Could not find text to replace in file: {edit.old}."
        
        if edit.replace_all:
            return content.replace(edit.old, edit.new)
        else:
            return content.replace(edit.old, edit.new, 1)
    
    replace_tool = Tool(
        function=_replace, 
        takes_ctx=False, 
        max_retries=max_retries,
        require_parameter_descriptions=True,
        name="replace",
    )
    return replace_tool