from smolagents import Tool
from run_logging.evaluate_log_run import dry_run_evaluate_log_run

submit_check_tool_desc = """
A tool to check your submission before running final_answer tool.
You must run this tool before running the final_answer tool.

Returns:
Error code and information about the submission check. If it's 0, then the submission is valid.
You are not allowed to call final_answer unless your submission is valid.
"""

class SubmitCheckTool(Tool):
    name = "submit_check"
    description = submit_check_tool_desc
    inputs = {
    }
    output_type = "string"

    def __init__(self, config):
        self.config = config
        super().__init__()

    def forward(self):
        return dry_run_evaluate_log_run(self.config)