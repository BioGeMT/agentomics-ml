from tools.bash import create_bash_tool
from tools.write_python_tool import create_write_python_tool
from tools.run_python_tool import create_run_python_tool

def create_tools(config):
    tools =[
            create_bash_tool(
                agent_id=config.agent_id,
                workspace_dir=config.workspace_dir,
                run_mode=config.run_mode,
                timeout=config.bash_tool_timeout, 
                autoconda=True,
                max_retries=config.max_tool_retries,
                proxy=config.use_proxy,
                conda_prefix=True,
                auto_torch=False),
            create_write_python_tool( #Tries to create the same-name conda environment
                agent_id=config.agent_id, 
                workspace_dir=config.workspace_dir,
                run_mode=config.run_mode,
                timeout=config.write_python_tool_timeout, 
                add_code_to_response=False,
                max_retries=config.max_tool_retries),
            create_run_python_tool(
                agent_id=config.agent_id,
                workspace_dir=config.workspace_dir,
                run_mode=config.run_mode,
                timeout=config.run_python_tool_timeout,
                proxy=config.use_proxy,
                max_retries=config.max_tool_retries),
        ]
    return tools