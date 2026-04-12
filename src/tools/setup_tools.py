from .bash_tool import create_bash_tool
from .write_python_tool import create_write_python_tool
from .run_python_tool import create_run_python_tool
from .knowledge_tool import create_knowledge_integration_tool
import weave

def create_tools(config):
    print(f"Initializing agent conda environment and tools...")
    tools = [
        create_bash_tool(
            agent_id=config.agent_id,
            runs_dir=config.runs_dir,
            timeout=config.bash_tool_timeout,
            autoconda=True,
            max_retries=config.max_tool_retries,
            proxy=config.use_proxy),
        create_write_python_tool(  # Tries to create the same-name conda environment
            agent_id=config.agent_id,
            max_retries=config.max_tool_retries,
            runs_dir=config.runs_dir),
        create_run_python_tool(
            agent_id=config.agent_id,
            runs_dir=config.runs_dir,
            timeout=config.run_python_tool_timeout,
            proxy=config.use_proxy,
            max_retries=config.max_tool_retries),
    ]

    # On-demand RAG retrieval is only exposed to the agent when knowledge_mode is "rag_od".
    # In "rag" mode, retrieval runs upfront once per run in architecture.py instead.
    if config.knowledge_mode == "rag_od":
        tools.append(create_knowledge_integration_tool(max_retries=config.max_tool_retries))

    # wrap each tool.run with @weave.op for top-level tracing
    for tool in tools:
        tool.run = weave.op(tool.run, call_display_name=tool.name)
    return tools
