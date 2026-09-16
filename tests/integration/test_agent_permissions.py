import shlex
from pathlib import Path

from agentomics.tools.tool_registry import create_tools
from agentomics.utils.config import Config


def test_bash_commands_run_as_restricted_agent_user(bash_tool):
    result = bash_tool.function("whoami")

    assert result.splitlines()[0] == Config.AGENT_USER


def test_bash_commands_start_in_current_step_directory(
    initialized_run_config,
    bash_tool,
):
    result = bash_tool.function("pwd")

    assert result.splitlines()[0] == str(initialized_run_config.current_step_dir)


def test_current_step_directory_is_writable_by_agent(
    initialized_run_config,
    bash_tool,
):
    artifact_path = initialized_run_config.current_step_dir / "agent-artifact.txt"

    result = bash_tool.function(
        f"printf %s agent-created > {shlex.quote(str(artifact_path))}"
    )

    assert "Command failed" not in result
    assert artifact_path.read_text(encoding="utf-8") == "agent-created"


def test_prior_run_artifacts_are_readable_by_agent(
    initialized_run_config,
    bash_tool,
):
    prior_artifact = (
        initialized_run_config.current_iteration_dir / "prior_step" / "artifact.txt"
    )
    prior_artifact.parent.mkdir()
    prior_artifact.write_text("prior result", encoding="utf-8")

    result = bash_tool.function(f"cat {shlex.quote(str(prior_artifact))}")

    assert result.splitlines()[0] == "prior result"


def test_agent_cannot_write_outside_current_step_directory(
    initialized_run_config,
    bash_tool,
):
    forbidden_path = initialized_run_config.current_iteration_dir / "agent-write.txt"

    result = bash_tool.function(f"touch {shlex.quote(str(forbidden_path))}")

    assert "Command failed" in result
    assert not forbidden_path.exists()


def test_agent_environment_excludes_api_keys(
    initialized_run_config,
    default_agent_environment: Path,
    restricted_agent_workspace: Path,
    monkeypatch,
):
    secret_name = "UNRECOGNIZED_PROVIDER_API_KEY"
    monkeypatch.setenv(secret_name, "test-secret")
    sentinel_name = "AGENTOMICS_PERMISSION_TEST_SENTINEL"
    sentinel_value = "harmless-environment-sentinel"
    monkeypatch.setenv(sentinel_name, sentinel_value)
    bash_tool = create_tools(initialized_run_config, ["bash"])[0]

    result = bash_tool.function("env")
    environment = dict(
        line.split("=", 1) for line in result.splitlines() if "=" in line
    )

    assert environment.get(sentinel_name) == sentinel_value
    assert secret_name not in environment


def test_image_repository_contains_no_agent_readable_dotenv_files(bash_tool):
    """Check image content: any readable .env under /repository is a failure."""
    result = bash_tool.function(
        "if find /repository -type f -name .env -readable -print -quit 2>/dev/null "
        "| grep -q .; "
        "then echo DOTENV_READABLE; else echo DOTENV_UNAVAILABLE; fi"
    )

    assert result.splitlines()[0] == "DOTENV_UNAVAILABLE"


def test_python_scripts_run_in_the_runs_environment(
    initialized_run_config,
    default_agent_environment: Path,
    run_python_tool,
):
    script_path = initialized_run_config.current_step_dir / "show_environment.py"
    script_path.write_text(
        "import sys\nprint(sys.prefix)\n",
        encoding="utf-8",
    )

    result = run_python_tool.function(python_file_path=str(script_path))

    assert result.splitlines()[0] == str(initialized_run_config.shared_environment_path)


def test_python_tool_rejects_scripts_outside_current_iteration(
    initialized_run_config,
):
    outside_script = Path(initialized_run_config.workspace_dir) / "outside.py"
    outside_script.write_text("print('should not run')\n", encoding="utf-8")
    run_python_tool = create_tools(initialized_run_config, ["run_python"])[0]

    result = run_python_tool.function(python_file_path=str(outside_script))

    assert "must be inside the current iteration directory" in result
