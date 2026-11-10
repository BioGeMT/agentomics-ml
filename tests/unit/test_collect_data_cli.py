import sys

from agentomics.cli import collect_data as collect_data_cli


def test_command_points_the_workflow_to_the_mounted_request_and_folder(tmp_path, monkeypatch):
    calls = []
    monkeypatch.setattr(collect_data_cli, "run_python_in_docker", lambda **arguments: calls.append(arguments) or 3)
    request_path = tmp_path / "request.json"
    request_path.write_text("{}")
    collected_data_directory = tmp_path / "collected"
    monkeypatch.setattr(sys, "argv", [
        "agentomics-collect-data", "--request", str(request_path), "--collected-data-dir", str(collected_data_directory),
        "--provider", "codex", "--model", "gpt-6.1-sol",
    ])

    exit_code = collect_data_cli.main()

    _module_option, _module, *workflow_arguments = calls[0]["python_arguments"]
    container_arguments = collect_data_cli.build_parser().parse_args(workflow_arguments)
    assert exit_code == 3
    assert f"type=bind,src={request_path},dst={container_arguments.request},readonly" in calls[0]["mounts"]
    assert f"type=bind,src={collected_data_directory},dst={container_arguments.collected_data_dir}" in calls[0]["mounts"]
    assert (container_arguments.provider, container_arguments.model) == ("codex", "gpt-6.1-sol")
