from pathlib import Path

from agentomics.cli.collect_data import build_container_arguments, build_mounts, build_parser


def test_container_arguments_point_the_workflow_to_the_mounted_paths():
    arguments = build_parser().parse_args([
        "--request", "request.json", "--collected-data-dir", "collected", "--provider", "codex", "--model", "gpt-6.1-sol",
    ])

    _module_option, _module, *workflow_arguments = build_container_arguments(arguments)
    container_arguments = build_parser().parse_args(workflow_arguments)

    assert build_mounts(Path("/host/request.json"), Path("/host/collected")) == [
        f"type=bind,src=/host/request.json,dst={container_arguments.request},readonly",
        f"type=bind,src=/host/collected,dst={container_arguments.collected_data_dir}",
    ]
    assert (container_arguments.provider, container_arguments.model) == ("codex", "gpt-6.1-sol")
