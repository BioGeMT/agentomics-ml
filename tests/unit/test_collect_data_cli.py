from pathlib import Path

from agentomics.cli.collect_data import build_container_arguments, build_parser


def test_container_arguments_point_the_workflow_to_the_mounted_paths():
    arguments = build_parser().parse_args([
        "--request", "request.json", "--collected-data-dir", "collected", "--provider", "codex", "--model", "gpt-6.1-sol",
    ])

    module_option, _module, *workflow_arguments = build_container_arguments(arguments)
    container_arguments = build_parser().parse_args(workflow_arguments)

    assert module_option == "-m"
    assert container_arguments.request == Path("/request.json")
    assert container_arguments.collected_data_dir == Path("/collected-data")
    assert (container_arguments.provider, container_arguments.model) == ("codex", "gpt-6.1-sol")
