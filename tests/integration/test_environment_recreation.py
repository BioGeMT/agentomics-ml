import subprocess
from pathlib import Path

import yaml

from agentomics.runtime.conda_utils import (
    create_environment_from_descriptor,
    export_environment_descriptor_to_path,
)


def test_mixed_conda_and_pip_dependencies_survive_environment_recreation(
    tmp_path: Path,
):
    source_descriptor = tmp_path / "source-environment.yml"
    source_descriptor.write_text(
        "channels:\n"
        "  - conda-forge\n"
        "dependencies:\n"
        "  - python=3.12\n"
        "  - pip\n"
        "  - pyyaml\n"
        "  - chardet=5.2.0\n",
        encoding="utf-8",
    )
    source_environment = tmp_path / "source-environment"
    create_environment_from_descriptor(source_descriptor, source_environment)
    subprocess.run(
        [
            str(source_environment / "bin" / "python"),
            "-c",
            "import chardet; assert chardet.__version__ == '5.2.0'",
        ],
        check=True,
    )
    subprocess.run(
        [
            str(source_environment / "bin" / "python"),
            "-m",
            "pip",
            "install",
            "click==8.1.8",
            "chardet==5.1.0",
        ],
        check=True,
    )
    inspect_dependencies = (
        "from importlib.metadata import version; "
        "import chardet; assert chardet.__version__ == '5.1.0'; "
        "print(version('PyYAML')); print(version('click')); "
        "print(chardet.__version__)"
    )
    source_versions = subprocess.run(
        [str(source_environment / "bin" / "python"), "-c", inspect_dependencies],
        check=True,
        capture_output=True,
        text=True,
    ).stdout
    exported_descriptor = tmp_path / "exported-environment.yml"
    recreated_environment = tmp_path / "recreated-environment"

    export_environment_descriptor_to_path(
        source_environment,
        exported_descriptor,
    )
    descriptor = yaml.safe_load(exported_descriptor.read_text(encoding="utf-8"))
    conda_dependencies = {
        dependency
        for dependency in descriptor["dependencies"]
        if isinstance(dependency, str)
    }
    pip_dependencies = {
        package
        for dependency in descriptor["dependencies"]
        if isinstance(dependency, dict)
        for package in dependency.get("pip", [])
    }
    assert "chardet=5.2.0" not in conda_dependencies
    assert "chardet==5.1.0" in pip_dependencies
    create_environment_from_descriptor(
        exported_descriptor,
        recreated_environment,
    )

    recreated_versions = subprocess.run(
        [
            str(recreated_environment / "bin" / "python"),
            "-c",
            inspect_dependencies,
        ],
        check=True,
        capture_output=True,
        text=True,
    ).stdout
    assert recreated_versions == source_versions
