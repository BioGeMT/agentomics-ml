import subprocess
from pathlib import Path

import yaml

from agentomics.runtime.conda_utils import (
    create_environment_from_descriptor,
    export_environment_descriptor_to_path,
)


def test_export_uses_conda_version_for_placeholder_python_distribution(
    tmp_path: Path,
):
    source_descriptor = tmp_path / "source-environment.yml"
    source_descriptor.write_text(
        "channels:\n"
        "  - conda-forge\n"
        "dependencies:\n"
        "  - python=3.12\n"
        "  - conda-pack=0.9.1\n",
        encoding="utf-8",
    )
    source_environment = tmp_path / "source-environment"
    create_environment_from_descriptor(source_descriptor, source_environment)
    exported_descriptor = tmp_path / "exported-environment.yml"

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
    assert "conda-pack=0.9.1" in conda_dependencies
    assert "conda-pack==0.0.0" not in pip_dependencies


def test_export_strips_local_version_from_real_pip_distribution(tmp_path: Path):
    source_descriptor = tmp_path / "source-environment.yml"
    source_descriptor.write_text(
        "channels:\n"
        "  - conda-forge\n"
        "dependencies:\n"
        "  - python=3.12\n"
        "  - pip\n"
        "  - setuptools\n",
        encoding="utf-8",
    )
    source_environment = tmp_path / "source-environment"
    create_environment_from_descriptor(source_descriptor, source_environment)

    package = tmp_path / "local-version-package"
    package.mkdir()
    (package / "pyproject.toml").write_text(
        "[build-system]\n"
        "requires = []\n"
        "build-backend = 'setuptools.build_meta'\n"
        "\n"
        "[project]\n"
        "name = 'agentomics-local-version-package'\n"
        "version = '1.2.3+cpu'\n"
        "\n"
        "[tool.setuptools]\n"
        "py-modules = ['local_version_package']\n",
        encoding="utf-8",
    )
    (package / "local_version_package.py").write_text(
        "PACKAGE_READY = True\n",
        encoding="utf-8",
    )
    subprocess.run(
        [
            str(source_environment / "bin" / "python"),
            "-m",
            "pip",
            "install",
            "--no-build-isolation",
            "--no-deps",
            str(package),
        ],
        check=True,
    )
    exported_descriptor = tmp_path / "exported-environment.yml"

    export_environment_descriptor_to_path(
        source_environment,
        exported_descriptor,
    )

    descriptor = yaml.safe_load(exported_descriptor.read_text(encoding="utf-8"))
    pip_dependencies = {
        dependency
        for section in descriptor["dependencies"]
        if isinstance(section, dict)
        for dependency in section.get("pip", [])
    }
    assert "agentomics-local-version-package==1.2.3" in pip_dependencies
    assert "agentomics-local-version-package==1.2.3+cpu" not in pip_dependencies
