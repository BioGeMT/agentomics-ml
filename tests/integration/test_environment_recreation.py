import subprocess
from pathlib import Path

import pytest
import yaml

from agentomics.runtime.conda_utils import (
    create_environment_from_descriptor,
    export_environment_descriptor_to_path,
)


def _create_source_environment(tmp_path: Path, conda_dependencies: list[str]) -> Path:
    source_descriptor = tmp_path / "source-environment.yml"
    source_descriptor.write_text(
        yaml.safe_dump(
            {
                "channels": ["conda-forge"],
                "dependencies": ["python=3.12", "pip", *conda_dependencies],
            },
            sort_keys=False,
        ),
        encoding="utf-8",
    )
    source_environment = tmp_path / "source-environment"
    create_environment_from_descriptor(source_descriptor, source_environment)
    return source_environment


def _imported_module_version(environment: Path, module_name: str) -> str:
    result = subprocess.run(
        [
            str(environment / "bin" / "python"), "-c",
            "import importlib, sys; print(importlib.import_module(sys.argv[1]).__version__)",
            module_name,
        ],
        check=True, capture_output=True, text=True,
    )
    return result.stdout.strip()


@pytest.mark.parametrize(
    ("conda_name", "pip_name", "module_name", "conda_version", "pip_version"),
    [
        pytest.param(
            "chardet",
            "chardet",
            "chardet",
            "5.2.0",
            "5.1.0",
            id="chardet",
        ),
    ],
)
def test_pip_replacement_of_conda_package_survives_environment_recreation(
    tmp_path: Path,
    conda_name: str,
    pip_name: str,
    module_name: str,
    conda_version: str,
    pip_version: str,
):
    conda_requirement = f"{conda_name}={conda_version}"
    pip_requirement = f"{pip_name}=={pip_version}"
    source_environment = _create_source_environment(tmp_path, [conda_requirement])
    assert _imported_module_version(source_environment, module_name) == conda_version
    subprocess.run(
        [
            str(source_environment / "bin" / "python"),
            "-m",
            "pip",
            "install",
            pip_requirement,
        ],
        check=True,
    )
    assert _imported_module_version(source_environment, module_name) == pip_version
    exported_descriptor = tmp_path / "exported-environment.yml"
    export_environment_descriptor_to_path(source_environment, exported_descriptor)
    dependencies = yaml.safe_load(exported_descriptor.read_text(encoding="utf-8"))["dependencies"]
    recreated_environment = tmp_path / "recreated-environment"
    conda_dependencies = {item for item in dependencies if isinstance(item, str)}
    pip_dependencies = {
        package
        for item in dependencies if isinstance(item, dict)
        for package in item.get("pip", [])
    }
    assert conda_requirement not in conda_dependencies
    assert pip_requirement in pip_dependencies
    create_environment_from_descriptor(
        exported_descriptor,
        recreated_environment,
    )

    assert _imported_module_version(recreated_environment, module_name) == pip_version


@pytest.mark.parametrize(
    ("package_name", "module_name", "pip_version", "conda_version"),
    [
        pytest.param("chardet", "chardet", "5.1.0", "5.2.0", id="chardet"),
    ],
)
@pytest.mark.skip(
    reason="Deferred: pip/conda/pip is not captured by a normal export. Fix proposal: agent installs new packages through editing yaml, not conda/pip install commands"
)
def test_pip_conda_pip_installation_order_survives_environment_recreation(
    tmp_path: Path,
    package_name: str,
    module_name: str,
    pip_version: str,
    conda_version: str,
):
    pip_requirement = f"{package_name}=={pip_version}"
    conda_requirement = f"{package_name}={conda_version}"
    source_environment = _create_source_environment(tmp_path, [])
    subprocess.run(
        [str(source_environment / "bin" / "python"), "-m", "pip", "install", pip_requirement],
        check=True,
    )
    assert _imported_module_version(source_environment, module_name) == pip_version

    subprocess.run(
        ["conda", "install", "-p", str(source_environment), "-c", "conda-forge", "-y", "-q", conda_requirement],
        check=True,
    )
    assert _imported_module_version(source_environment, module_name) == conda_version

    # Without --force-reinstall, pip may correctly report the requirement as satisfied without overwriting the conda package.
    subprocess.run(
        [str(source_environment / "bin" / "python"), "-m", "pip", "install", "--force-reinstall", pip_requirement],
        check=True,
    )
    assert _imported_module_version(source_environment, module_name) == pip_version

    exported_descriptor = tmp_path / "exported-environment.yml"
    export_environment_descriptor_to_path(source_environment, exported_descriptor)
    dependencies = yaml.safe_load(exported_descriptor.read_text(encoding="utf-8"))["dependencies"]
    conda_dependencies = {item for item in dependencies if isinstance(item, str)}
    pip_dependencies = {
        package
        for item in dependencies if isinstance(item, dict)
        for package in item.get("pip", [])
    }
    assert conda_requirement not in conda_dependencies
    assert pip_requirement in pip_dependencies

    recreated_environment = tmp_path / "recreated-environment"
    create_environment_from_descriptor(exported_descriptor, recreated_environment)
    assert _imported_module_version(recreated_environment, module_name) == pip_version


@pytest.mark.parametrize(
    ("package_name", "module_name", "pip_version", "conda_version"),
    [
        pytest.param("chardet", "chardet", "5.1.0", "5.2.0", id="chardet"),
    ],
)
def test_conda_replacement_of_pip_package_survives_environment_recreation(
    tmp_path: Path,
    package_name: str,
    module_name: str,
    pip_version: str,
    conda_version: str,
):
    pip_requirement = f"{package_name}=={pip_version}"
    conda_requirement = f"{package_name}={conda_version}"
    source_environment = _create_source_environment(tmp_path, [])
    subprocess.run(
        [str(source_environment / "bin" / "python"), "-m", "pip", "install", pip_requirement],
        check=True,
    )
    assert _imported_module_version(source_environment, module_name) == pip_version

    subprocess.run(
        ["conda", "install", "-p", str(source_environment), "-c", "conda-forge", "-y", "-q", conda_requirement],
        check=True,
    )
    assert _imported_module_version(source_environment, module_name) == conda_version

    exported_descriptor = tmp_path / "exported-environment.yml"
    export_environment_descriptor_to_path(source_environment, exported_descriptor)
    dependencies = yaml.safe_load(exported_descriptor.read_text(encoding="utf-8"))["dependencies"]
    conda_dependencies = {item for item in dependencies if isinstance(item, str)}
    pip_dependencies = {
        package
        for item in dependencies if isinstance(item, dict)
        for package in item.get("pip", [])
    }
    assert conda_requirement in conda_dependencies
    assert not any(package.startswith(f"{package_name}==") for package in pip_dependencies)

    recreated_environment = tmp_path / "recreated-environment"
    create_environment_from_descriptor(exported_descriptor, recreated_environment)
    assert _imported_module_version(recreated_environment, module_name) == conda_version


@pytest.mark.parametrize(
    ("package_name", "module_name", "pip_version"),
    [
        # Conda's graphviz installs the program that renders diagrams (called "dot").
        # pip's graphviz installs Python code that uses that program. They share a
        # package name but provide different pieces, so both must be exported.
        pytest.param(
            "graphviz",
            "graphviz",
            "0.21",
            id="graphviz",
        ),
    ],
)
def test_same_named_conda_and_pip_packages_with_distinct_components_coexist_after_recreation(
    tmp_path: Path,
    package_name: str,
    module_name: str,
    pip_version: str,
):
    source_environment = _create_source_environment(tmp_path, [package_name])
    subprocess.run(
        [
            str(source_environment / "bin" / "python"),
            "-m", "pip", "install", f"{package_name}=={pip_version}",
        ],
        check=True,
    )
    exported_descriptor = tmp_path / "exported-environment.yml"
    export_environment_descriptor_to_path(source_environment, exported_descriptor)
    dependencies = yaml.safe_load(exported_descriptor.read_text(encoding="utf-8"))["dependencies"]
    conda_dependencies = {item for item in dependencies if isinstance(item, str)}
    pip_dependencies = {
        package
        for item in dependencies if isinstance(item, dict)
        for package in item.get("pip", [])
    }
    assert any(item.startswith(f"{package_name}=") for item in conda_dependencies)
    assert f"{package_name}=={pip_version}" in pip_dependencies

    recreated_environment = tmp_path / "recreated-environment"
    create_environment_from_descriptor(exported_descriptor, recreated_environment)
    assert _imported_module_version(recreated_environment, module_name) == pip_version


@pytest.mark.parametrize(
    ("conda_name", "module_name", "expected_version"),
    [
        pytest.param(
            "python-graphviz",
            "graphviz",
            "0.21",
            id="python-graphviz",
        ),
    ],
)
def test_conda_installed_python_package_survives_environment_recreation(
    tmp_path: Path,
    conda_name: str,
    module_name: str,
    expected_version: str,
):
    conda_requirement = f"{conda_name}={expected_version}"
    source_environment = _create_source_environment(tmp_path, [conda_requirement])
    exported_descriptor = tmp_path / "exported-environment.yml"
    export_environment_descriptor_to_path(source_environment, exported_descriptor)
    dependencies = yaml.safe_load(exported_descriptor.read_text(encoding="utf-8"))["dependencies"]
    assert conda_requirement in dependencies
    assert not any(isinstance(item, dict) for item in dependencies)

    recreated_environment = tmp_path / "recreated-environment"
    create_environment_from_descriptor(exported_descriptor, recreated_environment)
    assert _imported_module_version(recreated_environment, module_name) == expected_version


@pytest.mark.parametrize(
    ("pip_name", "module_name", "version"),
    [
        pytest.param("click", "click", "8.1.8", id="click"),
    ],
)
def test_pip_installed_python_package_survives_environment_recreation(
    tmp_path: Path,
    pip_name: str,
    module_name: str,
    version: str,
):
    pip_requirement = f"{pip_name}=={version}"
    source_environment = _create_source_environment(tmp_path, [])
    subprocess.run(
        [str(source_environment / "bin" / "python"), "-m", "pip", "install", pip_requirement],
        check=True,
    )
    assert _imported_module_version(source_environment, module_name) == version

    exported_descriptor = tmp_path / "exported-environment.yml"
    export_environment_descriptor_to_path(source_environment, exported_descriptor)
    dependencies = yaml.safe_load(exported_descriptor.read_text(encoding="utf-8"))["dependencies"]
    conda_dependencies = {item for item in dependencies if isinstance(item, str)}
    pip_dependencies = {
        package
        for item in dependencies if isinstance(item, dict)
        for package in item.get("pip", [])
    }
    assert not any(item.startswith(f"{pip_name}=") for item in conda_dependencies)
    assert pip_requirement in pip_dependencies

    recreated_environment = tmp_path / "recreated-environment"
    create_environment_from_descriptor(exported_descriptor, recreated_environment)
    assert _imported_module_version(recreated_environment, module_name) == version


@pytest.mark.parametrize(
    ("conda_name", "conda_module", "conda_version", "pip_name", "pip_module", "pip_version"),
    [
        pytest.param("pyyaml", "yaml", "6.0.2", "click", "click", "8.1.8", id="pyyaml-and-click"),
    ],
)
def test_unrelated_conda_and_pip_packages_survive_environment_recreation(
    tmp_path: Path,
    conda_name: str,
    conda_module: str,
    conda_version: str,
    pip_name: str,
    pip_module: str,
    pip_version: str,
):
    conda_requirement = f"{conda_name}={conda_version}"
    pip_requirement = f"{pip_name}=={pip_version}"
    source_environment = _create_source_environment(tmp_path, [conda_requirement])
    assert _imported_module_version(source_environment, conda_module) == conda_version
    subprocess.run(
        [str(source_environment / "bin" / "python"), "-m", "pip", "install", pip_requirement],
        check=True,
    )
    assert _imported_module_version(source_environment, pip_module) == pip_version

    exported_descriptor = tmp_path / "exported-environment.yml"
    export_environment_descriptor_to_path(source_environment, exported_descriptor)
    dependencies = yaml.safe_load(exported_descriptor.read_text(encoding="utf-8"))["dependencies"]
    conda_dependencies = {item for item in dependencies if isinstance(item, str)}
    pip_dependencies = {
        package
        for item in dependencies if isinstance(item, dict)
        for package in item.get("pip", [])
    }
    assert conda_requirement in conda_dependencies
    assert pip_requirement in pip_dependencies

    recreated_environment = tmp_path / "recreated-environment"
    create_environment_from_descriptor(exported_descriptor, recreated_environment)
    assert _imported_module_version(recreated_environment, conda_module) == conda_version
    assert _imported_module_version(recreated_environment, pip_module) == pip_version
