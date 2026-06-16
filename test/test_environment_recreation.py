import shutil
import subprocess
import sys
import tempfile
import unittest
from pathlib import Path
from unittest import mock

REPO_ROOT = Path(__file__).resolve().parents[1]
SRC_PATH = REPO_ROOT / "src"
if str(SRC_PATH) not in sys.path:
    sys.path.insert(0, str(SRC_PATH))

from runtime.conda_utils import (
    create_environment_from_descriptor,
    export_environment_descriptor_to_path,
    remove_environment,
)
from runtime import conda_utils


def _pip_in_env(env_path: Path) -> str:
    return str(env_path / "bin" / "pip")


def _python_in_env(env_path: Path) -> str:
    return str(env_path / "bin" / "python")


def _conda_install(env_path: Path, packages: list[str]) -> None:
    subprocess.run(
        ["conda", "install", "-p", str(env_path), *packages,
         "--yes", "-q", "--override-channels", "-c", "conda-forge"],
        check=True, capture_output=True,
    )


def _pip_install(env_path: Path, packages: list[str]) -> None:
    subprocess.run(
        [_pip_in_env(env_path), "install", *packages],
        check=True, capture_output=True,
    )


def _get_installed_packages(env_path: Path) -> dict[str, str]:
    """Return {package_name: version} for all packages in the env."""
    result = subprocess.run(
        [_pip_in_env(env_path), "list", "--format=freeze"],
        check=True, capture_output=True, text=True,
    )
    packages = {}
    for line in result.stdout.strip().splitlines():
        if "==" in line:
            name, version = line.split("==", 1)
            packages[name.lower()] = version
    return packages


def _create_env(env_path: Path, python_version: str = "3.12") -> None:
    subprocess.run(
        ["conda", "create", "-p", str(env_path), f"python={python_version}", "pip",
         "--yes", "-q", "--override-channels", "-c", "conda-forge"],
        check=True, capture_output=True,
    )


def _assert_packages_importable(test: unittest.TestCase, env_path: Path, import_statements: str) -> None:
    result = subprocess.run(
        [_python_in_env(env_path), "-c", import_statements],
        capture_output=True, text=True,
    )
    test.assertEqual(result.returncode, 0, f"Imports failed in recreated env:\n{result.stderr}")


def _assert_versions_match(
    test: unittest.TestCase,
    source_packages: dict[str, str],
    recreated_packages: dict[str, str],
    package_names: list[str],
) -> None:
    for pkg in package_names:
        test.assertIn(pkg, recreated_packages, f"{pkg} missing from recreated env")
        test.assertEqual(
            source_packages[pkg],
            recreated_packages[pkg],
            f"{pkg} version mismatch: source={source_packages[pkg]}, recreated={recreated_packages[pkg]}",
        )


def _export_destroy_recreate(
    test: unittest.TestCase,
    source_env: Path,
    recreated_env: Path,
    descriptor_path: Path,
) -> None:
    export_environment_descriptor_to_path(env_path=source_env, descriptor_path=descriptor_path)
    test.assertTrue(descriptor_path.exists(), "environment.yml should be created by export")
    remove_environment(source_env)
    create_environment_from_descriptor(descriptor_path=descriptor_path, env_path=recreated_env)
    test.assertTrue(recreated_env.exists(), "Recreated env directory should exist")


class TestEnvironmentDescriptorExport(unittest.TestCase):
    def test_export_strips_pip_local_version_labels(self):
        """PyPI's default index does not accept local version pins like torch==2.11.0+cpu."""
        with tempfile.TemporaryDirectory() as temp_dir:
            root = Path(temp_dir)
            source_env = root / "source_env"
            source_env.mkdir(parents=True)
            descriptor_path = root / "environment.yml"

            with mock.patch.object(
                conda_utils,
                "_collect_environment_packages",
                return_value=(
                    [("python", "3.12.0")],
                    [("torch", "2.11.0+cpu"), ("torchaudio", "2.11.0+cpu")],
                ),
            ):
                export_environment_descriptor_to_path(
                    env_path=source_env, descriptor_path=descriptor_path,
                )

            yml = descriptor_path.read_text(encoding="utf-8")
            self.assertIn("  - pip\n", yml, "pip must be an explicit conda dependency")
            self.assertIn("torch==2.11.0\n", yml)
            self.assertIn("torchaudio==2.11.0\n", yml)
            self.assertNotIn("+cpu", yml)


@unittest.skipUnless(shutil.which("conda"), "conda not available")
class TestEnvironmentRecreation(unittest.TestCase):
    """Integration test: export a conda env to environment.yml, then recreate it from that descriptor."""

    def setUp(self):
        self._temp_dir = tempfile.TemporaryDirectory()
        self.root = Path(self._temp_dir.name)
        self.source_env = self.root / "source_env"
        self.recreated_env = self.root / "recreated_env"

    def tearDown(self):
        remove_environment(self.source_env)
        remove_environment(self.recreated_env)
        self._temp_dir.cleanup()

    def test_mixed_conda_and_pip_packages_survive_export_recreation(self):
        _create_env(self.source_env, python_version="3.11")
        _conda_install(self.source_env, ["numpy", "pyyaml"])
        _pip_install(self.source_env, ["requests", "click"])

        source_packages = _get_installed_packages(self.source_env)
        expected = ["numpy", "pyyaml", "requests", "click"]
        for pkg in expected:
            self.assertIn(pkg, source_packages, f"{pkg} should be installed in source env")

        descriptor_path = self.root / "environment.yml"
        content_before = None

        _export_destroy_recreate(self, self.source_env, self.recreated_env, descriptor_path)
        content_before = descriptor_path.read_text(encoding="utf-8")
        self.assertIn("pip", content_before, "Descriptor should contain a pip section")

        recreated_packages = _get_installed_packages(self.recreated_env)
        _assert_versions_match(self, source_packages, recreated_packages, expected)
        _assert_packages_importable(
            self, self.recreated_env,
            "import numpy; import yaml; import requests; import click; print('ok')",
        )

    def test_graphviz_pip_conda_name_mismatch_survives_recreation(self):
        """graphviz has different names: 'python-graphviz' in conda, 'graphviz' in pip.
        Both refer to the same Python package. Agents may install either way."""
        _create_env(self.source_env, python_version="3.12")
        # Install the system graphviz binary via conda (needed by the Python package)
        # and the Python bindings via pip (as an agent would).
        _conda_install(self.source_env, ["graphviz"])
        _pip_install(self.source_env, ["graphviz"])

        source_packages = _get_installed_packages(self.source_env)
        self.assertIn("graphviz", source_packages)

        descriptor_path = self.root / "environment.yml"
        _export_destroy_recreate(self, self.source_env, self.recreated_env, descriptor_path)

        recreated_packages = _get_installed_packages(self.recreated_env)
        _assert_versions_match(self, source_packages, recreated_packages, ["graphviz"])
        _assert_packages_importable(
            self, self.recreated_env,
            "import graphviz; g = graphviz.Digraph(); g.node('a'); print('ok')",
        )

    def test_graphviz_conda_only_survives_recreation(self):
        """graphviz Python bindings installed purely via conda (as python-graphviz).
        No pip section — the package stays in the conda deps of the export."""
        _create_env(self.source_env, python_version="3.12")
        _conda_install(self.source_env, ["python-graphviz"])

        source_packages = _get_installed_packages(self.source_env)
        self.assertIn("graphviz", source_packages)

        descriptor_path = self.root / "environment.yml"
        _export_destroy_recreate(self, self.source_env, self.recreated_env, descriptor_path)

        content = descriptor_path.read_text(encoding="utf-8")
        self.assertNotIn("- pip:", content, "Conda-only env should have no pip section")

        recreated_packages = _get_installed_packages(self.recreated_env)
        _assert_versions_match(self, source_packages, recreated_packages, ["graphviz"])
        _assert_packages_importable(
            self, self.recreated_env,
            "import graphviz; g = graphviz.Digraph(); g.node('a'); print('ok')",
        )

    def test_heavy_mixed_environment_survives_recreation(self):
        """Realistic agent environment: conda scientific stack + pip ML packages,
        installed in the order agents typically do it (conda base, then pip extras)."""
        _create_env(self.source_env, python_version="3.12")
        _conda_install(self.source_env, [
            "numpy", "pandas", "scikit-learn", "scipy", "pyyaml", "joblib",
        ])
        _pip_install(self.source_env, [
            "catboost", "matplotlib", "plotly", "requests",
            "click", "tqdm", "graphviz",
        ])

        source_packages = _get_installed_packages(self.source_env)
        conda_expected = ["numpy", "pandas", "scikit-learn", "scipy", "pyyaml", "joblib"]
        pip_expected = ["catboost", "matplotlib", "plotly", "requests", "click", "tqdm", "graphviz"]
        all_expected = conda_expected + pip_expected
        for pkg in all_expected:
            self.assertIn(pkg, source_packages, f"{pkg} should be installed in source env")

        descriptor_path = self.root / "environment.yml"
        _export_destroy_recreate(self, self.source_env, self.recreated_env, descriptor_path)

        recreated_packages = _get_installed_packages(self.recreated_env)
        _assert_versions_match(self, source_packages, recreated_packages, all_expected)
        _assert_packages_importable(
            self, self.recreated_env,
            "import numpy; import pandas; import sklearn; import scipy; "
            "import yaml; import joblib; "
            "import catboost; import matplotlib; import plotly; "
            "import requests; import click; import tqdm; import graphviz; "
            "print('ok')",
        )

    def test_pip_before_conda_order_survives_recreation(self):
        """Some agents install pip packages first, then conda packages on top.
        This can confuse the solver if the export doesn't capture state correctly."""
        _create_env(self.source_env, python_version="3.12")
        _pip_install(self.source_env, ["requests", "click", "tqdm"])
        _conda_install(self.source_env, ["numpy", "scipy"])
        _pip_install(self.source_env, ["matplotlib"])

        source_packages = _get_installed_packages(self.source_env)
        expected = ["requests", "click", "tqdm", "numpy", "scipy", "matplotlib"]

        descriptor_path = self.root / "environment.yml"
        _export_destroy_recreate(self, self.source_env, self.recreated_env, descriptor_path)

        recreated_packages = _get_installed_packages(self.recreated_env)
        _assert_versions_match(self, source_packages, recreated_packages, expected)
        _assert_packages_importable(
            self, self.recreated_env,
            "import requests; import click; import tqdm; "
            "import numpy; import scipy; import matplotlib; print('ok')",
        )

    def test_export_uses_pip_names_not_conda_aliases(self):
        """conda env export renames pip packages to conda equivalents (conda#9338).
        Our custom export queries pip directly, so it must use the correct PyPI name."""
        _create_env(self.source_env, python_version="3.12")
        _conda_install(self.source_env, ["graphviz"])
        _pip_install(self.source_env, ["graphviz"])

        descriptor_path = self.root / "environment.yml"
        export_environment_descriptor_to_path(
            env_path=self.source_env, descriptor_path=descriptor_path,
        )
        yml = descriptor_path.read_text(encoding="utf-8")

        self.assertNotIn("python-graphviz", yml)
        self.assertRegex(yml, r"- graphviz==")

    def test_export_captures_actual_version_after_conda_overwrites_pip(self):
        """When conda install replaces a pip package, conda env export reports
        the stale pip version (conda#11452). Our export must capture the real
        version AND the recreated env must have that version."""
        _create_env(self.source_env, python_version="3.12")
        _pip_install(self.source_env, ["numpy==2.0.0"])
        _conda_install(self.source_env, ["numpy"])

        actual_ver = subprocess.run(
            [_python_in_env(self.source_env), "-c", "import numpy; print(numpy.__version__)"],
            capture_output=True, text=True, check=True,
        ).stdout.strip()
        self.assertNotEqual(actual_ver, "2.0.0", "conda should have upgraded numpy")

        descriptor_path = self.root / "environment.yml"
        _export_destroy_recreate(self, self.source_env, self.recreated_env, descriptor_path)

        yml = descriptor_path.read_text(encoding="utf-8")
        self.assertNotIn("numpy==2.0.0", yml, "Stale pip version should not appear")
        self.assertIn(f"numpy={actual_ver}", yml)

        recreated_ver = subprocess.run(
            [_python_in_env(self.recreated_env), "-c", "import numpy; print(numpy.__version__)"],
            capture_output=True, text=True, check=True,
        ).stdout.strip()
        self.assertEqual(recreated_ver, actual_ver,
                         f"Recreated env has numpy {recreated_ver}, expected {actual_ver}")

    def test_underscore_hyphen_packages_survive_recreation(self):
        """Packages with underscores/hyphens in names (e.g. scikit-learn, google_auth)
        must survive export→recreate despite PEP 503 name normalization."""
        _create_env(self.source_env, python_version="3.12")
        _conda_install(self.source_env, ["scikit-learn"])
        _pip_install(self.source_env, ["google-auth"])

        source_packages = _get_installed_packages(self.source_env)
        self.assertIn("scikit-learn", source_packages)
        self.assertIn("google-auth", source_packages)

        descriptor_path = self.root / "environment.yml"
        _export_destroy_recreate(self, self.source_env, self.recreated_env, descriptor_path)

        recreated_packages = _get_installed_packages(self.recreated_env)
        _assert_versions_match(self, source_packages, recreated_packages,
                               ["scikit-learn", "google-auth"])

    def test_export_excludes_phantom_packages_from_base_env(self):
        """Packages visible in the base conda env (e.g. conda itself) must not
        leak into the exported pip deps. pip inspect reads only the target env's
        dist-info directories, so phantoms are excluded by design."""
        _create_env(self.source_env, python_version="3.12")
        _pip_install(self.source_env, ["requests"])

        descriptor_path = self.root / "environment.yml"
        export_environment_descriptor_to_path(
            env_path=self.source_env, descriptor_path=descriptor_path,
        )

        yml = descriptor_path.read_text(encoding="utf-8")
        self.assertNotIn("conda==", yml, "conda itself should not appear in pip deps")
        self.assertIn("requests==", yml, "Real pip-installed package should still be exported")

    def test_pip_upgrade_of_conda_package_not_duplicated_in_export(self):
        """When pip upgrades a conda-managed package, the stale conda-meta entry
        must NOT appear in the conda section of the exported YAML. Otherwise the
        YAML carries two conflicting versions of the same package, which wastes
        time (double install) and can fail if the stale conda version is removed
        from the channel."""
        _create_env(self.source_env, python_version="3.12")
        _conda_install(self.source_env, ["chardet=5.2.0"])
        _pip_install(self.source_env, ["chardet==5.1.0"])

        actual_ver = subprocess.run(
            [_python_in_env(self.source_env), "-c",
             "import chardet; print(chardet.__version__)"],
            capture_output=True, text=True, check=True,
        ).stdout.strip()
        self.assertEqual(actual_ver, "5.1.0")

        descriptor_path = self.root / "environment.yml"
        export_environment_descriptor_to_path(
            env_path=self.source_env, descriptor_path=descriptor_path,
        )
        yml = descriptor_path.read_text(encoding="utf-8")

        self.assertIn("chardet==5.1.0", yml, "pip version should be in pip section")
        self.assertNotIn("chardet=5.2.0", yml,
                         "Stale conda version must not appear when pip owns the package")

        _export_destroy_recreate(self, self.source_env, self.recreated_env, descriptor_path)
        recreated_ver = subprocess.run(
            [_python_in_env(self.recreated_env), "-c",
             "import chardet; print(chardet.__version__)"],
            capture_output=True, text=True, check=True,
        ).stdout.strip()
        self.assertEqual(recreated_ver, "5.1.0",
                         "Recreated env should have the pip-upgraded version")

    def test_create_environment_raises_on_missing_descriptor(self):
        with self.assertRaises(FileNotFoundError):
            create_environment_from_descriptor(
                descriptor_path=self.root / "nonexistent.yml",
                env_path=self.root / "env",
            )


if __name__ == "__main__":
    unittest.main()
