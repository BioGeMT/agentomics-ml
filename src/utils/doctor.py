"""
Environment and provider pre-flight checks for Agentomics.

This module implements UX-101: fast, safe diagnostic checks that run before
expensive operations like image pulls, Conda environment creation, or provider calls.
"""

import json
import os
import platform
import shutil
import subprocess
import sys
from dataclasses import dataclass, asdict
from enum import Enum
from pathlib import Path
from typing import List, Optional, Dict, Any


class Severity(Enum):
    """Check result severity levels."""
    PASS = "PASS"
    INFO = "INFO"
    WARN = "WARN"
    FAIL = "FAIL"


class OverallStatus(Enum):
    """Overall doctor status."""
    READY = "READY"
    READY_WITH_WARNINGS = "READY WITH WARNINGS"
    NOT_READY = "NOT READY"


@dataclass
class Check:
    """Represents a single diagnostic check result."""
    code: str  # Stable machine-readable identifier
    severity: Severity
    summary: str  # One-line result
    remediation: Optional[str] = None  # Optional next action
    details: Optional[str] = None  # Optional non-secret diagnostic facts

    def to_dict(self) -> Dict[str, Any]:
        """Convert to dictionary for JSON serialization."""
        result = {
            "code": self.code,
            "severity": self.severity.value,
            "summary": self.summary,
        }
        if self.remediation:
            result["remediation"] = self.remediation
        if self.details:
            result["details"] = self.details
        return result


@dataclass
class DoctorReport:
    """Complete diagnostic report."""
    schema_version: int
    deployment_mode: str  # "docker" or "local"
    provider_name: Optional[str]
    privacy_classification: str  # "local", "external", "unknown"
    overall_status: OverallStatus
    checks: List[Check]

    def to_dict(self) -> Dict[str, Any]:
        """Convert to dictionary for JSON serialization."""
        return {
            "schema_version": self.schema_version,
            "deployment_mode": self.deployment_mode,
            "provider_name": self.provider_name,
            "privacy_classification": self.privacy_classification,
            "overall_status": self.overall_status.value,
            "checks": [check.to_dict() for check in self.checks]
        }


class DoctorChecks:
    """Implements all diagnostic checks."""

    def __init__(self, repo_root: Path, deployment_mode: str, provider_name: Optional[str] = None, cpu_only: bool = False):
        self.repo_root = repo_root
        self.deployment_mode = deployment_mode
        self.provider_name = provider_name
        self.cpu_only = cpu_only
        self.checks: List[Check] = []

    def check_host_platform(self) -> Check:
        """Check if the host platform is supported."""
        system = platform.system()
        if system == "Linux":
            return Check(
                code="HOST_SUPPORTED",
                severity=Severity.PASS,
                summary=f"Host platform: {system}",
                details=f"Platform: {system}, Machine: {platform.machine()}"
            )
        else:
            return Check(
                code="HOST_SUPPORTED",
                severity=Severity.WARN,
                summary=f"Host platform: {system} (untested)",
                remediation="Linux is the primary tested platform. Other platforms may work but are not officially supported.",
                details=f"Platform: {system}, Machine: {platform.machine()}"
            )

    def check_repository_layout(self) -> Check:
        """Check that required repository files exist."""
        required_files = ["run.sh", "scripts/bash_helpers.sh", "src"]
        missing = []

        for file_path in required_files:
            if not (self.repo_root / file_path).exists():
                missing.append(file_path)

        if not missing:
            return Check(
                code="REPO_LAYOUT",
                severity=Severity.PASS,
                summary="Required repository files found"
            )
        else:
            return Check(
                code="REPO_LAYOUT",
                severity=Severity.FAIL,
                summary=f"Missing required files: {', '.join(missing)}",
                remediation="Ensure you are running from the repository root directory."
            )

    def check_docker_cli(self) -> Optional[Check]:
        """Check if Docker CLI is available (Docker mode only)."""
        if self.deployment_mode != "docker":
            return None

        docker_path = shutil.which("docker")
        if docker_path:
            return Check(
                code="DOCKER_CLI",
                severity=Severity.PASS,
                summary="Docker CLI found",
                details=f"Path: {docker_path}"
            )
        else:
            return Check(
                code="DOCKER_CLI",
                severity=Severity.FAIL,
                summary="Docker CLI not found",
                remediation="Install Docker from https://docs.docker.com/get-docker/"
            )

    def check_docker_daemon(self) -> Optional[Check]:
        """Check if Docker daemon is reachable (Docker mode only)."""
        if self.deployment_mode != "docker":
            return None

        # First check if CLI exists
        if not shutil.which("docker"):
            return None  # Already reported by check_docker_cli

        try:
            # Use a non-mutating call to check daemon
            result = subprocess.run(
                ["docker", "info"],
                capture_output=True,
                timeout=5,
                text=True
            )

            if result.returncode == 0:
                return Check(
                    code="DOCKER_DAEMON",
                    severity=Severity.PASS,
                    summary="Docker daemon reachable"
                )
            else:
                # Check for common permission errors
                if "permission denied" in result.stderr.lower():
                    return Check(
                        code="DOCKER_DAEMON",
                        severity=Severity.FAIL,
                        summary="Docker daemon: permission denied",
                        remediation="Add your user to the docker group: sudo usermod -aG docker $USER, then log out and log back in."
                    )
                else:
                    return Check(
                        code="DOCKER_DAEMON",
                        severity=Severity.FAIL,
                        summary="Docker daemon not reachable",
                        remediation="Ensure Docker daemon is running. Try: sudo systemctl start docker"
                    )
        except subprocess.TimeoutExpired:
            return Check(
                code="DOCKER_DAEMON",
                severity=Severity.FAIL,
                summary="Docker daemon not responding",
                remediation="Docker daemon appears unresponsive. Check Docker service status."
            )
        except Exception as e:
            return Check(
                code="DOCKER_DAEMON",
                severity=Severity.FAIL,
                summary="Failed to check Docker daemon",
                details=f"Error: {type(e).__name__}"
            )

    def check_conda_cli(self) -> Optional[Check]:
        """Check if Conda is available (local mode only)."""
        if self.deployment_mode != "local":
            return None

        conda_path = shutil.which("conda")
        if conda_path:
            return Check(
                code="CONDA_CLI",
                severity=Severity.PASS,
                summary="Conda found",
                details=f"Path: {conda_path}"
            )
        else:
            return Check(
                code="CONDA_CLI",
                severity=Severity.FAIL,
                summary="Conda not found",
                remediation="Install Miniconda from https://docs.conda.io/en/latest/miniconda.html"
            )

    def check_workspace_writable(self) -> Check:
        """Check if workspace directory is writable."""
        workspace_parent = self.repo_root

        try:
            # Test write access
            test_file = workspace_parent / ".doctor_write_test"
            test_file.touch()
            test_file.unlink()

            return Check(
                code="WORKSPACE_WRITABLE",
                severity=Severity.PASS,
                summary="Workspace is writable"
            )
        except (PermissionError, OSError) as e:
            return Check(
                code="WORKSPACE_WRITABLE",
                severity=Severity.FAIL,
                summary="Workspace is not writable",
                remediation=f"Ensure you have write permissions for {workspace_parent}",
                details=f"Error: {type(e).__name__}"
            )

    def check_outputs_writable(self) -> Check:
        """Check if outputs directory is writable."""
        outputs_dir = self.repo_root / "outputs"

        # Create outputs dir if it doesn't exist
        outputs_dir.mkdir(exist_ok=True)

        try:
            # Test write access
            test_file = outputs_dir / ".doctor_write_test"
            test_file.touch()
            test_file.unlink()

            return Check(
                code="OUTPUTS_WRITABLE",
                severity=Severity.PASS,
                summary="Outputs directory is writable"
            )
        except (PermissionError, OSError) as e:
            return Check(
                code="OUTPUTS_WRITABLE",
                severity=Severity.FAIL,
                summary="Outputs directory is not writable",
                remediation=f"Ensure you have write permissions for {outputs_dir}",
                details=f"Error: {type(e).__name__}"
            )

    def check_disk_space(self) -> Check:
        """Check available disk space."""
        try:
            stat = shutil.disk_usage(self.repo_root)
            free_gb = stat.free / (1024 ** 3)

            # Warning threshold: 10 GB
            # This is a documented threshold, not a hard requirement
            if free_gb >= 10:
                return Check(
                    code="DISK_SPACE",
                    severity=Severity.PASS,
                    summary=f"Available disk space: {free_gb:.1f} GB"
                )
            elif free_gb >= 5:
                return Check(
                    code="DISK_SPACE",
                    severity=Severity.WARN,
                    summary=f"Low disk space: {free_gb:.1f} GB available",
                    remediation="Free up disk space. Docker images and training artifacts can be large."
                )
            else:
                return Check(
                    code="DISK_SPACE",
                    severity=Severity.FAIL,
                    summary=f"Very low disk space: {free_gb:.1f} GB available",
                    remediation="Free up disk space before proceeding. At least 5 GB recommended."
                )
        except Exception as e:
            return Check(
                code="DISK_SPACE",
                severity=Severity.INFO,
                summary="Could not check disk space",
                details=f"Error: {type(e).__name__}"
            )

    def check_gpu_available(self) -> Check:
        """Check GPU availability (informational)."""
        if self.cpu_only:
            return Check(
                code="GPU_AVAILABLE",
                severity=Severity.INFO,
                summary="GPU disabled (--cpu-only flag set)"
            )

        # Try nvidia-smi for NVIDIA GPUs
        try:
            result = subprocess.run(
                ["nvidia-smi", "-L"],
                capture_output=True,
                timeout=2,
                text=True
            )

            if result.returncode == 0:
                gpu_count = len([line for line in result.stdout.split('\n') if line.strip()])
                return Check(
                    code="GPU_AVAILABLE",
                    severity=Severity.INFO,
                    summary=f"NVIDIA GPU available ({gpu_count} detected)"
                )
        except (FileNotFoundError, subprocess.TimeoutExpired):
            pass

        return Check(
            code="GPU_AVAILABLE",
            severity=Severity.INFO,
            summary="No GPU detected (will use CPU)"
        )

    def check_datasets_present(self) -> Check:
        """Check if any public datasets are present."""
        datasets_dir = self.repo_root / "datasets"

        if not datasets_dir.exists():
            return Check(
                code="DATASETS_PRESENT",
                severity=Severity.WARN,
                summary="No datasets directory found",
                remediation="Download an example dataset: ./scripts/download_example_dataset.sh --dataset breast_cancer"
            )

        # Look for directories that contain train/
        dataset_dirs = [d for d in datasets_dir.iterdir()
                       if d.is_dir() and (d / "train").exists()]

        if dataset_dirs:
            return Check(
                code="DATASETS_PRESENT",
                severity=Severity.PASS,
                summary=f"Found {len(dataset_dirs)} dataset(s)"
            )
        else:
            return Check(
                code="DATASETS_PRESENT",
                severity=Severity.WARN,
                summary="No datasets found",
                remediation="Download an example dataset: ./scripts/download_example_dataset.sh --dataset breast_cancer\nOr list available: ./scripts/download_example_dataset.sh --list"
            )

    def check_provider_configured(self) -> tuple[Check, str]:
        """
        Check if provider credentials appear configured.

        Returns: (Check, privacy_classification)
        """
        # This is a presence check only - never validates by making API calls
        # Never prints credential values

        env_vars = {
            "OPENROUTER_API_KEY": ("openrouter", "external"),
            "OPENAI_API_KEY": ("openai", "external"),
            "ANTHROPIC_API_KEY": ("anthropic", "external"),
            "OLLAMA_BASE_URL": ("ollama", "local"),
        }

        configured_providers = []
        privacy = "unknown"

        # Check .env file
        env_file = self.repo_root / ".env"
        env_values = {}

        if env_file.exists():
            try:
                with open(env_file) as f:
                    for line in f:
                        line = line.strip()
                        if line and not line.startswith('#') and '=' in line:
                            key, value = line.split('=', 1)
                            key = key.strip()
                            value = value.strip().strip('"').strip("'")
                            if value:  # Non-empty value
                                env_values[key] = value
            except Exception:
                pass

        # Also check environment variables
        for env_var, (provider_name, provider_privacy) in env_vars.items():
            if env_values.get(env_var) or os.environ.get(env_var):
                configured_providers.append(provider_name)
                if privacy == "unknown":
                    privacy = provider_privacy

        # Check for Codex auth (look for auth file without reading it)
        codex_auth_file = Path.home() / ".cache" / "codex" / "auth.json"
        if codex_auth_file.exists():
            try:
                # Just check it's readable and non-zero size
                if codex_auth_file.stat().st_size > 0:
                    configured_providers.append("codex")
                    if privacy == "unknown":
                        privacy = "external"  # Codex is external
            except Exception:
                pass

        if configured_providers:
            provider_list = ", ".join(configured_providers)
            return Check(
                code="PROVIDER_CONFIGURED",
                severity=Severity.PASS,
                summary=f"Provider credentials found: {provider_list}"
            ), privacy
        else:
            return Check(
                code="PROVIDER_CONFIGURED",
                severity=Severity.FAIL,
                summary="No provider credentials found",
                remediation="Configure a provider:\n  - External: Add API key to .env (OPENROUTER_API_KEY, OPENAI_API_KEY, or ANTHROPIC_API_KEY)\n  - Local: Install Ollama and set OLLAMA_BASE_URL in .env"
            ), "unknown"

    def run_all_checks(self) -> DoctorReport:
        """Run all applicable checks and return a report."""
        checks = []

        # Common checks
        checks.append(self.check_host_platform())
        checks.append(self.check_repository_layout())
        checks.append(self.check_workspace_writable())
        checks.append(self.check_outputs_writable())
        checks.append(self.check_disk_space())
        checks.append(self.check_datasets_present())

        # Deployment-specific checks
        if self.deployment_mode == "docker":
            docker_cli = self.check_docker_cli()
            if docker_cli:
                checks.append(docker_cli)

            docker_daemon = self.check_docker_daemon()
            if docker_daemon:
                checks.append(docker_daemon)

            checks.append(self.check_gpu_available())
        else:  # local mode
            conda_cli = self.check_conda_cli()
            if conda_cli:
                checks.append(conda_cli)

            # Add local mode security warning
            checks.append(Check(
                code="LOCAL_MODE_SECURITY",
                severity=Severity.WARN,
                summary="Local mode executes agent-generated code without containerization",
                remediation="Use Docker mode for better isolation, or ensure you trust the data and configured provider."
            ))

        # Provider check
        provider_check, privacy = self.check_provider_configured()
        checks.append(provider_check)

        # Add privacy warning if external
        if privacy == "external":
            checks.append(Check(
                code="EXTERNAL_PROVIDER_WARNING",
                severity=Severity.WARN,
                summary="External provider selected; data-derived context may leave this machine",
                remediation="For sensitive data, use local Ollama models or verify your organization's data governance policies."
            ))
        elif privacy == "unknown":
            checks.append(Check(
                code="UNKNOWN_PROVIDER_PRIVACY",
                severity=Severity.WARN,
                summary="Provider privacy classification unknown",
                remediation="Verify the endpoint and policy of your configured provider before using with sensitive data."
            ))

        # Determine overall status
        has_fail = any(c.severity == Severity.FAIL for c in checks)
        has_warn = any(c.severity == Severity.WARN for c in checks)

        if has_fail:
            overall = OverallStatus.NOT_READY
        elif has_warn:
            overall = OverallStatus.READY_WITH_WARNINGS
        else:
            overall = OverallStatus.READY

        return DoctorReport(
            schema_version=1,
            deployment_mode=self.deployment_mode,
            provider_name=self.provider_name,
            privacy_classification=privacy,
            overall_status=overall,
            checks=checks
        )


def render_human_output(report: DoctorReport) -> str:
    """Render the report in human-readable format."""
    lines = []
    lines.append("Agentomics setup check")
    lines.append("")

    for check in report.checks:
        # Severity label with padding
        severity_label = f"{check.severity.value:4}"
        lines.append(f"{severity_label}  {check.summary}")

        if check.remediation:
            # Indent remediation
            for remediation_line in check.remediation.split('\n'):
                lines.append(f"      {remediation_line}")

    lines.append("")

    # Count warnings and errors
    fail_count = sum(1 for c in report.checks if c.severity == Severity.FAIL)
    warn_count = sum(1 for c in report.checks if c.severity == Severity.WARN)

    result_text = f"Result: {report.overall_status.value}"
    if warn_count > 0 and fail_count == 0:
        result_text += f" ({warn_count} warning{'s' if warn_count != 1 else ''})"
    elif fail_count > 0:
        result_text += f" ({fail_count} error{'s' if fail_count != 1 else ''}"
        if warn_count > 0:
            result_text += f", {warn_count} warning{'s' if warn_count != 1 else ''}"
        result_text += ")"

    lines.append(result_text)

    return "\n".join(lines)


def render_json_output(report: DoctorReport) -> str:
    """Render the report as JSON."""
    return json.dumps(report.to_dict(), indent=2)


def run_doctor(
    repo_root: Path,
    deployment_mode: str = "docker",
    provider_name: Optional[str] = None,
    cpu_only: bool = False,
    json_output: bool = False
) -> tuple[str, int]:
    """
    Run all doctor checks and return output string and exit code.

    Returns:
        (output_string, exit_code)
        Exit codes: 0 = ready (including warnings), 1 = not ready, 2 = invalid usage
    """
    doctor = DoctorChecks(
        repo_root=repo_root,
        deployment_mode=deployment_mode,
        provider_name=provider_name,
        cpu_only=cpu_only
    )

    report = doctor.run_all_checks()

    if json_output:
        output = render_json_output(report)
    else:
        output = render_human_output(report)

    # Exit code: 0 for ready/ready with warnings, 1 for not ready
    exit_code = 0 if report.overall_status != OverallStatus.NOT_READY else 1

    return output, exit_code
