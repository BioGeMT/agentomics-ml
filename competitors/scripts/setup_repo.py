import argparse
import subprocess
import sys
from pathlib import Path

import yaml


def main() -> None:
    parser = argparse.ArgumentParser(description="Clone and install biomlbench fork")
    parser.add_argument("--config", required=True, help="Path to config.yaml")
    args = parser.parse_args()

    config_path = Path(args.config)
    cfg = yaml.safe_load(config_path.read_text())

    repo_url = cfg["biomlbench_repo"]
    branch = cfg.get("biomlbench_branch", "main")
    competitors_dir = config_path.parent
    clone_dir = competitors_dir / "biomlbench"

    if clone_dir.exists():
        print(f"[setup_repo] Directory {clone_dir} exists, pulling latest changes")
        subprocess.run(["git", "-C", str(clone_dir), "fetch", "origin"], check=True)
        subprocess.run(["git", "-C", str(clone_dir), "checkout", branch], check=True)
        subprocess.run(["git", "-C", str(clone_dir), "pull", "origin", branch], check=True)
    else:
        print(f"[setup_repo] Cloning {repo_url} (branch {branch})")
        subprocess.run(["git", "clone", "--branch", branch, repo_url, str(clone_dir)], check=True)

    print(f"[setup_repo] Installing {clone_dir} in editable mode")
    subprocess.run([sys.executable, "-m", "pip", "install", "-e", str(clone_dir)], check=True)
    print("[setup_repo] Done")

if __name__ == "__main__":
    main()
