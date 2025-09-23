import subprocess
import sys
from hrid import HRID

def _change_repository_permissions(agent_id):
    """Change to read-only agent access for repository with specific restrictions."""
    subprocess.run(["setfacl", "-R", "-m", f"u:{agent_id}:r-X", "/repository"], check=True)

    subprocess.run(["setfacl", "-m", f"u:{agent_id}:---", "/repository/.env"], check=True)

    subprocess.run(["setfacl", "-R", "-m", f"u:{agent_id}:---", "/repository/datasets"], check=True)
    subprocess.run(["setfacl", "-R", "-m", f"u:{agent_id}:---", "/repository/prepared_datasets"], check=True)

def create_new_user_and_rundir(config):
    run_id = "_".join(HRID().generate().replace("-", "_").replace(" ","_").split("_")[:-1])[:32]
    run_dir = config.runs_dir / run_id
    snapshot_dir = config.snapshots_dir / run_id

    if config.root_privileges:
        subprocess.run(
            ["useradd", "-d", run_dir, "-m", "-U", run_id],
            check=True
        )
        _change_repository_permissions(run_id)
    else:
        run_dir.mkdir(parents=True, exist_ok=True)
    
    snapshot_dir.mkdir(parents=True, exist_ok=True)
    
    return run_id

if __name__ == "__main__":
    try:
        run_id = create_new_user_and_rundir()
        print(run_id)
        sys.exit(0)
    except Exception as e:
        print(f"Error: {str(e)}", file=sys.stderr)
        sys.exit(1)