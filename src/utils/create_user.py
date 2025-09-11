import subprocess
import sys
from hrid import HRID

def create_new_user_and_rundir(config):
    run_id = "_".join(HRID().generate().replace("-", "_").replace(" ","_").split("_")[:-1])[:32]
    run_dir = config.runs_dir / run_id
    snapshot_dir = config.snapshots_dir / run_id

    if config.root_privileges:
        subprocess.run(
            ["sudo", "useradd", "-d", run_dir, "-m", "-p", "1234", run_id],
            check=True
        )
        subprocess.run(
            ["sudo", "chmod", "o-rwx", run_dir],
            check=True
        )
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