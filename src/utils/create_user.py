import os
import subprocess
import sys
from hrid import HRID

def create_new_user_and_rundir():
    run_id = "_".join(HRID().generate().replace("-", "_").replace(" ","_").split("_")[:-1])[:32]
    run_dir = os.path.join("/workspace/runs", run_id)
    subprocess.run(
        ["sudo", "useradd", "-d", run_dir, "-m", "-p", "1234", run_id],
        check=True
    )
    subprocess.run(
        ["sudo", "chmod", "o-rwx", run_dir],
        check=True
    )
    subprocess.run(
        ["sudo", "mkdir", f"/snapshots/{run_id}"],
        check=True
    )
    return run_id

if __name__ == "__main__":
    try:
        run_id = create_new_user_and_rundir()
        print(run_id)
        sys.exit(0)
    except Exception as e:
        print(f"Error: {str(e)}", file=sys.stderr)
        sys.exit(1)