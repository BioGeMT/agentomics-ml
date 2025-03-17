import os
import subprocess
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
    return run_id