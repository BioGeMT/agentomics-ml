import os
import subprocess
from hrid import HRID

def create_new_user_and_rundir():
    run_id = "_".join(HRID().generate().replace("-", "_").split("_")[:-1])[:32]    
    run_dir = os.path.join("/workspace/runs", run_id)
    subprocess.run(["mkdir", run_dir])
    subprocess.run(["chmod", "o-rwx", run_dir], check=True)
    return run_id
