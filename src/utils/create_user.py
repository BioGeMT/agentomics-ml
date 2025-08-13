import subprocess
import sys
from hrid import HRID

def create_new_user_and_rundir(config):
    print("🆔 Generating unique run ID...")
    run_id = "_".join(HRID().generate().replace("-", "_").replace(" ","_").split("_")[:-1])[:32]
    print(f"✅ Generated run ID: {run_id}")
    
    run_dir = config.workspace_dir / run_id
    snapshot_dir = config.snapshot_dir / run_id

    if config.root_privileges:
        print("👤 Creating user with root privileges...")
        subprocess.run(
        ["sudo", "useradd", "-d", run_dir, "-m", "-p", "1234", run_id],
        check=True
        )
        
        print("🔒 Setting directory permissions...")
        subprocess.run(
            ["sudo", "chmod", "o-rwx", run_dir],
            check=True
        )
    else:
        print("📁 Creating run directory (no root privileges)...")
        run_dir.mkdir(parents=True, exist_ok=True)
    
    print("📸 Creating snapshot directory...")
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