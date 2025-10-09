import sys
from hrid import HRID

def create_new_user_and_rundir(config):
    run_id = "_".join(HRID().generate().replace("-", "_").replace(" ","_").split("_")[:-1])[:32]
    run_dir = config.runs_dir / run_id
    snapshot_dir = config.snapshots_dir / run_id

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