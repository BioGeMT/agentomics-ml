import sys
from hrid import HRID


def create_agent_id():
    return "_".join(HRID().generate().replace("-", "_").replace(" ","_").split("_")[:-1])[:32]

if __name__ == "__main__":
    try:
        run_id = create_agent_id()
        print(run_id)
        sys.exit(0)
    except Exception as e:
        print(f"Error: {str(e)}", file=sys.stderr)
        sys.exit(1)
