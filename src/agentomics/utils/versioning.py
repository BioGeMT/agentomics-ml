from __future__ import annotations

from agentomics import __version__


def get_version() -> str:
    return __version__

def _major(version: str) -> int:
    return int(version.split(".")[0])

def check_run_compatible(recorded_version: str) -> None:
    if _major(recorded_version) == _major(get_version()):
        return
    raise RuntimeError(
        f"This run was produced by Agentomics {recorded_version}, which is "
        f"incompatible with the running version {get_version()}. Install a "
        f"matching version to work with it, for example: "
        f"pip install 'agentomics=={_major(recorded_version)}.*'"
    )
