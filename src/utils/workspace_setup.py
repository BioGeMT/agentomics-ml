from pathlib import Path

def ensure_workspace_folders(config):
    #parents=False to avoid misconfigured paths
    Path(config.snapshots_dir).mkdir(parents=False, exist_ok=True)
    Path(config.reports_dir).mkdir(parents=False, exist_ok=True)
    Path(config.runs_dir).mkdir(parents=False, exist_ok=True)
    Path(config.fallbacks_dir).mkdir(parents=False, exist_ok=True)