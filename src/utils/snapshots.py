import re
import shutil
from pathlib import Path
from utils.metrics import get_classification_metrics_functions, get_higher_is_better_map, get_regression_metrics_functions

def get_metrics_from_file(file_path):
    metrics = {}
    with open(file_path, "r") as f:
        for line in f:
            key, value = line.strip().split(": ")
            metrics[key] = float(value)
    return metrics

def get_valid_and_train_metrics(base_path):
    base_path = Path(base_path)
    val_metrics = get_metrics_from_file(base_path / "validation_metrics.txt")
    train_metric = get_metrics_from_file(base_path / "train_metrics.txt")
    all_metrics = {}
    val_metrics = {f"validation/{k}": v for k, v in val_metrics.items()}
    train_metric = {f"train/{k}": v for k, v in train_metric.items()}
    all_metrics.update(val_metrics)
    all_metrics.update(train_metric)
    return all_metrics

def get_best_metrics(config):
    if(not best_metrics_exists(config)):
        return {}
    else:
        return get_valid_and_train_metrics(config.snapshots_dir / config.agent_id)

def get_new_metrics(config):
    if(not new_metrics_exists(config)):
        return {}
    else:
        return get_valid_and_train_metrics(config.runs_dir / config.agent_id)

def get_new_and_best_metrics(config):
    return get_new_metrics(config), get_best_metrics(config)

def best_metrics_exists(config):
    best_metrics_path = config.snapshots_dir / config.agent_id / "validation_metrics.txt"
    if(best_metrics_path.is_file()):
        return True
    return False

def new_metrics_exists(config):
    new_metrics_path = config.runs_dir / config.agent_id / "validation_metrics.txt"
    if(new_metrics_path.is_file()):
        return True
    return False

def is_new_best(config):
    if(not best_metrics_exists(config)):
        return True
    
    new_metrics, best_metrics = get_new_and_best_metrics(config)

    #TODO parametrize improvement threshold
    necessary_improvement = 0
    metric_name = config.val_metric
    
    higher_is_better_map = get_higher_is_better_map()
    higher_is_better = higher_is_better_map[metric_name]

    new_val = new_metrics[f"validation/{metric_name}"]
    best_val = best_metrics[f"validation/{metric_name}"]

    if higher_is_better:
        is_new_best = new_val > best_val + necessary_improvement
    else:
        is_new_best = new_val < best_val - necessary_improvement
    print(f"is_new_best: {is_new_best}")
    print(f"New metrics: {new_metrics}")
    print(f"Best metrics: {best_metrics}")
    return is_new_best
       
def delete_snapshot(snapshot_dir):
    if snapshot_dir.exists():
        shutil.rmtree(snapshot_dir)

def snapshot(config, iteration, delete_old_snapshot=True):
    run_dir = config.runs_dir / config.agent_id
    snapshot_dir = config.snapshots_dir / config.agent_id

    if delete_old_snapshot:
        delete_snapshot(snapshot_dir)

    snapshot_dir.mkdir(parents=True, exist_ok=True)

    files_to_skip = [
        "train.csv",
        "validation.csv",
    ]
    folders_to_skip = [
        "__pycache__",
        ".cache",
    ]
    # iterate the snapshot dir for all files
    for element in run_dir.iterdir():
        # if hidden file and not in a folder, skip it
        if re.match(r"^\..*", element.name) and element.is_file():
            continue
        if element.is_file() and element.name in files_to_skip:
            continue
        if element.is_dir() and element.name in folders_to_skip:
            continue
        if element.is_file():
            # hard copy the file into snapshot dir
            # print(f"Snapshotting {element.name}")
            absolute_dest = snapshot_dir / element.name
            shutil.copy2(element, absolute_dest)
            if element.name.endswith(".py"):
                replace_workspace_path_with_snapshots(run_dir, snapshot_dir, absolute_path_snapshot_file=absolute_dest)
        if element.is_dir():
            # hard copy the folder into snapshot dir
            # print(f"Snapshotting {element.name}")
            shutil.copytree(element, snapshot_dir / element.name, dirs_exist_ok=True, symlinks=True)
            # if dir is not hidden, replace the workspace path in python files
            if(not re.match(r"^\..*", element.name)):
                for file_path in (snapshot_dir / element.name).rglob('*.py'):
                    replace_workspace_path_with_snapshots(run_dir, snapshot_dir, absolute_path_snapshot_file=file_path)
        
    with open(snapshot_dir / "iteration_number.txt", "w") as f:
        f.write(str(iteration))

def get_best_iteration(config):
    snapshot_dir = config.snapshots_dir / config.agent_id
    iteration_file = snapshot_dir / "iteration_number.txt"
    if iteration_file.exists():
        with open(iteration_file, 'r') as f:
            return int(f.read().strip())
    return 0

def replace_workspace_path_with_snapshots(run_dir, snapshot_dir, absolute_path_snapshot_file):
    # Replaces hard-coded paths in the files to point to the snapshot dir
    with open(absolute_path_snapshot_file, "r") as f:
        old_content = f.read()
    new_content = old_content.replace(str(run_dir), str(snapshot_dir))
    if(old_content != new_content):
        with open(absolute_path_snapshot_file, "w") as f:
            f.write(new_content)

def replace_snapshot_path_with_relative(snapshot_dir):
    for file_path in snapshot_dir.rglob('*.py'):
        if '.conda' in file_path.parts:
            # Don't check installed packages scripts
            continue
        with open(file_path, "r") as f:
            old_content = f.read()
        new_content = old_content.replace(str(snapshot_dir), ".")
        with open(file_path, "w") as f:
            f.write(new_content)