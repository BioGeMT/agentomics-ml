import os
from pathlib import Path
import re
import shutil

def get_metrics_from_file(file_path):
    metrics = {}
    with open(file_path, "r") as f:
        for line in f:
            key, value = line.strip().split(": ")
            metrics[key] = float(value)
    return metrics

def get_valid_and_train_metrics(base_path):
    val_metrics = get_metrics_from_file(f"{base_path}/validation_metrics.txt")
    train_metric = get_metrics_from_file(f"{base_path}/train_metrics.txt")
    all_metrics = {}
    val_metrics = {f"validation/{k}": v for k, v in val_metrics.items()}
    train_metric = {f"train/{k}": v for k, v in train_metric.items()}
    all_metrics.update(val_metrics)
    all_metrics.update(train_metric)
    return all_metrics

def get_best_metrics(agent_id):
    if(not best_metrics_exists(agent_id)):
        return {}
    else:
        return get_valid_and_train_metrics(f"/snapshots/{agent_id}")

def get_new_metrics(agent_id):
    if(not new_metrics_exists(agent_id)):
        return {}
    else:
        return get_valid_and_train_metrics(f"/workspace/runs/{agent_id}")

def get_new_and_best_metrics(agent_id):
    return get_new_metrics(agent_id), get_best_metrics(agent_id)

def best_metrics_exists(agent_id):
    best_metrics_path = f"/snapshots/{agent_id}/validation_metrics.txt"
    if(os.path.isfile(best_metrics_path)):
        return True
    return False

def new_metrics_exists(agent_id):
    new_metrics_path = f"/workspace/runs/{agent_id}/validation_metrics.txt"
    if(os.path.isfile(new_metrics_path)):
        return True
    return False

def is_new_best(agent_id, comparison_metric):
    if(not best_metrics_exists(agent_id)):
        return True
    
    new_metrics, best_metrics = get_new_and_best_metrics(agent_id)

    #TODO parametrize improvement threshold
    necessary_improvement = 0
    is_new_best = new_metrics[f'validation/{comparison_metric}'] > best_metrics[f'validation/{comparison_metric}'] + necessary_improvement
    return is_new_best
       
def delete_snapshot(agent_id):
    snapshot_dir = Path(f"/snapshots/{agent_id}")
    if snapshot_dir.exists():
        shutil.rmtree(snapshot_dir)

def snapshot(agent_id, iteration, delete_old_snapshot=True):
    if delete_old_snapshot:
        delete_snapshot(agent_id)

    run_dir = f"/workspace/runs/{agent_id}"
    snapshot_dir = f"/snapshots/{agent_id}"
    Path(snapshot_dir).mkdir(parents=True, exist_ok=True)
    
    files_to_skip = [
        "train.csv",
        "validation.csv",
    ]
    folders_to_skip = [
        "__pycache__",
        ".cache",
    ]
    # iterate the snapshot dir for all files
    for element in os.listdir(run_dir):
        element = Path(run_dir)/ element
        # if hidden file and not in a folder, skip it
        if re.match(r"^\..*", element.name) and element.is_file():
            continue
        if element.is_file() and element.name in files_to_skip:
            continue
        if element.is_dir() and element.name in folders_to_skip:
            continue
        if element.is_file():
            # hard copy the file into snapshot dir
            shutil.copy2(element, Path(snapshot_dir) / element.name)
        if element.is_dir():
            # hard copy the folder into snapshot dir
            shutil.copytree(element, Path(snapshot_dir) / element.name, dirs_exist_ok=True)
    
    with open(Path(snapshot_dir) / "iteration_number.txt", "w") as f:
        f.write(str(iteration))
