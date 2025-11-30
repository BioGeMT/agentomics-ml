import re
import shutil
import hashlib
import os
import stat
import subprocess
import wandb
from pathlib import Path
from utils.metrics import get_classification_metrics_functions, get_higher_is_better_map, get_regression_metrics_functions
from run_logging.logging_helpers import is_wandb_active

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
    if not new_metrics:
        return False

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
       
def delete_dir(dir_path):
    if dir_path.exists():
        shutil.rmtree(dir_path)

def file_fingerprint(path, chunk_size=65536):
    hasher = hashlib.sha256()
    try:
        with open(path, 'rb') as f:
            while chunk := f.read(chunk_size):
                hasher.update(chunk)
    except FileNotFoundError:
        return str(path) # mis-matches any different paths
    
    return hasher.hexdigest()

def create_split_fingerprint(config):
    train_csv = config.runs_dir / config.agent_id / 'train.csv'
    valid_csv = config.runs_dir / config.agent_id / 'validation.csv'
    return file_fingerprint(train_csv) + file_fingerprint(valid_csv)

def reset_snapshot_if_val_split_changed(config, iteration, old_fingerprint, new_fingerprint):
    if(old_fingerprint == None): #old fingerprint is none - split didnt exist,
        if is_wandb_active():
            wandb.log({"validation/snapshot_reset":0}, step=iteration)
        return False
    
    if new_fingerprint == None or old_fingerprint != new_fingerprint: #new_fingerprint is none - agent deleted the split
        if is_wandb_active():
            wandb.log({"validation/snapshot_reset":1}, step=iteration)
        snapshot_dir = config.snapshots_dir / config.agent_id
        delete_dir(snapshot_dir)
        snapshot_dir.mkdir(parents=True, exist_ok=True)
        return True
    else:
        if is_wandb_active():
            wandb.log({"validation/snapshot_reset":0}, step=iteration)
        return False

def lock_split_files(config):
    train_split = config.runs_dir / config.agent_id / 'train.csv'
    validation_split = config.runs_dir / config.agent_id / 'validation.csv'
    
    read_only_mode = stat.S_IRUSR | stat.S_IRGRP | stat.S_IROTH
    os.chmod(train_split, read_only_mode)
    os.chmod(validation_split, read_only_mode)

def snapshot(config, iteration, structured_outputs, delete_old_snapshot=True):
    snapshot_dir = config.snapshots_dir / config.agent_id
    run_dir = config.runs_dir / config.agent_id
    snapshot_run_dir_into_dest(
        config=config,
        destination_dir=snapshot_dir,
        delete_old_destination_dir=delete_old_snapshot,
        structured_outputs=structured_outputs,
        snapshot_conda=True,
        is_best=True,
        path_to_replace=run_dir,
        path_replacement = ".",
    )

    with open(snapshot_dir / "iteration_number.txt", "w") as f:
        f.write(str(iteration))

def get_best_iteration(config):
    snapshot_dir = config.snapshots_dir / config.agent_id
    iteration_file = snapshot_dir / "iteration_number.txt"
    if iteration_file.exists():
        with open(iteration_file, 'r') as f:
            return int(f.read().strip())
    return None

def export_conda_to_dir(config, run_dir, destination_dir, include_packages=False):
    if include_packages:
        conda_env = run_dir / ".conda"
        shutil.copytree(str(conda_env), str(destination_dir / ".conda"), symlinks=False, dirs_exist_ok=True)
    
    env_name = f"{config.agent_id}_env"
    conda_env = run_dir / ".conda" / "envs" / env_name
    if conda_env.exists():
        subprocess.run(['conda', 'env', 'export', '-p', str(conda_env), '-f', str(destination_dir / "conda_environment.yml")], check=True, capture_output=True)

def save_outputs_to_dir(dest_dir, structured_outputs):
    with open(dest_dir / "structured_outputs.txt", "w") as f:
        f.write(str(structured_outputs))

def purge_conda_from_all_iteration_folders(config):
    for element in (config.runs_dir / config.agent_id).iterdir():
        if element.is_dir() and element.name.startswith("iteration_"):
            conda_env = element / ".conda"
            if conda_env.exists():
                shutil.rmtree(conda_env)

def delete_metrics_from_iteration_dir(config, iteration):
    run_dir = config.runs_dir / config.agent_id
    iteration_dir = run_dir / f"iteration_{iteration}"
    train_metrics_path = iteration_dir / "train_metrics.txt"
    validation_metrics_path = iteration_dir / "validation_metrics.txt"

    if train_metrics_path.exists():
        train_metrics_path.unlink()
    if validation_metrics_path.exists():
        validation_metrics_path.unlink()

def populate_iteration_dir(config, run_index, is_best, structured_outputs):
    run_dir = config.runs_dir / config.agent_id
    iteration_dir = run_dir / f"iteration_{run_index}"
    snapshot_dir = config.snapshots_dir / config.agent_id
    if(is_best):
        purge_conda_from_all_iteration_folders(config)

    snapshot_run_dir_into_dest(
        config=config,
        destination_dir=iteration_dir,
        delete_old_destination_dir=True,
        structured_outputs=structured_outputs,
        snapshot_conda=True,
        is_best=is_best,
        path_to_replace=snapshot_dir, #Rundir is replaced with Snapshot paths before populate_iteration_dir is called
        path_replacement=iteration_dir,
        remove_source_files=True,
    )

def snapshot_run_dir_into_dest(config, destination_dir, delete_old_destination_dir, structured_outputs, snapshot_conda, path_to_replace, path_replacement, remove_source_files=False, is_best=False):
    run_dir = config.runs_dir / config.agent_id
    destination_dir = Path(destination_dir)

    if delete_old_destination_dir and destination_dir.exists():
        delete_dir(destination_dir)
        
    destination_dir.mkdir()

    files_to_skip = [
        "train.csv",
        "validation.csv",
    ]

    files_to_delete = [
        'dry_run_metrics.txt',
    ]
    folders_to_delete = [
        "__pycache__",
        ".cache",
    ]
    replace_python_paths(folder_path=run_dir, current_path=path_to_replace, new_path=path_replacement)

    for element in run_dir.iterdir():
        if element.is_dir() and (element.name.startswith("iteration_") or element.name == ".conda"):
            continue
        if element.name in files_to_skip:
            continue
        if element.name in files_to_delete and element.exists():
            element.unlink()
            continue
        if element.name in folders_to_delete and element.is_dir():
            shutil.rmtree(element)
            continue
        if element.is_dir():
            shutil.copytree(str(element), str(destination_dir / element.name), symlinks=False, dirs_exist_ok=True)
            if(remove_source_files):
                shutil.rmtree(element)
        else:
            shutil.copy(str(element), str(destination_dir / element.name), follow_symlinks=True)
            if(remove_source_files):
                element.unlink()


    if snapshot_conda:
        export_conda_to_dir(config, run_dir, destination_dir, include_packages=is_best)
    if structured_outputs is not None:
        save_outputs_to_dir(destination_dir, structured_outputs)

def wipe_current_iter_files(config):
    run_dir = config.runs_dir / config.agent_id

    files_to_skip = [
        "train.csv",
        "validation.csv",
    ]
    for element in run_dir.iterdir():
        if element.is_dir() and (element.name.startswith("iteration_") or element.name == ".conda"):
            continue
        if element.name in files_to_skip:
            continue

        if element.is_dir():
            shutil.rmtree(element)
        else:
            element.unlink()

def replace_python_paths(folder_path, current_path, new_path):
    for element in folder_path.iterdir():
        if element.is_dir() and element.name.startswith('.'):
            continue
        if element.is_dir():
            replace_python_paths(element, current_path, new_path)
        if element.is_file() and element.name.endswith('.py'):
            replace_python_paths_in_file(element, current_path, new_path)

def replace_python_paths_in_file(file_path, current_path, new_path):
    with open(file_path, "r") as f:
        old_content = f.read()
    new_content = old_content.replace(str(current_path), str(new_path))
    if(old_content != new_content):
        make_file_writable(file_path)
        with open(file_path, "w") as f:
            f.write(new_content)

def make_file_writable(file_path):
    st = os.stat(file_path)
    write_bits = stat.S_IWUSR | stat.S_IWGRP | stat.S_IWOTH
    os.chmod(file_path, st.st_mode | write_bits)