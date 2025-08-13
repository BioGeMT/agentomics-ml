import re
import shutil
from pathlib import Path

def get_metrics_from_file(file_path):
    """
    Read metrics from a file with enhanced error handling and debugging.
    
    Args:
        file_path: Path to the metrics file
        
    Returns:
        Dict of metrics
        
    Raises:
        Exception: If file doesn't exist, is empty, or has invalid format
    """
    file_path = Path(file_path)
    
    print(f"🔍 Reading metrics from: {file_path}")
    
    if not file_path.exists():
        raise Exception(f"Metrics file does not exist: {file_path}")
    
    file_size = file_path.stat().st_size
    print(f"   📊 File size: {file_size} bytes")
    
    if file_size == 0:
        raise Exception(f"Metrics file is empty: {file_path}")
    
    try:
        metrics = {}
        with open(file_path, "r") as f:
            content = f.read().strip()
            print(f"   📄 File content: {content}")
            
            if not content:
                raise Exception(f"Metrics file contains no data: {file_path}")
            
            for line_num, line in enumerate(content.split('\n'), 1):
                line = line.strip()
                if not line:
                    continue
                    
                if ": " not in line:
                    raise Exception(f"Invalid format in metrics file {file_path} at line {line_num}: '{line}' (expected 'key: value')")
                
                try:
                    key, value = line.split(": ", 1)
                    metrics[key] = float(value)
                except ValueError as e:
                    raise Exception(f"Could not parse value '{value}' for key '{key}' in {file_path} at line {line_num}: {e}")
        
        print(f"   ✅ Successfully parsed metrics: {metrics}")
        return metrics
        
    except Exception as e:
        print(f"   ❌ Error reading metrics file {file_path}: {e}")
        raise

def get_valid_and_train_metrics(base_path):
    """
    Get validation and training metrics from files with enhanced error handling.
    
    Args:
        base_path: Base directory containing metrics files
        
    Returns:
        Dict of combined metrics with prefixes
    """
    base_path = Path(base_path)
    val_metrics_file = base_path / "validation_metrics.txt"
    train_metrics_file = base_path / "train_metrics.txt"
    
    print(f"🔍 Getting validation and training metrics from: {base_path}")
    print(f"   📁 Validation metrics file: {val_metrics_file}")
    print(f"   📁 Training metrics file: {train_metrics_file}")
    
    all_metrics = {}
    
    # Try to get validation metrics
    try:
        val_metrics = get_metrics_from_file(val_metrics_file)
        val_metrics = {f"validation/{k}": v for k, v in val_metrics.items()}
        all_metrics.update(val_metrics)
        print(f"   ✅ Validation metrics loaded: {val_metrics}")
    except Exception as e:
        print(f"   ❌ Could not load validation metrics: {e}")
        # Add placeholder metrics to maintain structure for both classification and regression
        all_metrics.update({
            "validation/AUPRC": -1,
            "validation/AUROC": -1,
            "validation/ACC": -1,
            "validation/MSE": -1,
            "validation/RMSE": -1,
            "validation/MAE": -1,
            "validation/R2": -1,
        })
    
    # Try to get training metrics
    try:
        train_metrics = get_metrics_from_file(train_metrics_file)
        train_metrics = {f"train/{k}": v for k, v in train_metrics.items()}
        all_metrics.update(train_metrics)
        print(f"   ✅ Training metrics loaded: {train_metrics}")
    except Exception as e:
        print(f"   ❌ Could not load training metrics: {e}")
        print(f"   ❌ This is likely why training metrics are not appearing in W&B!")
        # Add placeholder metrics to maintain structure for both classification and regression
        all_metrics.update({
            "train/AUPRC": -1,
            "train/AUROC": -1,
            "train/ACC": -1,
            "train/MSE": -1,
            "train/RMSE": -1,
            "train/MAE": -1,
            "train/R2": -1,
        })
    
    print(f"   📊 Combined metrics: {all_metrics}")
    return all_metrics

def get_best_metrics(config):
    if(not best_metrics_exists(config)):
        return {}
    else:
        return get_valid_and_train_metrics(config.snapshot_dir / config.agent_id)

def get_new_metrics(config):
    if(not new_metrics_exists(config)):
        return {}
    else:
        return get_valid_and_train_metrics(config.workspace_dir / config.agent_id)

def get_new_and_best_metrics(config):
    return get_new_metrics(config), get_best_metrics(config)

def best_metrics_exists(config):
    best_metrics_path = config.snapshot_dir / config.agent_id / "validation_metrics.txt"
    if(best_metrics_path.is_file()):
        return True
    return False

def new_metrics_exists(config):
    new_metrics_path = config.workspace_dir / config.agent_id / "validation_metrics.txt"
    if(new_metrics_path.is_file()):
        return True
    return False

def is_new_best(config):
    if(not best_metrics_exists(config)):
        return True
    
    new_metrics, best_metrics = get_new_and_best_metrics(config)

    #TODO parametrize improvement threshold
    necessary_improvement = 0
    is_new_best = new_metrics[f"validation/{config.best_metric}"] > best_metrics[f"validation/{config.best_metric}"] + necessary_improvement
    print(f"is_new_best: {is_new_best}")
    print(f"New metrics: {new_metrics}")
    print(f"Best metrics: {best_metrics}")
    return is_new_best
       
def delete_snapshot(snapshot_dir):
    if snapshot_dir.exists():
        shutil.rmtree(snapshot_dir)

def snapshot(config, iteration, delete_old_snapshot=True):
    run_dir = config.workspace_dir / config.agent_id
    snapshot_dir = config.snapshot_dir / config.agent_id

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
        # print(f"Element: {element}")
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
        print(f"Snapshotting iteration number")
        f.write(str(iteration))

def get_best_iteration(config):
    snapshot_dir = config.snapshot_dir / config.agent_id
    iteration_file = snapshot_dir / "iteration_number.txt"
    if iteration_file.exists():
        with open(iteration_file, 'r') as f:
            return int(f.read().strip())
    return 0

def replace_workspace_path_with_snapshots(workspace_dir, snapshot_dir, absolute_path_snapshot_file):
    # Replaces hard-coded paths in the files to point to the snapshot dir
    with open(absolute_path_snapshot_file, "r") as f:
        old_content = f.read()
    new_content = old_content.replace(str(workspace_dir), str(snapshot_dir))
    if(old_content != new_content):
        print(f"Replaced {workspace_dir} with {snapshot_dir} in {absolute_path_snapshot_file}")
        with open(absolute_path_snapshot_file, "w") as f:
            f.write(new_content)