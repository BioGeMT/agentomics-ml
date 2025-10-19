from pathlib import Path
import shutil

def save_splits_to_fallback(config):
    runs_dir = config.runs_dir
    fallbacks_dir = config.fallbacks_dir
    run_dir = Path(runs_dir) / config.agent_id
    fallback_dir = Path(fallbacks_dir) / config.agent_id
    run_dir.mkdir(parents=True, exist_ok=True)
    fallback_dir.mkdir(parents=True, exist_ok=True)

    train_name = 'train.csv'
    val_name = 'validation.csv'

    if (run_dir / train_name).exists():
        shutil.copy2(run_dir / train_name, fallback_dir / train_name)
    else:
        print("TRAIN CSV TO SAVE AS FALLBACK NOT FOUND")
    if (run_dir / val_name).exists():
        shutil.copy2(run_dir / val_name, fallback_dir / val_name)
    else:
        print("VALIDATION CSV TO SAVE AS FALLBACK NOT FOUND")

def load_fallbacks_to_rundir(config):
    runs_dir = config.runs_dir
    fallbacks_dir = config.fallbacks_dir
    run_dir = Path(runs_dir) / config.agent_id
    fallback_dir = Path(fallbacks_dir) / config.agent_id
    train_name = 'train.csv'
    val_name = 'validation.csv'

    if (fallback_dir / train_name).exists():
        shutil.copy2(fallback_dir / train_name, run_dir / train_name)
    else:
        print("TRAIN CSV FALLBACK NOT FOUND")
    if (fallback_dir / val_name).exists():
        shutil.copy2(fallback_dir / val_name, run_dir / val_name)
    else:
        print("VALIDATION CSV FALLBACK NOT FOUND")