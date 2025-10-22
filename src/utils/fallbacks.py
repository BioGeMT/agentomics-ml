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

def load_fallbacks_to_rundir(config, iteration):
    runs_dir = config.runs_dir
    fallbacks_dir = config.fallbacks_dir
    run_dir = Path(runs_dir) / config.agent_id
    fallback_dir = Path(fallbacks_dir) / config.agent_id
    train_name = 'train.csv'
    val_name = 'validation.csv'

    failed_to_retrieve = False
    if (fallback_dir / train_name).exists():
        shutil.copy2(fallback_dir / train_name, run_dir / train_name)
    else:
        print("TRAIN CSV FALLBACK NOT FOUND")
        failed_to_retrieve = True

    if (fallback_dir / val_name).exists():
        shutil.copy2(fallback_dir / val_name, run_dir / val_name)
    else:
        print("VALIDATION CSV FALLBACK NOT FOUND")
        failed_to_retrieve = True

    if(failed_to_retrieve):
        # If no split was successful yet, increase the allowed split iteration budget
        if(not config.can_iteration_split_data(iteration+1)): #check if next iter can split
            print("Increasing split iteration budget due to a nonexisting split fallback")
            config.split_allowed_iterations = config.split_allowed_iterations + 1