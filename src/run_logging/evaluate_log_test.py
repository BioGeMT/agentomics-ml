import json
from pathlib import Path
import argparse

from utils.config import Config
from utils.report_logger import add_final_test_metrics_to_best_report
from run_logging.evaluate_log_run import run_inference_and_log
from run_logging.logging_helpers import log_inference_stage_and_metrics
from run_logging.wandb_setup import resume_wandb_run
from utils.snapshots import replace_snapshot_path_with_relative

def run_test_evaluation(workspace_dir):
    config = load_run_config(workspace_dir)
    resume_wandb_run(config)

    snapshot_dir = config.snapshots_dir / config.agent_id
    snapshot_inference_script = snapshot_dir / "inference.py"

    if not snapshot_inference_script.exists():
        print("TEST EVAL SKIPPED: No snapshot found from previous runs.")
        print(f"Expected snapshot at: {snapshot_dir}")
        log_inference_stage_and_metrics(1, task_type=config.task_type)
        return

    print("\nRunning final test evaluation...")
    try:
        run_inference_and_log(config, iteration=None, evaluation_stage='test', use_best_snapshot=True)
        add_final_test_metrics_to_best_report(config)
    except Exception as e:
        print('FINAL TEST EVAL FAIL', str(e))
        log_inference_stage_and_metrics(1, task_type=config.task_type)

    replace_snapshot_path_with_relative(snapshot_dir = config.snapshots_dir / config.agent_id)

def load_run_config(workspace_dir):
    config_path = workspace_dir.resolve() / "config.json"
    with open(config_path, 'r') as f:
        config_dict = json.load(f)

    config_constructor_params = {      
          'agent_id': config_dict['agent_id'],                                                                                      
          'model_name': config_dict['model_name'],
          'feedback_model_name': config_dict['feedback_model_name'],
          'dataset': config_dict['dataset'],
          'tags': config_dict['tags'],
          'val_metric': config_dict['val_metric'],
          'workspace_dir': Path(config_dict['workspace_dir']),
          'prepared_datasets_dir': Path(config_dict['prepared_dataset_dir']).parent,
          'prepared_test_sets_dir': Path(config_dict['prepared_test_set_dir']).parent,
          'agent_datasets_dir': Path(config_dict['agent_dataset_dir']).parent,
          'user_prompt': config_dict['user_prompt'],
          'iterations': config_dict['iterations'],
      }
     
    config = Config(**config_constructor_params)
    config.wandb_run_id = config_dict.get('wandb_run_id')

    return config

def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('--workspace-dir', type=Path, default=Path('/workspace').resolve(), help='Path to workspace directory')
    args = parser.parse_args()

    run_test_evaluation(args.workspace_dir)

if __name__ == "__main__":
    main()