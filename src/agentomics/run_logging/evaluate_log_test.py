import json
from pathlib import Path
import argparse
import time

from utils.config import Config
from run_logging.evaluate_log_run import run_inference_and_log
from run_logging.logging_helpers import log_inference_stage_and_metrics, log_test_inference_duration
from run_logging.wandb_setup import resume_wandb_run
from utils.snapshots import replace_python_paths

def run_test_evaluation(workspace_dir, agent_id=None):
    start = time.time()
    print("\nRunning final test evaluation...")
    config = None
    try:
        config = load_run_config(extras_dir = Path(workspace_dir) / 'extras')
        resume_wandb_run(config)
        run_inference_and_log(config, iteration=None, evaluation_stage='test', use_best_snapshot=True)
    except Exception as e:
        print('FINAL TEST EVAL FAIL', str(e))
        if config is not None:
            log_inference_stage_and_metrics(1, task_type=config.task_type)
        else:
            log_inference_stage_and_metrics(1, task_type='classification') #fallback
    log_test_inference_duration(time.time() - start)

def load_run_config(extras_dir):
    config_path = extras_dir.resolve() / "config.json"
        
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
    parser.add_argument('--agent-id', type=str, help='Agent id for snapshot selection')
    args = parser.parse_args()

    run_test_evaluation(args.workspace_dir, agent_id=args.agent_id)

if __name__ == "__main__":
    main()
