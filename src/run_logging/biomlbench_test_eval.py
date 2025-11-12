import argparse
import json
from pathlib import Path

from utils.config import Config
from run_logging.logging_helpers import log_inference_stage_and_metrics
from run_logging.wandb_setup import resume_wandb_run
from run_logging.evaluate_log_run import get_metrics

def run_test_evaluation(config_path, predictions_path, labeled_test_path, label_col, output_metrics_file):
    config = load_run_config(config_path)
    resume_wandb_run(config)

    metrics = get_metrics(
        results_file=predictions_path,
        test_file=labeled_test_path,
        output_file=output_metrics_file,
        numeric_label_col=label_col,
        delete_preds=False,
        task_type=config.task_type,
    )
    log_inference_stage_and_metrics(2, metrics=metrics, task_type=config.task_type)

def load_run_config(config_path):
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
    parser.add_argument('--config-path', type=str, required=True, help='Path to config JSON file')
    parser.add_argument('--predictions-path', type=str, required=True, help='Path to predictions CSV file')
    parser.add_argument('--labeled-test-path', type=str, required=True, help='Path to labeled test CSV file')
    parser.add_argument('--label-col', type=str, required=True, help='Name of the label column')
    parser.add_argument('--output-metrics-file', type=str, required=True, help='Path to output metrics file')
    args = parser.parse_args()

    run_test_evaluation(
        config_path=args.config_path,
        predictions_path=args.predictions_path,
        labeled_test_path=args.labeled_test_path,
        label_col=args.label_col,
        output_metrics_file=args.output_metrics_file
    )

if __name__ == "__main__":
    main()

