import argparse
import json
from pathlib import Path
import dotenv
import wandb
import statistics
import pandas as pd

from utils.config import Config
from run_logging.logging_helpers import log_inference_stage_and_metrics
from run_logging.wandb_setup import resume_wandb_run
from run_logging.evaluate_log_run import get_metrics
from utils.biomlbench_target_utils import get_target_col_from_description

def run_test_evaluation(config_path, predictions_path, labeled_test_path, label_col, output_metrics_file, biomlbench_grade_dict):
    dotenv.load_dotenv()
    config = load_run_config(config_path)
    resume_wandb_run(config)
    is_proteingym = 'fitness_score_fold_random_5' in pd.read_csv(predictions_path).columns
    if(is_proteingym):
        print('PROTEINGYM TEST EVAL DETECTED')
        all_metrics = {}
        unique_original_metrics = set()
        fold_cols = ['fold_random_5','fold_modulo_5','fold_contiguous_5']
        for fold_col in fold_cols:
            fold_pred_col = f'fitness_score_{fold_col}'
            metrics = get_metrics(
                pred_col=fold_pred_col,
                results_file=predictions_path,
                test_file=labeled_test_path,
                output_file=str(output_metrics_file) + f'_{fold_col}',
                numeric_label_col=label_col,
                delete_preds=False,
                task_type=config.task_type,
            )
            for k in metrics.keys():
                unique_original_metrics.add(k)
                metrics[f'{k}_{fold_col}'] = metrics.pop(k)
            all_metrics.update(metrics)
        for metric in unique_original_metrics:
            all_metrics[metric] = statistics.mean([all_metrics[f'{metric}_{fold_col}'] for fold_col in fold_cols]) #take avg over all fold metrics (biomlbench style)
        log_inference_stage_and_metrics(2, metrics=all_metrics, task_type=config.task_type)
        log_biomlbench_grade(biomlbench_grade_dict)
    else:
        metrics = get_metrics(
            results_file=predictions_path,
            test_file=labeled_test_path,
            output_file=output_metrics_file,
            numeric_label_col=label_col,
            delete_preds=False,
            task_type=config.task_type,
        )
        log_inference_stage_and_metrics(2, metrics=metrics, task_type=config.task_type)
        log_biomlbench_grade(biomlbench_grade_dict)

def load_submission_jsonl(jsonl_path):
    with open(jsonl_path, 'r') as f:
        line = f.readline().strip()
        data = json.loads(line)

    return {
        'task_id': data['task_id'],
        'submission_path': data['submission_path'],
        'logs_path': data['logs_path'],
        'code_path': data['code_path']
    }

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
          'task_type': config_dict['task_type'],
      }
     
    config = Config(**config_constructor_params)
    config.wandb_run_id = config_dict.get('wandb_run_id')

    return config

def log_biomlbench_grade(grade_dict):
    for k,v in grade_dict.items():
        wandb.log({f"biomlbench/{k}": v})

def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('--results-dir', type=str, required=True, help='Path to the biomlbench run results dir')
    parser.add_argument('--grade-json', type=str, required=True, help='String version of the grade json dictionary from running biomlbench grade-sample')
    args = parser.parse_args()

    grade_dict = json.loads(args.grade_json)

    results_dir = Path(args.results_dir)
    submission_info_path = results_dir / 'submission.jsonl'
    run_info = load_submission_jsonl(submission_info_path)

    code_dirs = [dir for dir in list(Path(run_info['code_path']).iterdir()) if dir.name != 'reports']
    assert len(code_dirs) == 1, code_dirs
    config_path = code_dirs[0]/'config.json'

    preds_path = Path(run_info['submission_path']).parent / 'submission_extended.csv'
    task_id = run_info['task_id']
    labeled_test_path = Path(f'~/.cache/bioml-bench/data/{task_id}/prepared/private/answers.csv').expanduser()
    desc_path = Path(f'~/.cache/bioml-bench/data/{task_id}/prepared/public/description.md').expanduser()

    target_col = get_target_col_from_description(desc_path)

    run_test_evaluation(
        config_path=config_path,
        predictions_path=preds_path,
        labeled_test_path=labeled_test_path,
        label_col=target_col,
        output_metrics_file=submission_info_path.parent / 'test_metrics.json',
        biomlbench_grade_dict=grade_dict,
    )

if __name__ == "__main__":
    main()

