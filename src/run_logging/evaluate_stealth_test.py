import json
import argparse
from pathlib import Path
from utils.config import Config
from eval.evaluate_result import get_metrics
from run_logging.wandb_setup import resume_wandb_run
import wandb

def evaluate_stealth_test(dataset, test_output_dir, experiment_folder):
    test_output_dir = Path(test_output_dir)
    experiment_folder = Path(experiment_folder)
    config_file = experiment_folder / "extras" / "config.json"
    with open(config_file) as f:
        config_dict = json.load(f)

    prepared_test_sets_dir = Path('/repository/prepared_test_sets')
    prepared_datasets_dir = Path('/repository/prepared_datasets')
    with open(prepared_datasets_dir / dataset / "metadata.json") as f:
        dataset_metadata = json.load(f)

    test_file = prepared_test_sets_dir / dataset / "test.csv"
    config_constructor_params = {
        'agent_id': config_dict['agent_id'],
        'model_name': config_dict['model_name'],
        'feedback_model_name': config_dict['feedback_model_name'],
        'dataset': config_dict['dataset'],
        'tags': config_dict['tags'],
        'val_metric': config_dict['val_metric'],
        'workspace_dir': Path(config_dict['workspace_dir']),
        'prepared_datasets_dir': Path(config_dict['prepared_dataset_dir']).parent,
        'prepared_test_sets_dir': prepared_test_sets_dir,
        'agent_datasets_dir': Path(config_dict['agent_dataset_dir']).parent,
        'user_prompt': config_dict['user_prompt'],
        'iterations': config_dict['iterations'],
    }
    config = Config(**config_constructor_params)
    config.wandb_run_id = config_dict.get('wandb_run_id')
    run = resume_wandb_run(config)

    pred_files = list(test_output_dir.glob("iteration_*_test_predictions.csv"))
    pred_files.sort(key=lambda f: int(f.stem.split("_")[1]))
    for pred_file in pred_files:
        iteration_name = pred_file.stem.replace("_test_predictions", "")
        try:
            metrics = get_metrics(
                results_file=pred_file,
                test_file=test_file,
                output_file=None,
                numeric_label_col=dataset_metadata['numeric_label_col'],
                delete_preds=False,
                task_type=dataset_metadata['task_type']
            )
            for metric_name, metric_value in metrics.items():
                # print(f"stealth_test/{metric_name} = {metric_value} at step {int(iteration_name.split('_')[1])}")
                run.define_metric(step_metric = 'iteration_index', name = metric_name)
                wandb.log({f"stealth_test/{metric_name}": metric_value, 'iteration_index': int(iteration_name.split("_")[1])})
        except Exception as e:
            print(f"Failed to evaluate {iteration_name}: {e}")
    wandb.finish()

def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('--dataset', required=True)
    parser.add_argument('--test-output-dir', required=True)
    parser.add_argument('--experiment-folder', required=True)
    args = parser.parse_args()
    evaluate_stealth_test(
        args.dataset,
        args.test_output_dir,
        args.experiment_folder
    )

if __name__ == "__main__":
    main()