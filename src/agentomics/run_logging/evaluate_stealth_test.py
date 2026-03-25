import json
import argparse
import tempfile
from pathlib import Path
from utils.config import Config
from eval.evaluate_result import get_metrics
from run_logging.wandb_setup import resume_wandb_run
import wandb
import pandas as pd
from run_agent_biomlbench import extract_task_type_from_val_metric

# Protein datasets that use CV retraining
PROTEINGYM_DATASETS = [
    "SPIKE_SARS2_Starr_2020_binding",
    "SPA_STAAU_Tsuboyama_2023_1LP1",
    "PSAE_PICP2_Tsuboyama_2023_1PSE_indels",
    "CBX4_HUMAN_Tsuboyama_2023_2K28",
    "Q8EG35_SHEON_Campbell_2022_indels",
    "CSN4_MOUSE_Tsuboyama_2023_1UFM_indels",
]

def is_proteingym_dataset(dataset_name):
    """Check if dataset is a proteingym dataset."""
    for pg_dataset in PROTEINGYM_DATASETS:
        if pg_dataset in dataset_name:
            return True
    return False

def evaluate_protein_cv_predictions(pred_file, train_file, task_type, numeric_label_col):
    """
    Evaluate protein CV predictions by comparing against ground truth.
    For protein datasets, predictions are made on CV folds and need special handling.
    """
    preds_df = pd.read_csv(pred_file)
    train_df = pd.read_csv(train_file)

    # Filter out test rows (fold_random_5 == -1) if fold column exists
    if 'fold_random_5' in train_df.columns:
        train_df = train_df[train_df['fold_random_5'] != -1]

    # Find prediction columns (fitness_score or fitness_score_fold_*)
    pred_cols = [col for col in preds_df.columns if col.startswith('fitness_score')]

    if len(pred_cols) == 0:
        raise ValueError(f"No fitness_score columns found in predictions: {preds_df.columns.tolist()}")

    # Determine ground truth column name
    if 'fitness_score' in train_df.columns:
        truth_col = 'fitness_score'
    elif 'numeric_label' in train_df.columns:
        truth_col = 'numeric_label'
    else:
        raise ValueError(f"Neither 'fitness_score' nor 'numeric_label' found in train data: {train_df.columns.tolist()}")

    # Merge predictions with ground truth
    merged = pd.merge(preds_df[['id'] + pred_cols], train_df[['id', truth_col]], on='id', how='inner')

    # The label column in merged dataframe
    label_col_in_merged = truth_col

    # Calculate metrics for each prediction column using temp directory
    all_metrics = {}
    with tempfile.TemporaryDirectory() as temp_dir:
        temp_dir = Path(temp_dir)

        for pred_col in pred_cols:
            # Use get_metrics with the prediction column
            # Results file should have id and prediction only (get_metrics will merge with test file for labels)
            temp_pred_df = merged[['id', pred_col]].rename(columns={pred_col: 'prediction'})
            temp_file = temp_dir / f"temp_{pred_file.stem}_{pred_col}.csv"
            temp_pred_df.to_csv(temp_file, index=False)

            # Test file should have id and numeric_label_col
            train_temp_file = temp_dir / f"temp_train_{pred_file.stem}.csv"
            merged[['id', label_col_in_merged]].rename(columns={label_col_in_merged: numeric_label_col}).to_csv(train_temp_file, index=False)

            metrics = get_metrics(
                results_file=temp_file,
                test_file=train_temp_file,
                output_file=None,
                numeric_label_col=numeric_label_col,
                delete_preds=False,
                task_type=task_type
            )

            # Store metrics with fold suffix if multiple folds
            if len(pred_cols) > 1:
                fold_suffix = pred_col.replace('fitness_score', '')
                for metric_name, metric_value in metrics.items():
                    all_metrics[f"{metric_name}{fold_suffix}"] = metric_value
            else:
                all_metrics.update(metrics)

    # Average metrics across folds if multiple, and return only base metric names
    if len(pred_cols) > 1:
        # Group metrics by base name and average
        base_metrics = {}
        for metric_name in list(all_metrics.keys()):
            if '_fold_' in metric_name:
                base_name = metric_name.split('_fold_')[0]
                if base_name not in base_metrics:
                    base_metrics[base_name] = []
                base_metrics[base_name].append(all_metrics[metric_name])

        # Replace all fold-specific metrics with averaged versions
        final_metrics = {}
        for base_name, values in base_metrics.items():
            final_metrics[base_name] = sum(values) / len(values)

        return final_metrics
    else:
        # Single fold - return metrics as-is
        return all_metrics

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

    is_proteingym = is_proteingym_dataset(dataset)

    if is_proteingym:
        # For protein datasets, use training data with fold columns
        test_file = prepared_datasets_dir / dataset / "train.csv"
    else:
        test_file = prepared_test_sets_dir / dataset / "test.csv"

    config_constructor_params = {
        'agent_id': config_dict['agent_id'],
        'model_name': config_dict['model_name'],
        'feedback_model_name': config_dict['feedback_model_name'],
        'dataset': config_dict['dataset'],
        'tags': config_dict['tags'],
        'val_metric': config_dict['val_metric'],
        'workspace_dir': Path(config_dict['workspace_dir']),
        'prepared_datasets_dir': prepared_datasets_dir,
        'prepared_test_sets_dir': prepared_test_sets_dir,
        'agent_datasets_dir': Path(config_dict['agent_dataset_dir']).parent,
        'user_prompt': config_dict['user_prompt'],
        'iterations': config_dict['iterations'],
        'task_type': extract_task_type_from_val_metric(config_dict['val_metric']),
    }
    config = Config(**config_constructor_params)
    config.wandb_run_id = config_dict.get('wandb_run_id')
    run = resume_wandb_run(config)

    pred_files = list(test_output_dir.glob("iteration_*_test_predictions.csv"))
    pred_files.sort(key=lambda f: int(f.stem.split("_")[1]))
    for pred_file in pred_files:
        iteration_name = pred_file.stem.replace("_test_predictions", "")
        try:
            if is_proteingym:
                # Use protein-specific evaluation
                metrics = evaluate_protein_cv_predictions(
                    pred_file=pred_file,
                    train_file=test_file,
                    task_type=dataset_metadata['task_type'],
                    numeric_label_col=dataset_metadata['numeric_label_col']
                )
            else:
                # Standard evaluation
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
            import traceback
            print(f"Failed to evaluate {iteration_name}: {e}")
            print(traceback.format_exc())
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