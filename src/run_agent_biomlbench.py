import asyncio
import os
import argparse
import shutil
import pandas as pd
import subprocess
from pathlib import Path
import yaml
import time
import wandb

from utils.dataset_utils import prepare_dataset
from run_agent import run_experiment
from utils.create_user import create_agent_id
from utils.biomlbench_target_utils import get_target_col_from_description

def setup_agentomics_folder_structure_and_files(description_path, train_data_path, task_type, dataset_name, is_proteingym):
    os.mkdir('/home/workspace')
    os.mkdir('/home/workspace/datasets')

    os.mkdir('/home/agent/raw_datasets')
    os.mkdir(f'/home/agent/raw_datasets/{dataset_name}')

    if(is_proteingym):
        # Only copy the main heading and ## Description section
        with open(description_path, 'r') as f:
            lines = f.readlines()

        output_lines = []
        in_description = False

        for i, line in enumerate(lines):
            # Always include the main heading (first line starting with #)
            if i == 0 or (not output_lines and line.startswith('#') and not line.startswith('##')):
                output_lines.append(line)
            # Start capturing when we find ## Description
            elif line.strip() == '## Description':
                in_description = True
                output_lines.append('\n')
                output_lines.append(line)
            # Stop capturing when we encounter the next ## section
            elif in_description and line.startswith('##'):
                break
            # Capture lines within the Description section
            elif in_description:
                output_lines.append(line)

        # Write the filtered content
        with open(f'/home/agent/raw_datasets/{dataset_name}/dataset_description.md', 'w') as f:
            f.writelines(output_lines)
    else:
        shutil.copy(description_path, f'/home/agent/raw_datasets/{dataset_name}/dataset_description.md')

    if is_proteingym:
        # Copy the csv, but leave out the following columns: fold_random_5,fold_modulo_5,fold_contiguous_5
        df = pd.read_csv(train_data_path)
        df = df[df['fold_random_5'] != -1]
        columns_to_drop = ['fold_random_5', 'fold_modulo_5', 'fold_contiguous_5']
        df = df.drop(columns=[col for col in columns_to_drop if col in df.columns])
        df.to_csv(f'/home/agent/raw_datasets/{dataset_name}/train.csv', index=False)
    else:
        shutil.copy(train_data_path, f'/home/agent/raw_datasets/{dataset_name}/train.csv')

    os.mkdir('/home/agent/prepared_datasets')
    os.mkdir(f'/home/agent/prepared_datasets/{dataset_name}')

    target_col = get_target_col_from_description(description_path)

    prepare_dataset(
        dataset_dir=f'/home/agent/raw_datasets/{dataset_name}',
        target_col=target_col,
        positive_class=None,
        negative_class=None,
        task_type=task_type,
        output_dir='/home/agent/prepared_datasets',
        test_sets_output_dir='/home/agent/prepared_test_sets',
    )

def run_inference_on_test_data(test_data_path):
    snapshots_dir = '/home/workspace/snapshots'
    run_names = os.listdir(snapshots_dir)
    assert len(run_names) == 1, "Expected exactly one run"
    run_name = run_names[0]

    env_path = Path(f"{snapshots_dir}/{run_name}") / ".conda"/ "envs" / f"{run_name}_env"
    inference_path = Path(f"{snapshots_dir}/{run_name}") / "inference.py"
    input_path = test_data_path
    artifacts_dir_path = f'{snapshots_dir}/{run_name}/training_artifacts'
    output_path = f'{snapshots_dir}/{run_name}/predictions.csv'

    command_dir_ensurance = f"cd {os.path.dirname(inference_path)} && "
    command_prefix=f"conda run -p {env_path} --no-capture-output"
    command = f"{command_dir_ensurance} {command_prefix} python \"{inference_path}\" --input \"{input_path}\" --output \"{output_path}\" --artifacts-dir \"{artifacts_dir_path}\""
    inference_out = subprocess.run(command, shell=True, executable="/bin/bash", capture_output=True, check=False)
    if inference_out.returncode != 0:
        print("Error during inference:")
        print(inference_out.stderr.decode())
    print('------INFERENCE OUTPUTS-------')
    print(inference_out.stdout)
    print(inference_out.stderr)
    print('---END OF INFERENCE OUTPUTS---')
    return output_path

def generate_preds_for_biomlbench_proteingym(config):
    print('---------------------------------')
    print('-GENERATING PREDS FOR BIOMLBENCH-')
    print('---------------------------------')

    # test_no_label = '/home/data/test_features.csv'
    SUBMISSION_DIR = os.getenv('SUBMISSION_DIR')
    CODE_DIR= os.getenv('CODE_DIR')
    submission_path = os.path.join(SUBMISSION_DIR, 'submission.csv')
    temp_csv_files = []  # Track temp files for cleanup (initialize before try block)
    snapshots_dir = '/home/workspace/snapshots'
    run_names = os.listdir(snapshots_dir)
    run_name = run_names[0]
    with open(Path(f"{snapshots_dir}/{run_name}") / "iteration_number.txt", 'r') as f:
        iteration = int(f.read().strip())

    start_time = time.time()
    try:
        # id,sequence,fitness_score,fold_random_5,fold_modulo_5,fold_contiguous_5
        assert len(run_names) == 1, "Expected exactly one run"

        env_path = Path(f"{snapshots_dir}/{run_name}") / ".conda"/ "envs" / f"{run_name}_env"
        inference_script_path = Path(f"{snapshots_dir}/{run_name}") / "inference.py"
        og_train_data = pd.read_csv('/home/data/data.csv')
        og_train_data = og_train_data[og_train_data['fold_random_5'] != -1]
        final_predictions_path = f'{snapshots_dir}/{run_name}/predictions.csv'
        train_script_path = Path(f"{snapshots_dir}/{run_name}") / 'train.py'
        cols_to_keep = ['id','sequence','fitness_score']
        assert all([col in og_train_data.columns for col in cols_to_keep])

        # Split the og_train_data based on fold_random_5, fold_modulo_5, fold_contiguous_5 columns (3 different ways to do cross-validation)
        # Train on the 4-other folds, make predictions for the fifth fold -> generate 3 different prediction files (for each of the split styles)
        # Return them as extra columns in the dataset (prefixed with the split style name)
        print('------CROSS VALIDATION LEAKAGE PREVENTION START-------')
        fold_col_to_preds = []

        for fold_col in ['fold_random_5','fold_modulo_5','fold_contiguous_5']:
            fold_predictions_dfs = []
            for current_test_fold_value in og_train_data[fold_col].unique():
                # Pick an arbitrary validation that will be additionally separated from the train folds for validation purposes
                num_folds = len(og_train_data[fold_col].unique())
                validation_fold_value = (current_test_fold_value + 1) % num_folds
                assert validation_fold_value != current_test_fold_value

                cols_to_keep = ['id','sequence','fitness_score']

                test_df = og_train_data[og_train_data[fold_col] == current_test_fold_value][cols_to_keep]
                valid_df = og_train_data[og_train_data[fold_col] == validation_fold_value][cols_to_keep]
                train_df = og_train_data[(og_train_data[fold_col] != current_test_fold_value) & (og_train_data[fold_col] != validation_fold_value)][cols_to_keep]

                assert len(train_df) > 0, f"Empty training set for fold {current_test_fold_value}"
                assert len(valid_df) > 0, f"Empty validation set for fold {current_test_fold_value}"
                assert len(test_df) > 0, f"Empty test set for fold {current_test_fold_value}"

                test_df = test_df.rename(columns={'fitness_score':'numeric_label'}, errors='raise')
                valid_df = valid_df.rename(columns={'fitness_score':'numeric_label'}, errors='raise')
                train_df = train_df.rename(columns={'fitness_score':'numeric_label'}, errors='raise')

                # Assert there's no intersection between these
                train_ids = set(train_df['id'])
                valid_ids = set(valid_df['id'])
                test_ids = set(test_df['id'])
                assert len(train_ids & valid_ids) == 0, "Intersection found between train and validation"
                assert len(train_ids & test_ids) == 0, "Intersection found between train and test"
                assert len(valid_ids & test_ids) == 0, "Intersection found between validation and test"

                # Export to csv files
                train_csv_path = f'{snapshots_dir}/{run_name}/{fold_col}_{current_test_fold_value}_train.csv'
                valid_csv_path = f'{snapshots_dir}/{run_name}/{fold_col}_{current_test_fold_value}_valid.csv'
                test_csv_path = f'{snapshots_dir}/{run_name}/{fold_col}_{current_test_fold_value}_test.csv'
                artifacts_dir = f'{snapshots_dir}/{run_name}/{fold_col}_{current_test_fold_value}_artifacts'
                predictions_csv_path = f'{snapshots_dir}/{run_name}/{fold_col}_{current_test_fold_value}_predictions.csv'

                train_df.to_csv(train_csv_path, index=False)
                valid_df.to_csv(valid_csv_path, index=False)
                test_df.to_csv(test_csv_path, index=False)
                os.makedirs(artifacts_dir, exist_ok=True)

                temp_csv_files.extend([train_csv_path, valid_csv_path, test_csv_path, artifacts_dir, predictions_csv_path])

                command_prefix=f"conda run -p {env_path} --no-capture-output"

                train_command_dir_ensurance = f"cd {os.path.dirname(train_script_path)} && "
                training_command = f"{train_command_dir_ensurance} {command_prefix} python \"{train_script_path}\" --train-data \"{train_csv_path}\" --validation-data \"{valid_csv_path}\" --artifacts-dir \"{artifacts_dir}\""
                training_out = subprocess.run(training_command, shell=True, executable="/bin/bash", capture_output=True, check=False)
                if training_out.returncode != 0:
                    print("Error during training:")
                    print(training_out.stderr.decode())
                print('------TRAINING OUTPUTS-------')
                print(training_out.stdout.decode() if isinstance(training_out.stdout, bytes) else training_out.stdout)
                print(training_out.stderr.decode() if isinstance(training_out.stderr, bytes) else training_out.stderr)
                print('---END OF TRAINING OUTPUTS---')

                inference_command_dir_ensurance = f"cd {os.path.dirname(inference_script_path)} && "
                inference_command = f"{inference_command_dir_ensurance} {command_prefix} python \"{inference_script_path}\" --input \"{test_csv_path}\" --output \"{predictions_csv_path}\" --artifacts-dir \"{artifacts_dir}\""
                inference_out = subprocess.run(inference_command, shell=True, executable="/bin/bash", capture_output=True, check=False)
                if inference_out.returncode != 0:
                    print("Error during inference:")
                    print(inference_out.stderr.decode())
                print('------INFERENCE OUTPUTS-------')
                print(inference_out.stdout.decode() if isinstance(inference_out.stdout, bytes) else inference_out.stdout)
                print(inference_out.stderr.decode() if isinstance(inference_out.stderr, bytes) else inference_out.stderr)
                print('---END OF INFERENCE OUTPUTS---')

                # Check all the ids are there for easy merge/concat
                preds_df = pd.read_csv(predictions_csv_path)
                assert len(preds_df) == len(test_df), f"Mismatch: {len(preds_df)} predictions vs {len(test_df)} test samples"
                fold_predictions_dfs.append(preds_df)

            # Concatenate all fold predictions for this fold column
            fold_preds = pd.concat(fold_predictions_dfs, ignore_index=True)
            if 'prediction' in fold_preds.columns:
                fold_preds = fold_preds.rename(columns={'prediction': f'fitness_score_{fold_col}'}) #Col name required by biomlbench proteingym grade function
            fold_col_to_preds.append(fold_preds)

        # Combine all predictions based on ID
        final_predictions_df = fold_col_to_preds[0]
        for fold_preds_df in fold_col_to_preds[1:]:
            final_predictions_df = final_predictions_df.merge(fold_preds_df, on='id', how='inner')

        assert len(final_predictions_df) == len(og_train_data), 'predictions dont match the original data'
        final_predictions_df.to_csv(final_predictions_path, index=False)

        target_col = get_target_col_from_description()

        # For now, use the first fold's predictions for the submission (average could be used instead)
        copy_and_format_predictions_for_biomlbench(
            preds_source_path=final_predictions_path,
            preds_dest_path=submission_path,
            target_col=target_col, #passed from outside the fn, refactor into reading prepared yaml metadata
            is_proteingym=True,
        )
        
        # TODO how are we using the original predictions? We compute metrics based on one col only, biomlbench averages three metrics -> what to do?
        copy_original_predictions(final_predictions_path, os.path.join(SUBMISSION_DIR, 'submission_extended.csv'))
        copy_dir(source_dir='/home/workspace/snapshots', dest_dir=CODE_DIR)
        copy_dir(source_dir='/home/workspace/reports', dest_dir=Path(str(CODE_DIR))/'reports')
    except Exception as e:
        import traceback
        print('-------TRACEBACK------TRACEBACK------')
        print(traceback.format_exc())
        print('-------TRACEBACK------TRACEBACK------')
    finally:
        # Clean up temporary CSV files and artifacts directories
        try:
            for temp_path in temp_csv_files:
                if os.path.isfile(temp_path):
                    os.remove(temp_path)
                elif os.path.isdir(temp_path):
                    shutil.rmtree(temp_path)
        except Exception as cleanup_error:
            print(f"Warning: Error during cleanup: {cleanup_error}")

        # Log execution time to wandb
        elapsed_time = time.time() - start_time
        wandb.log({"proteingym_preds_generation_time_seconds": elapsed_time}, step=iteration)

    print('------CROSS VALIDATION LEAKAGE PREVENTION END-------')
    print('---------------------------------')
    print('- FINISHED PREDS FOR BIOMLBENCH -')
    print('---------------------------------')



def generate_preds_for_biomlbench(config):
    print('---------------------------------')
    print('-GENERATING PREDS FOR BIOMLBENCH-')
    print('---------------------------------')

    test_no_label = '/home/data/test_features.csv'
    SUBMISSION_DIR = os.getenv('SUBMISSION_DIR', '')
    submission_path = os.path.join(SUBMISSION_DIR, 'submission.csv')
    try:
        predictions_path = run_inference_on_test_data(test_no_label)
        target_col = get_target_col_from_description()
        copy_and_format_predictions_for_biomlbench(
            preds_source_path=predictions_path,
            preds_dest_path=submission_path,
            target_col=target_col #passed from outside the fn, refactor into reading prepared yaml metadata
        )
        copy_original_predictions(predictions_path, os.path.join(SUBMISSION_DIR, 'submission_extended.csv'))
        copy_dir(source_dir='/home/workspace/snapshots', dest_dir=CODE_DIR)
        copy_dir(source_dir='/home/workspace/reports', dest_dir=Path(str(CODE_DIR))/'reports')
    except Exception as e:
        import traceback
        print('-------TRACEBACK------TRACEBACK------')
        print(traceback.format_exc())
        print('-------TRACEBACK------TRACEBACK------')

    print('---------------------------------')
    print('- FINISHED PREDS FOR BIOMLBENCH -')
    print('---------------------------------')

def copy_and_format_predictions_for_biomlbench(preds_source_path, preds_dest_path, target_col, is_proteingym=False):
    if(is_proteingym):
        preds_df = pd.read_csv(preds_source_path)
        preds_df.to_csv(preds_dest_path, index=False)
    else:
        preds_df = pd.read_csv(preds_source_path)
        # preds_df['id'] = preds_df.index #TODO remove since id is kept now?
        preds_df = preds_df[['id','prediction']].rename(columns={'prediction': target_col})
        preds_df.to_csv(preds_dest_path, index=False)

def copy_original_predictions(preds_source_path, preds_dest_path):
    preds_df = pd.read_csv(preds_source_path)
    preds_df.to_csv(preds_dest_path, index=False)

def extract_dataset_name_from_description(description_path):
    with open(description_path, 'r') as f:
        first_line = f.readline().lstrip('#').strip()
        # If first line is empty, try the second line
        if not first_line:
            second_line = f.readline().lstrip('#').strip()
            return second_line
        return first_line
    
def extract_val_metric_from_description(description_path, is_proteingym):
    biomlbench_metric_to_agentomics_metric = {
        'mean_absolute_error': 'MAE',
        'pr_auc': "AUPRC",
        'pearsonr': "PEARSON",
        'roc_auc': "AUROC",
        'spearman': "SPEARMAN",
    }

    # TODO scout readmes and fill in
    metric_keywords = {
        'spearman correlation': 'spearman',
        'pearson correlation': 'pearsonr',
        'roc_auc': 'roc_auc',
        'auroc': 'roc_auc',
        'pr_auc': 'pr_auc',
        'auprc': 'pr_auc',
        'mean absolute error': 'mean_absolute_error',
        'mae': 'mean_absolute_error',
    }

    if is_proteingym:
        with open(description_path, 'r') as f:
            content = f.read()
            content_lower = content.lower()

            # Search for metric keywords in the description
            found_keywords = []
            for keyword, metric_name in metric_keywords.items():
                if keyword in content_lower:
                    found_keywords.append(keyword)
            assert len(found_keywords) ==1, found_keywords
            return biomlbench_metric_to_agentomics_metric[metric_keywords[found_keywords[0]]]
    else:
        with open(description_path, 'r') as f:
            for line in f.readlines():
                if '**Main Metric' in line:
                    biomlbench_metric = line.split("**Main Metric:**")[-1].strip()
                    return biomlbench_metric_to_agentomics_metric[biomlbench_metric]

def extract_task_type_from_val_metric(val_metric):
    if val_metric in ['AUPRC', 'AUROC']:
        return 'classification'
    if val_metric in ['MAE', 'PEARSON', 'SPEARMAN']:
        return 'regression'
    raise Exception('Unknown val metric, update parsing or metrics.')

def copy_dir(source_dir, dest_dir):
    if not os.path.exists(source_dir):
        raise FileNotFoundError(f"Source directory does not exist: {source_dir}")

    os.makedirs(dest_dir, exist_ok=True)
    for item in os.listdir(source_dir):
        src_path = os.path.join(source_dir, item)
        dst_path = os.path.join(dest_dir, item)

        if os.path.isdir(src_path):
            copy_dir(src_path, dst_path)
        else:
            shutil.copy2(src_path, dst_path)

def parse_args():
    parser = argparse.ArgumentParser()
    parser.add_argument('--model', type=str, help='Model id to use (depends on provider)')
    parser.add_argument('--iterations', type=int, help='Number of iterations to run')
    parser.add_argument('--timeout', type=int, help='Timeout in seconds')
    parser.add_argument('--tags', nargs='*', default=[], help='(Optional) Tags for a wandb run logging')
    parser.add_argument('--provider', type=str, default='openrouter', help='Provider name (e.g., openai, openrouter)')
    parser.add_argument('--user-prompt', type=str, default=None, help='Custom user prompt to guide the agent')
    parser.add_argument('--split-allowed-iterations', type=int, help='Number of initial iterations that allow the agent to split the data into training and validation sets')

    args = parser.parse_args()
    return args

if __name__ == '__main__':
    args = parse_args()

    # Env vars (passed by biomlbench docker image)
    # For submission extraction
    SUBMISSION_DIR = os.getenv('SUBMISSION_DIR')
    # For logs extraction (currently no logs linked)
    LOGS_DIR= os.getenv('LOGS_DIR')
    # For agent-generated code extraction (currently no code output linked)
    CODE_DIR= os.getenv('CODE_DIR')
    # Not extracted
    AGENT_DIR= os.getenv('AGENT_DIR')
    # For agent prediction extraction
    os.environ['AGENT_ID'] = create_agent_id()

    # Locations of data (passed by biomlbench docker image)
    description_path = '/home/data/description.md'
    sample_submission = '/home/data/sample_submission.csv'

    dataset_name = extract_dataset_name_from_description(description_path)
    if 'proteingym' in dataset_name.lower():
        is_proteingym = True
        print('Proteingym dataset detected')
    else:
        is_proteingym = False
    val_metric = extract_val_metric_from_description(description_path, is_proteingym=is_proteingym)
    task_type = extract_task_type_from_val_metric(val_metric)

    if is_proteingym:
        train_data = '/home/data/data.csv'
    else:
        train_data = '/home/data/train.csv'


    setup_agentomics_folder_structure_and_files(
        description_path = description_path, 
        train_data_path = train_data, 
        task_type=task_type,
        dataset_name=dataset_name,
        is_proteingym=is_proteingym,
    )

    asyncio.run(run_experiment(
        model=args.model,
        dataset_name=dataset_name, # Name doesnt matter for biomlbench, has his own run structure, but matters for our logging
        val_metric=val_metric,
        iterations=args.iterations,
        user_prompt=args.user_prompt,
        # Testing prompt
        # user_prompt="Create only small CPU-only model like linear regression with small amounts of parameters and epochs",
        workspace_dir = '/home/workspace',
        prepared_datasets_dir= '/home/agent/prepared_datasets',
        prepared_test_sets_dir= '/home/agent/prepared_test_sets',
        agent_datasets_dir= '/home/workspace/datasets',
        tags=args.tags,
        provider=args.provider,
        split_allowed_iterations=args.split_allowed_iterations,
        on_new_best_callbacks=[generate_preds_for_biomlbench_proteingym if is_proteingym else generate_preds_for_biomlbench],
        timeout=args.timeout,
    ))
