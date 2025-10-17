import asyncio
import os
import argparse
import shutil
import pandas as pd
import subprocess
from pathlib import Path

from utils.dataset_utils import prepare_dataset
from run_agent import run_experiment
from utils.create_user import create_agent_id

def setup_agentomics_folder_structure_and_files(description_path, train_data_path, target_col, task_type, dataset_name):
    os.mkdir('/home/workspace')
    os.mkdir('/home/workspace/datasets')

    os.mkdir('/home/agent/raw_datasets')
    os.mkdir(f'/home/agent/raw_datasets/{dataset_name}')

    shutil.copy(description_path, f'/home/agent/raw_datasets/{dataset_name}/dataset_description.md')
    shutil.copy(train_data_path, f'/home/agent/raw_datasets/{dataset_name}/train.csv')

    os.mkdir('/home/agent/prepared_datasets')
    os.mkdir(f'/home/agent/prepared_datasets/{dataset_name}')

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
    output_path = f'{snapshots_dir}/{run_name}/predictions.csv'

    command_dir_ensurance = f"cd {os.path.dirname(inference_path)} && "
    command_prefix=f"conda run -p {env_path} --no-capture-output"
    command = f"{command_dir_ensurance} {command_prefix} python {inference_path} --input {input_path} --output {output_path}"
    inference_out = subprocess.run(command, shell=True, executable="/bin/bash", capture_output=True, check=False)
    if inference_out.returncode != 0:
        print("Error during inference:")
        print(inference_out.stderr.decode())
    return output_path

def copy_and_format_predictions_for_biomlbench(preds_source_path, preds_dest_path, target_col):
    preds_df = pd.read_csv(preds_source_path).reset_index()
    preds_df['id'] = preds_df.index
    preds_df = preds_df[['id','prediction']].rename(columns={'prediction': target_col})
    preds_df.to_csv(preds_dest_path, index=False)

def extract_dataset_name_from_description(description_path):
    with open(description_path, 'r') as f:
        first_line = f.readline()
        return first_line.lstrip('#').strip()

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
    parser.add_argument('--val-metric', type=str, help='Validation metric to use')
    parser.add_argument('--iterations', type=int, help='Number of iterations to run')
    parser.add_argument('--target-col', type=str, help='Name of the target column')
    parser.add_argument('--task-type', type=str, help='Task type: classification or regression')
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
    train_data = '/home/data/train.csv'
    sample_submission = '/home/data/sample_submission.csv'
    # Where to output predictions for biomlbench

    dataset_name = extract_dataset_name_from_description(description_path)

    setup_agentomics_folder_structure_and_files(
        description_path = description_path, 
        train_data_path = train_data, 
        target_col=args.target_col, 
        task_type=args.task_type,
        dataset_name=dataset_name
    )

    def generate_preds_for_biomlbench(config):
        print('---------------------------------')
        print('-GENERATING PREDS FOR BIOMLBENCH-')
        print('---------------------------------')

        test_no_label = '/home/data/test_features.csv'
        SUBMISSION_DIR = os.getenv('SUBMISSION_DIR', '')
        submission_path = os.path.join(SUBMISSION_DIR, 'submission.csv')

        try:
            predictions_path = run_inference_on_test_data(test_no_label)
            copy_and_format_predictions_for_biomlbench(
                preds_source_path=predictions_path,
                preds_dest_path=submission_path,
                target_col=args.target_col #passed from outside the fn, refactor into reading prepared yaml metadata
            )
            copy_dir(source_dir='/home/workspace/snapshots', dest_dir=CODE_DIR)
        except Exception as e:
            import traceback
            print('-------TRACEBACK------TRACEBACK------')
            print(traceback.format_exc())
            print('-------TRACEBACK------TRACEBACK------')

        print('---------------------------------')
        print('- FINISHED PREDS FOR BIOMLBENCH -')
        print('---------------------------------')

    asyncio.run(run_experiment(
        model=args.model,
        dataset_name=dataset_name, # Name doesnt matter for biomlbench, has his own run structure, but matters for our logging
        val_metric=args.val_metric,
        iterations=args.iterations,
        user_prompt=args.user_prompt,
        # Testing prompt
        # user_prompt="Create only small CPU-only model like linear regression with small amounts of parameters and epochs",
        workspace_dir = '/home/workspace',
        prepared_datasets_dir= '/home/agent/prepared_datasets',
        prepared_test_sets_dir= '/home/agent/prepared_test_sets',
        agent_datasets_dir= '/home/workspace/datasets',
        tags=[],
        provider=args.provider,
        split_allowed_iterations=args.split_allowed_iterations,
        on_new_best_callbacks=[generate_preds_for_biomlbench],
    ))
