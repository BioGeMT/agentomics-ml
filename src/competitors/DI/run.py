import asyncio
import os
import dotenv
import sys
import argparse
import json
import subprocess

sys.path.append("/repository/src")
from run_logging.wandb import setup_logging
from run_logging.evaluate_log_run import evaluate_log_metrics

import wandb

async def main():
    dotenv.load_dotenv("/repository/.env")
    args = parse_args()
    run_id = args.run_id
    config = {
        "dataset": args.dataset,
        "model": args.model,
        "run_id": run_id,
        "tags": args.tags,
        "agent_id": run_id,
    }
    wandb_key = os.getenv("WANDB_API_KEY")
    setup_logging(config, api_key=wandb_key, dir=f"/workspace/runs/{run_id}")

    run_dir = os.path.join("/workspace/runs", run_id)
    with open(f"/repository/datasets/{config['dataset']}/metadata.json") as f:
        dataset_metadata = json.load(f)
    train_csv_path = dataset_metadata['train_split']
    test_csv_no_labels_path = dataset_metadata['test_split_no_labels']
    test_csv_path = dataset_metadata['test_split_with_labels']
    label_to_scalar = dataset_metadata['label_to_scalar']
    class_col = dataset_metadata['class_col']
    dataset_knowledge_path = dataset_metadata['dataset_knowledge']
    with open(dataset_knowledge_path) as f:
        dataset_knowledge = f.read()
    
    prompt = f"""
        Create the best possible classifier that will generalize to new unseen data.
        You are using a linux system.
        You have access to both CPU and GPU resources.

        DATASET:
        - Training file: {train_csv_path}
        - Test file: {test_csv_no_labels_path}

        Dataset knowledge:
        {dataset_knowledge}

        REQUIREMENTS:
        1. Create and save three files:
           - {run_dir}/train.py
           - {run_dir}/inference.py
           - {run_dir}/environment.yaml

        2. For {run_dir}/train.py:
        - Train a robust model suitable for the given dataset
        - Save the trained model to: {run_dir}/model.pkl using joblib or pickle

        3. For {run_dir}/inference.py:
        - Accept arguments: --input and --output
        - Load the model from: {run_dir}/model.pkl
        - Output a CSV with column 'prediction' containing a score from 0 to 1

        4. For {run_dir}/environment.yaml:
        - Create a conda environment file with all necessary packages
        - Include all libraries used in both train.py and inference.py
        """

    # TODO run this in a secure shell that can't modify other stuff
    from metagpt.roles.di.data_interpreter import DataInterpreter
    print("Running DataInterpreter...")
    role = DataInterpreter(tools=["<all>"]) #use_reflection=False by default
    await role.run(prompt)
    print("DataInterpreter finished.")

    predictions_file_path = os.path.join(run_dir, "predictions.csv")
    cmd = f"source activate {run_id}_env && python {run_dir}/inference.py --input {test_csv_no_labels_path} --output {predictions_file_path}"
    process = subprocess.run(cmd, capture_output=True, text=True, shell=True, executable="/usr/bin/bash")
    print(f"STDOUT: {process.stdout}")
    print(f"STDERR: {process.stderr}")

    metrics_file_path = os.path.join(run_dir, f"metrics.txt")
    evaluate_log_metrics(
        results_file=predictions_file_path,
        test_file=test_csv_path,
        output_file=metrics_file_path,
        label_to_scalar=label_to_scalar,
        class_col=class_col,
    )
    wandb.finish()

def parse_args():
    parser = argparse.ArgumentParser()
    parser.add_argument("--dataset", required=True, help="Dataset name")
    parser.add_argument("--model", required=True,
                        help="Model name (e.g., gpt-4o-2024-08-06, claude-3.5-sonnet-20240620)")
    parser.add_argument("--run_id", required=True, help="Run ID for the agent")
    parser.add_argument("--tags", required=True, nargs='+', help="List of tags for wandb run")

    return parser.parse_args()

if __name__ == "__main__":
    asyncio.run(main())
