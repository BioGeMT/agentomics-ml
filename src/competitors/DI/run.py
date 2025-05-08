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
from run_logging.log_files import log_files
from run_logging.logging_helpers import log_inference_stage_and_metrics
from competitors.DI.set_config import set_config
from utils.api_keys import create_new_api_key, get_api_key_usage, delete_api_key

import wandb

async def main():
    dotenv.load_dotenv("/repository/.env")
    args = parse_args()
    run_id = args.run_id
    run_dir = os.path.join("/workspace/runs", run_id)

    api_key_data = create_new_api_key(name=run_id, limit=args.credit_budget)

    openrouter_api_key = api_key_data['key']
    openrouter_api_key_hash = api_key_data['hash']

    set_config(config_path=f"{run_dir}/.metagpt/config2.yaml", api_type='openrouter', model=args.model, base_url='https://openrouter.ai/api/v1', api_key=openrouter_api_key)

    # Hack to make metagpt have configurable config path
    import pathlib
    from pathlib import Path
    pathlib.Path.home = lambda: Path(f"/workspace/runs/{run_id}")
    from metagpt.roles.di.data_interpreter import DataInterpreter

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
    
    train_file_path = f"{run_dir}/train.py"
    inference_path = f"{run_dir}/inference.py"
    # env_yaml_path = f"{run_dir}/environment.yaml" #Not necessary since agent installs directly into the environment
    # model_path = f"{run_dir}/model.pkl" #Not necessary since the agent has freedom to choose the format
    prompt = f"""
        Create the best possible classifier that will generalize to new unseen data.
        You are using a linux system.
        You have access to both CPU and GPU resources.
        You run in a conda environment and you are allowed to install any dependencies you need.

        DATASET:
        - Training file: {train_csv_path}
        - Test file: {test_csv_no_labels_path}

        Dataset knowledge:
        {dataset_knowledge}

        REQUIREMENTS:
        1. Create and save three files:
           - {train_file_path}
           - {inference_path}
           - a model file

        2. For {train_file_path}:
        - Train a robust model suitable for the given dataset
        - Save the trained model into {run_dir} to be loaded later by the inference script

        3. For {inference_path}:
        - Accept arguments: --input and --output
        - Load the saved model
        - Output a CSV with column 'prediction' containing a score from 0 to 1

        You must write all three files to the correct paths before you finish.
        """

    try:
        print("Running DataInterpreter...")
        role = DataInterpreter(tools=["<all>"]) #use_reflection=False by default
        await role.run(prompt)
        print("DataInterpreter finished.")
    except Exception as e:
        print(f"Error running DataInterpreter: {e}")
        log_inference_stage_and_metrics(0)
        stats = get_api_key_usage(openrouter_api_key_hash)
        if stats['usage'] >= stats['limit']:
            wandb.log({"out_of_credits": True})
        return
    finally:
        stats = get_api_key_usage(openrouter_api_key_hash)
        wandb.log(stats)
        delete_api_key(openrouter_api_key_hash)

    try:
        log_files(files=[train_file_path, inference_path], agent_id=run_id)
    except Exception as e:
        print(f"Error logging files: {e}")
        log_inference_stage_and_metrics(0)
        return

    predictions_file_path = os.path.join(run_dir, "predictions.csv")
    agent_env_path = f"{run_dir}/{run_id}_env" 
    cmd = f"source /opt/conda/etc/profile.d/conda.sh && conda activate {agent_env_path} && python {run_dir}/inference.py --input {test_csv_no_labels_path} --output {predictions_file_path}"
    
    process = subprocess.run(cmd, capture_output=True, text=True, shell=True, executable="/usr/bin/bash")
    print(f"STDOUT: {process.stdout}")
    print(f"STDERR: {process.stderr}")
    if process.returncode != 0:
        print(f"Error running inference: {process.stderr}")
        log_inference_stage_and_metrics(1)
        return
        
    metrics_file_path = os.path.join(run_dir, f"metrics.txt")
    try:
        evaluate_log_metrics(
            results_file=predictions_file_path,
            test_file=test_csv_path,
            output_file=metrics_file_path,
            label_to_scalar=label_to_scalar,
            class_col=class_col,
        )
    except Exception as e:
        print(f"Error evaluating metrics: {e}")
        log_inference_stage_and_metrics(1)
        return
    
    wandb.finish()

def parse_args():
    parser = argparse.ArgumentParser()
    parser.add_argument("--dataset", required=True, help="Dataset name")
    parser.add_argument("--model", required=True,
                        help="Model name (e.g., gpt-4o-2024-08-06, claude-3.5-sonnet-20240620)")
    parser.add_argument("--run_id", required=True, help="Run ID for the agent")
    parser.add_argument("--tags", required=True, nargs='+', help="List of tags for wandb run")
    parser.add_argument("--credit-budget", type=float, default=0.0, help="Credit budget for the API key")

    return parser.parse_args()

if __name__ == "__main__":
    asyncio.run(main())
