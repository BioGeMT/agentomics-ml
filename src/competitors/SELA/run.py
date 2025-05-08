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

import wandb

import MetaGPT.metagpt.ext.sela.data.dataset as sela_dataset
from MetaGPT.metagpt.ext.sela.insights.solution_designer import SolutionDesigner
from MetaGPT.metagpt.ext.sela.utils import DATA_CONFIG


async def main():
    dotenv.load_dotenv("/repository/.env")
    args = parse_args()
    run_id = args.run_id

    # Hack to make metagpt have configurable config path
    # import pathlib
    # from pathlib import Path
    # pathlib.Path.home = lambda: Path(f"/workspace/runs/{run_id}")
    # from metagpt.roles.di.data_interpreter import DataInterpreter

    config = {
        "dataset": args.dataset,
        "model": args.model,
        "run_id": run_id,
        "tags": args.tags,
        "agent_id": run_id,
    }

    # - DONE - TODO write the following command in python or make a bash script for it and call it, work_dir: os.path.join("/workspace/runs", run_name)
    # cat > MetaGPT/metagpt/exp/sela/data.yaml << 'EOF'
    # datasets_dir: "/repository/datasets" # path to the datasets directory
    # work_dir: /workspace/runs/$run_id # path to the workspace directory
    # role_dir: storage/SELA # relative path to the role directory
    # EOF
    subprocess.run(["bash", "overwrite_data_yaml.sh", os.path.join("/workspace/runs", run_id)])
    
    # EDIT: SKIP THIS FOR NOW, THE CODE HAS BEEN UPDATED IN THE FORK
    # to accept the dataset name as an argument
    # TODO finish the TODO in overwrite_dataset.sh script
    # call the overwrite_dataset.sh here with the args.dataset argument

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
    
    # TODO
    # python MetaGPT/metagpt/exp/sela/data/dataset.py --save_analysis_pool --dataset $args.dataset --target_col "class"
    # command = [
    #     "python", 
    #     "MetaGPT/metagpt/exp/sela/data/dataset.py", 
    #     "--save_analysis_pool", 
    #     "--dataset", 
    #     args.dataset, 
    #     "--target_col", 
    #     "class"
    # ]
    # result = subprocess.run(command, check=True, capture_output=True, text=True)
    # print(result.stdout)
    datasets_dir = DATA_CONFIG["datasets_dir"]
    target_col = "class"
    force_update = True
    save_analysis_pool = True
    datasets_dict = {"datasets": {}}
    solution_designer = SolutionDesigner()
   
    custom_dataset = sela_dataset.ExpDataset(args.dataset, datasets_dir, target_col=target_col, force_update=force_update)
    asyncio.run(sela_dataset.process_dataset(custom_dataset, solution_designer, save_analysis_pool, datasets_dict))

    sela_dataset.save_datasets_dict_to_yaml(datasets_dict)

    # TODO
    # python MetaGPT/metagpt/exp/sela/data/python run_experiment.py --exp_mode mcts --task $args.dataset --rollouts $args.rollouts
    # command = [
    #     "python", 
    #     "MetaGPT/metagpt/exp/sela/data/run_experiment.py", 
    #     "--exp_mode", 
    #     "mcts", 
    #     "--task", 
    #     args.dataset, 
    #     "--rollouts", 
    #     str(args.rollouts) 
    # ]
    # result = subprocess.run(command, check=True, capture_output=True, text=True)
    # print(result.stdout)

    try:
        log_files(files=[train_file_path, inference_path], agent_id=run_id)
    except Exception as e:
        print(f"Error logging files: {e}")
        log_inference_stage_and_metrics(0)
        return

    predictions_file_path = os.path.join(run_dir, "predictions.csv")
    cmd = f"source /opt/conda/etc/profile.d/conda.sh && conda activate /tmp/di_env && python {run_dir}/inference.py --input {test_csv_no_labels_path} --output {predictions_file_path}"
    try:
        process = subprocess.run(cmd, capture_output=True, text=True, shell=True, executable="/usr/bin/bash")
        print(f"STDOUT: {process.stdout}")
        print(f"STDERR: {process.stderr}")
    except Exception as e:
        print(f"Error running inference: {e}")
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

    return parser.parse_args()

if __name__ == "__main__":
    asyncio.run(main())
