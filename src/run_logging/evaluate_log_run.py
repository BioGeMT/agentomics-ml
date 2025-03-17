import wandb
import json
import subprocess
from eval.evaluate_result import evaluate_log_metrics
import os

def evaluate_log_run(config):
    with open(f"/repository/datasets/{config['dataset']}/metadata.json") as f:
        dataset_metadata = json.load(f)

    inference_path = f"/workspace/runs/{config['agent_id']}/inference.py"
    if not os.path.exists(inference_path):
        # Level 0: inference.py doesn't exist
        wandb.log({"Level": 0})
        error_metrics = {
            'AUPRC': -1,
            'AUROC': -1,
        }
        wandb.log(error_metrics)
        return

    try:
        agent_env_name = f"/workspace/runs/{config['agent_id']}/.conda/envs/{config['agent_id']}_env"
        result = subprocess.run(f"source activate {agent_env_name} && python /workspace/runs/{config['agent_id']}/inference.py --input " + 
                    f"{dataset_metadata['test_split_no_labels']} --output /workspace/runs/{config['agent_id']}/eval_predictions.csv", 
                    shell=True, executable="/bin/bash", capture_output=True, text=True)
        
        if result.returncode != 0:
            print(f"Inference script error: {result.stderr}")

            # Level 1: inference.py exists but execution failed
            wandb.log({"Level": 1})
            
            error_metrics = {
                'AUPRC': -1,
                'AUROC': -1,
            }

            wandb.log(error_metrics)

            return
        
    except Exception as e:
        print(e)
        # Level 1: inference.py exists but execution failed
        wandb.log({"Level": 1})
        
        error_metrics = {
            'AUPRC': -1,
            'AUROC': -1,
        }
        wandb.log(error_metrics)
        return
    
    try:
        evaluate_log_metrics(
            results_file=f"/workspace/runs/{config['agent_id']}/eval_predictions.csv",
            test_file=f"{dataset_metadata['test_split_with_labels']}",
            output_file=f"/workspace/runs/{config['agent_id']}/metrics.txt",
            logging_fn=wandb.log,
        )
    except Exception as e:
        print(e)
        # Level 1: inference.py exists and ran, but metrics calculation failed
        wandb.log({"Level": 1})

        error_metrics = {
            'AUPRC': -1,
            'AUROC': -1,
        }
        wandb.log(error_metrics)


def dry_run_evaluate_log_run(config):
    with open(f"/repository/datasets/{config['dataset']}/metadata.json") as f:
        dataset_metadata = json.load(f)

    agent_env_name = f"/workspace/runs/{config['agent_id']}/.conda/envs/{config['agent_id']}_env"
    return subprocess.run(f"source activate {agent_env_name} && python /workspace/runs/{config['agent_id']}/inference.py --input " + 
                f"{dataset_metadata['train_split']} --output /dev/null", 
                shell=True, executable="/bin/bash", capture_output=True)
