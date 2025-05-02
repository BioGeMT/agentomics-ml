import json
import subprocess
from eval.evaluate_result import evaluate_log_metrics
import os
from run_logging.logging_helpers import log_inference_stage_and_metrics

def evaluate_log_run(config):
    with open(f"/repository/datasets/{config['dataset']}/metadata.json") as f:
        dataset_metadata = json.load(f)

    inference_path = f"/workspace/runs/{config['agent_id']}/inference.py"
    if not os.path.exists(inference_path):
        # Level 0: inference.py doesn't exist
        log_inference_stage_and_metrics(0)
        return

    try:
        agent_env_name = f"/workspace/runs/{config['agent_id']}/.conda/envs/{config['agent_id']}_env"
        result = subprocess.run(f"source activate {agent_env_name} && python /workspace/runs/{config['agent_id']}/inference.py --input " + 
                    f"{dataset_metadata['test_split_no_labels']} --output /workspace/runs/{config['agent_id']}/eval_predictions.csv", 
                    shell=True, executable="/bin/bash", capture_output=True, text=True)
        
        if result.returncode != 0:
            print(f"Inference script error: {result.stderr}")

            # Level 1: inference.py exists but execution failed
            log_inference_stage_and_metrics(1)

            return
        
    except Exception as e:
        print(e)
        # Level 1: inference.py exists but execution failed
        log_inference_stage_and_metrics(1)
        
        return
    
    try:
        return evaluate_log_metrics(
            results_file=f"/workspace/runs/{config['agent_id']}/eval_predictions.csv",
            test_file=f"{dataset_metadata['test_split_with_labels']}",
            output_file=f"/workspace/runs/{config['agent_id']}/metrics.txt"
        )

    except Exception as e:
        print(e)
        # Level 1: inference.py exists and ran, but metrics calculation failed
        log_inference_stage_and_metrics(1)


def dry_run_evaluate_log_run(config):
    with open(f"/repository/datasets/{config['dataset']}/metadata.json") as f:
        dataset_metadata = json.load(f)

    agent_env_name = f"/workspace/runs/{config['agent_id']}/.conda/envs/{config['agent_id']}_env"
    return subprocess.run(f"source activate {agent_env_name} && python /workspace/runs/{config['agent_id']}/inference.py --input " + 
                f"{dataset_metadata['train_split']} --output /dev/null", 
                shell=True, executable="/bin/bash", capture_output=True)
