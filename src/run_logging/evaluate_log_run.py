import json
import subprocess
from eval.evaluate_result import evaluate_log_metrics
import os
from run_logging.logging_helpers import log_inference_stage_and_metrics

def evaluate_log_run(config):
    # Load dataset metadata
    meta_path = f"/repository/datasets/{config['dataset']}/metadata.json"
    with open(meta_path) as f:
        dataset_metadata = json.load(f)

    # Paths
    run_dir = f"/workspace/runs/{config['agent_id']}"
    inference_script = os.path.join(run_dir, "inference.py")
    env_name = f"{config['agent_id']}_env"
    yaml_file = os.path.join(run_dir, f"{config['agent_id']}_env.yaml")
    test_input = dataset_metadata['test_split_no_labels']
    predictions_file = os.path.join(run_dir, "eval_predictions.csv")

    # Stage 0: check inference.py exists
    if not os.path.isfile(inference_script):
        log_inference_stage_and_metrics(0)
        return

    # Stage 1: create & activate conda env and run inference
    try:
        # Construct a bash login shell command that sources conda, creates env, activates it, and runs inference
        cmd = (
            "bash -lc '"
            "source $(conda info --base)/etc/profile.d/conda.sh && "
            f"conda env create -n {env_name} -f {yaml_file} --force && "
            f"conda activate {env_name} && "
            f"python {inference_script} --input {test_input} --output {predictions_file} && "
            "exit 0'"
        )
        result = subprocess.run(cmd, shell=True, executable="/bin/bash", capture_output=True, text=True)
        print("Command:", cmd)
        print("stdout:", result.stdout)
        print("stderr:", result.stderr)
        if result.returncode != 0:
            print(f"Inference step failed with code {result.returncode}")
            log_inference_stage_and_metrics(1)
            return
    except Exception as e:
        print("Error running inference:", e)
        log_inference_stage_and_metrics(1)
        return

    # Stage 2: evaluate metrics
    try:
        metrics = evaluate_log_metrics(
            results_file=predictions_file,
            test_file=dataset_metadata['test_split_with_labels'],
            output_file=os.path.join(run_dir, "metrics.txt")
        )
        return metrics
    except Exception as e:
        print("Error computing metrics:", e)
        log_inference_stage_and_metrics(1)


def dry_run_evaluate_log_run(config):
    # Load dataset metadata
    meta_path = f"/repository/datasets/{config['dataset']}/metadata.json"
    with open(meta_path) as f:
        dataset_metadata = json.load(f)

    run_dir = f"/workspace/runs/{config['agent_id']}"
    inference_script = os.path.join(run_dir, "inference.py")
    env_name = f"{config['agent_id']}_env"
    yaml_file = os.path.join(run_dir, f"{config['agent_id']}_env.yaml")
    train_input = dataset_metadata['train_split']

    try:
        # Similar login shell for dry-run
        cmd = (
            "bash -lc '"
            "source $(conda info --base)/etc/profile.d/conda.sh && "
            f"conda env create -n {env_name} -f {yaml_file} -y && "
            f"conda activate {env_name} && "
            f"python {inference_script} --input {train_input} --output /dev/null && "
            "exit 0'"
        )
        result = subprocess.run(cmd, shell=True, executable="/bin/bash", capture_output=True, text=True)
        print("Dry-run cmd:", cmd)
        print("stdout:", result.stdout)
        print("stderr:", result.stderr)
        return result
    except Exception as e:
        print("Error in dry-run:", e)
        log_inference_stage_and_metrics(0)
        # Return a non-zero CompletedProcess
        return subprocess.CompletedProcess(args=[cmd], returncode=1, stdout="", stderr=str(e))

