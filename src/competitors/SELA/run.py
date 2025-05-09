import asyncio
import os
import dotenv
import sys # Needed for sys.exit and sys.path
import argparse
import json
import subprocess
import traceback # For printing tracebacks on error

# Add the 'src' directory of your Agentomics-ML project to Python's search path
AGENTOMICS_SRC_PATH = "/repository/Agentomics-ML/src"
if AGENTOMICS_SRC_PATH not in sys.path:
    sys.path.insert(0, AGENTOMICS_SRC_PATH)

# Now imports from the src directory should work
try:
    from run_logging.wandb import setup_logging
    from run_logging.evaluate_log_run import evaluate_log_metrics
    from run_logging.log_files import log_files
    from run_logging.logging_helpers import log_inference_stage_and_metrics
except ImportError as e:
    print(f"FATAL ERROR: Could not import run_logging modules from {AGENTOMICS_SRC_PATH}: {e}")
    print("Ensure the path is correct and run_logging exists.")
    sys.exit(1)

import wandb

async def main():
    args = parse_args()
    run_id = args.run_id # This is the AGENT_ID passed from run.sh

    config = {
        "dataset": args.dataset,
        "model": args.model,
        "run_id": run_id,
        "tags": args.tags,
        "agent_id": run_id,
    }

    current_run_output_dir = os.path.join("/workspace/runs", run_id)
    os.makedirs(current_run_output_dir, exist_ok=True)


    # Update data.yaml using the script copied to CWD (/tmp/sela)
    print(f"Updating data.yaml to use work_dir: {current_run_output_dir}...")
    try:
        # Ensure overwrite_data_yaml.sh uses the correct absolute path internally
        subprocess.run(
            ["bash", "./overwrite_data_yaml.sh", current_run_output_dir],
            check=True, cwd=os.getcwd() # Should be /tmp/sela
        )
        print("data.yaml updated successfully by overwrite_data_yaml.sh.")
    except subprocess.CalledProcessError as e:
        print(f"Error running overwrite_data_yaml.sh: {e}\nstdout: {e.stdout}\nstderr: {e.stderr}")
        sys.exit(f"FATAL: Failed to configure data.yaml for run {run_id}.")
    except FileNotFoundError:
        print(f"Error: overwrite_data_yaml.sh not found in {os.getcwd()}. Make sure set_up.sh copied it.")
        sys.exit(f"FATAL: Missing overwrite_data_yaml.sh in {os.getcwd()}.")


    # Setup Weights & Biases logging
    wandb_key = os.getenv("WANDB_API_KEY")
    if not wandb_key: print("Warning: WANDB_API_KEY not found in environment.")
    try:
      setup_logging(config, api_key=wandb_key, dir=current_run_output_dir)
      print(f"W&B logging setup for run_id: {run_id}.")
    except Exception as e:
      print(f"Error setting up WandB logging: {e}"); traceback.print_exc()
      sys.exit("FATAL: WandB setup failed.")


    # Check for dataset metadata
    metadata_file_path = f"/repository/Agentomics-ML/datasets/{config['dataset']}/metadata.json" # Corrected path
    print(f"Looking for dataset metadata at: {metadata_file_path}")
    try:
        with open(metadata_file_path) as f: dataset_metadata = json.load(f)
        print("Dataset metadata loaded successfully.")
    except FileNotFoundError:
        print(f"Error: Dataset metadata file not found at {metadata_file_path}")
        try: log_inference_stage_and_metrics(-1, message="Dataset metadata not found.")
        except Exception: pass
        wandb.finish(exit_code=1); sys.exit(1)
    except json.JSONDecodeError:
        print(f"Error: Could not decode JSON from {metadata_file_path}")
        try: log_inference_stage_and_metrics(-1, message="Dataset metadata JSON decode error.")
        except Exception: pass
        wandb.finish(exit_code=1); sys.exit(1)

    # Extract needed info from metadata
    train_csv_path = dataset_metadata.get('train_split') # Path to original train data
    test_csv_no_labels_path = dataset_metadata.get('test_split_no_labels') # Path to original test data (no labels)
    test_csv_path = dataset_metadata.get('test_split_with_labels') # Path to original test data (with labels) for eval
    label_to_scalar = dataset_metadata.get('label_to_scalar')
    class_col = dataset_metadata.get('class_col')

    if not all([train_csv_path, test_csv_no_labels_path, test_csv_path]):
        print("Error: One or more essential data paths missing in metadata.json")
        try: log_inference_stage_and_metrics(-1, message="Essential data paths missing in metadata.")
        except Exception: pass
        wandb.finish(exit_code=1); sys.exit(1)

    # Define SELA code paths and environment for subprocesses
    metagpt_fork_root_dir = "/tmp/MetaGPT_fork_sela"
    sela_code_base_dir = os.path.join(metagpt_fork_root_dir, "metagpt", "ext", "sela")
    sela_data_subdir = "data" # Subdir where dataset.py is

    metagpt_config_path_for_run = os.path.join(current_run_output_dir, ".metagpt", "config2.yaml")
    print(f"Setting METAGPT_CONFIG_PATH for SELA subprocesses: {metagpt_config_path_for_run}")
    python_path_for_sela_scripts = f"{metagpt_fork_root_dir}:{os.environ.get('PYTHONPATH', '')}"
    env_for_sela_scripts = { **os.environ, "PYTHONPATH": python_path_for_sela_scripts, "METAGPT_CONFIG_PATH": metagpt_config_path_for_run }
    print(f"PYTHONPATH for SELA subprocesses: {python_path_for_sela_scripts}")


    # --- Call SELA's dataset.py script ---
    print(f"Invoking SELA's dataset.py for dataset: {args.dataset}...")
    dataset_script_name = "dataset.py"
    full_dataset_script_path = os.path.join(sela_code_base_dir, sela_data_subdir, dataset_script_name)
    target_col_arg = class_col if class_col else "class"

    command_dataset = [ "python", full_dataset_script_path, "--dataset", args.dataset, "--target_col", target_col_arg ]
    try:
        print(f"Executing: {' '.join(command_dataset)} in CWD: {sela_code_base_dir}") # CWD for dataset.py is sela base dir
        result_dataset = subprocess.run(
            command_dataset, cwd=sela_code_base_dir, check=True,
            capture_output=True, text=True, env=env_for_sela_scripts
        )
        print("SELA dataset.py STDOUT:\n", result_dataset.stdout)
        if result_dataset.stderr: print("SELA dataset.py STDERR:\n", result_dataset.stderr)
        print("SELA dataset.py finished successfully.")
    except subprocess.CalledProcessError as e:
        print(f"Error running SELA dataset.py: {e}\nSTDOUT:\n{e.stdout}\nSTDERR:\n{e.stderr}")
        wandb.finish(exit_code=1); sys.exit(1)
    except FileNotFoundError:
        print(f"Error: SELA dataset.py not found at {full_dataset_script_path}")
        wandb.finish(exit_code=1); sys.exit(1)

    # --- Call SELA's run_experiment.py script ---
    experiment_script_name = "run_experiment.py"
    # ---> CORRECTED PATH: run_experiment.py is directly in sela_code_base_dir <---
    full_experiment_script_path = os.path.join(sela_code_base_dir, experiment_script_name)
    print(f"Invoking SELA's experiment script: {full_experiment_script_path}...")

    if not os.path.exists(full_experiment_script_path):
        print(f"Error: SELA run_experiment.py not found at expected location: {full_experiment_script_path}")
        wandb.finish(exit_code=1); sys.exit(1)

    command_experiment = [ "python", full_experiment_script_path, "--exp_mode", "mcts", "--task", args.dataset, "--rollouts", str(args.rollouts) ]
    try:
        # run_experiment.py likely also expects to be run from the sela base dir
        print(f"Executing: {' '.join(command_experiment)} in CWD: {sela_code_base_dir}")
        result_experiment = subprocess.run(
            command_experiment, cwd=sela_code_base_dir, check=True,
            capture_output=True, text=True, env=env_for_sela_scripts
        )
        print("SELA run_experiment.py STDOUT:\n", result_experiment.stdout)
        if result_experiment.stderr: print("SELA run_experiment.py STDERR:\n", result_experiment.stderr)
        print("SELA run_experiment.py finished. Expecting generated code in output directory.")
    except subprocess.CalledProcessError as e:
        print(f"Error running SELA run_experiment.py: {e}\nSTDOUT:\n{e.stdout}\nSTDERR:\n{e.stderr}")
        wandb.finish(exit_code=1); sys.exit(1)
    except FileNotFoundError: # Should be caught by the check above, but just in case
        print(f"Error: SELA run_experiment.py not found at {full_experiment_script_path} during execution attempt.")
        wandb.finish(exit_code=1); sys.exit(1)

    # --- Log generated SELA agent files and run inference ---
    # SELA (via run_experiment.py) should have saved its output in current_run_output_dir.
    # Check for expected output files (names might vary based on SELA's implementation)
    # Assuming it generates 'train.py' and 'inference.py' directly in current_run_output_dir
    generated_train_script_path = os.path.join(current_run_output_dir, "train.py")
    generated_inference_script_path = os.path.join(current_run_output_dir, "inference.py")

    print(f"Checking for generated files: \n  Train script: {generated_train_script_path}\n  Inference script: {generated_inference_script_path}")
    if not os.path.exists(generated_train_script_path) or not os.path.exists(generated_inference_script_path):
        error_msg = f"SELA run did not generate the expected script(s) in {current_run_output_dir}."
        print(f"Error: {error_msg}")
        # Check contents of output dir for debugging
        print(f"Contents of {current_run_output_dir}:")
        try: print(os.listdir(current_run_output_dir))
        except FileNotFoundError: print("  Directory not found.")
        log_inference_stage_and_metrics(0, message=error_msg)
        wandb.finish(exit_code=1); sys.exit(1)

    print("Generated SELA agent files found. Logging to WandB...")
    try:
        log_files(files=[generated_train_script_path, generated_inference_script_path], agent_id=run_id)
        print("Generated SELA agent files logged.")
    except Exception as e:
        print(f"Error during logging of generated SELA files: {e}")
        log_inference_stage_and_metrics(0, message=f"Error logging SELA files: {str(e)}")
        wandb.finish(exit_code=1); sys.exit(1)

    # Run the generated inference script
    predictions_output_file_path = os.path.join(current_run_output_dir, "predictions.csv")
    conda_env_for_inference = "/tmp/sela_env"

    # Use the split file saved in work_dir as input for inference
    inference_input_path = os.path.join(current_run_output_dir, "split_test_wo_target.csv")
    if not os.path.exists(inference_input_path):
         # Fallback to original test path if split wasn't created? Or error out?
         print(f"Warning: Split test file {inference_input_path} not found. Falling back to original test path {test_csv_no_labels_path}")
         inference_input_path = test_csv_no_labels_path # Use original path if split missing

    if not os.path.exists(generated_inference_script_path):
         print(f"Error: Generated inference script {generated_inference_script_path} not found. Cannot run inference.")
         log_inference_stage_and_metrics(1, message="Generated inference script missing.")
         wandb.finish(exit_code=1); sys.exit(1)

    cmd_inference = ( f"source /opt/conda/etc/profile.d/conda.sh && conda activate {conda_env_for_inference} && python {generated_inference_script_path} --input {inference_input_path} --output {predictions_output_file_path}" )
    print(f"Executing generated inference script. CWD for inference: {current_run_output_dir}")
    print(f"Command: {cmd_inference}")
    try:
        process_inference = subprocess.run(
            cmd_inference, capture_output=True, text=True, shell=True,
            executable="/bin/bash", check=True, cwd=current_run_output_dir
        )
        print(f"Inference script STDOUT: {process_inference.stdout}")
        if process_inference.stderr: print(f"Inference script STDERR: {process_inference.stderr}")
        print(f"Inference script finished. Predictions should be at {predictions_output_file_path}")
    except subprocess.CalledProcessError as e:
        print(f"Error running generated inference script: {e}\nSTDOUT:\n{e.stdout}\nSTDERR:\n{e.stderr}")
        log_inference_stage_and_metrics(1, message=f"Generated inference script failed: {str(e.stderr)[:250]}")
        wandb.finish(exit_code=1); sys.exit(1)

    # Evaluate metrics
    metrics_output_file_path = os.path.join(current_run_output_dir, "metrics.txt")
    # Use the ground truth split saved in work_dir for evaluation
    evaluation_gt_path = os.path.join(current_run_output_dir, "split_test_target.csv")
    if not os.path.exists(evaluation_gt_path):
        print(f"Warning: Ground truth split file {evaluation_gt_path} not found. Falling back to original test path {test_csv_path}")
        evaluation_gt_path = test_csv_path # Use original path if split missing

    if not all([evaluation_gt_path, label_to_scalar, class_col]):
        missing_items_msg = f"Missing items for metrics: gt_path={evaluation_gt_path is not None}, label_to_scalar={label_to_scalar is not None}, class_col={class_col is not None}"
        print(f"Error: {missing_items_msg}")
        log_inference_stage_and_metrics(1, message=missing_items_msg)
        wandb.finish(exit_code=1); sys.exit(1)
    if not os.path.exists(predictions_output_file_path):
        print(f"Error: Predictions file {predictions_output_file_path} not found. Cannot evaluate metrics.")
        log_inference_stage_and_metrics(1, message="Predictions file missing for metrics.")
        wandb.finish(exit_code=1); sys.exit(1)

    try:
        print(f"Evaluating metrics: results={predictions_output_file_path}, ground_truth={evaluation_gt_path}")
        # Ensure evaluate_log_metrics uses 'target' column from the split_test_target.csv
        evaluate_log_metrics(
            results_file=predictions_output_file_path, test_file=evaluation_gt_path,
            output_file=metrics_output_file_path, label_to_scalar=label_to_scalar, class_col='target', # Use 'target' as column name
        )
        print(f"Metrics evaluated and logged. Summary in {metrics_output_file_path}")
    except Exception as e:
        print(f"Error during metrics evaluation: {e}")
        traceback.print_exc()
        log_inference_stage_and_metrics(1, message=f"Metrics evaluation failed: {str(e)}")
        wandb.finish(exit_code=1); sys.exit(1)

    print(f"Run (ID: {run_id}) for dataset {args.dataset} with model {args.model} completed successfully.")
    wandb.finish() # Successful completion


def parse_args():
    parser = argparse.ArgumentParser(description="Run SELA agent pipeline for a given dataset and model.")
    parser.add_argument("--dataset", required=True, help="Dataset name (must match a folder in /repository/Agentomics-ML/datasets/ and have a metadata.json)")
    parser.add_argument("--model", required=True, help="Model name for the LLM (as used by MetaGPT config)")
    parser.add_argument("--run_id", required=True, help="Unique Run ID for this execution (usually AGENT_ID from run.sh)")
    parser.add_argument("--tags", required=True, nargs='+', help="List of tags for Weights & Biases run")
    parser.add_argument("--rollouts", default="10", help="Number of rollouts for MCTS in SELA's run_experiment.py (default: 10)")
    return parser.parse_args()

if __name__ == "__main__":
    print(f"--- Starting run.py ---")
    print(f"CWD: {os.getcwd()}")
    print(f"Python executable: {sys.executable}")
    print(f"---")
    asyncio.run(main())

