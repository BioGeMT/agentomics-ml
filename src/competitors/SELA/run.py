import asyncio
import os
import dotenv
import sys # Needed for sys.exit and sys.path
import argparse
import json
import subprocess
import traceback # For printing tracebacks on error

# Add the 'src' directory of your Agentomics-ML project to Python's search path
# AGENTOMICS_SRC_PATH = "/repository/Agentomics-ML/src" # Original path from file
AGENTOMICS_SRC_PATH = "/repository/src" # Corrected path as per context
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

# Helper function to stream subprocess output
async def stream_subprocess(command_list, cwd, env, log_prefix=""):
    print(f"{log_prefix}Executing: {' '.join(command_list)} in CWD: {cwd}")
    process = await asyncio.create_subprocess_exec(
        *command_list,
        stdout=asyncio.subprocess.PIPE,
        stderr=asyncio.subprocess.PIPE,
        cwd=cwd,
        env=env
    )

    async def log_stream(stream, stream_name):
        if stream:
            async for line_bytes in stream:
                line = line_bytes.decode().rstrip()
                print(f"{log_prefix}[{stream_name}] {line}")

    await asyncio.gather(
        log_stream(process.stdout, "STDOUT"),
        log_stream(process.stderr, "STDERR")
    )

    await process.wait()
    if process.returncode != 0:
        print(f"{log_prefix}Command exited with error code {process.returncode}")
    else:
        print(f"{log_prefix}Command finished successfully.")
    return process.returncode

async def main():
    args = parse_args()
    run_id = args.run_id # This is the AGENT_ID passed from run.sh

    config = {
        "dataset": args.dataset,
        "model": args.model,
        "run_id": run_id,
        "tags": args.tags,
        "agent_id": run_id, # SELA run.sh uses AGENT_ID for run_id
    }

    current_run_output_dir = os.path.join("/workspace/runs", run_id)
    os.makedirs(current_run_output_dir, exist_ok=True)


    # Update data.yaml using the script copied to CWD (/tmp/sela)
    print(f"Updating data.yaml to use work_dir: {current_run_output_dir}...")
    try:
        # Ensure overwrite_data_yaml.sh uses the correct absolute path internally
        # Assuming overwrite_data_yaml.sh is in the CWD when run.sh executes this run.py
        # The CWD for run.py itself when called by run.sh is /tmp/sela
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
    metadata_file_path = f"/repository/datasets/{config['dataset']}/metadata.json"
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

    if not all([train_csv_path, test_csv_no_labels_path, test_csv_path, isinstance(label_to_scalar, dict), class_col]):
        print(f"Error: One or more essential data paths or metadata fields (label_to_scalar, class_col) missing/invalid in {metadata_file_path}")
        try: log_inference_stage_and_metrics(-1, message="Essential data paths/metadata missing in metadata.json.")
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
    target_col_arg = class_col if class_col else "class" # Fallback, though class_col should exist

    command_dataset = [ "python", "-u", full_dataset_script_path, "--dataset", args.dataset, "--target_col", target_col_arg ]
    try:
        return_code_dataset = await stream_subprocess(command_dataset, cwd=sela_code_base_dir, env=env_for_sela_scripts, log_prefix="[dataset.py] ")
        if return_code_dataset != 0:
            print(f"Error running SELA dataset.py, exited with code {return_code_dataset}")
            wandb.finish(exit_code=1); sys.exit(1)
    except FileNotFoundError:
        print(f"Error: SELA dataset.py not found at {full_dataset_script_path}")
        wandb.finish(exit_code=1); sys.exit(1)
    except Exception as e:
        print(f"An unexpected error occurred while running SELA dataset.py: {e}")
        traceback.print_exc()
        wandb.finish(exit_code=1); sys.exit(1)

    # --- Call SELA's run_experiment.py script ---
    experiment_script_name = "run_experiment.py"
    full_experiment_script_path = os.path.join(sela_code_base_dir, experiment_script_name) # Corrected path
    print(f"Invoking SELA's experiment script: {full_experiment_script_path}...")

    if not os.path.exists(full_experiment_script_path):
        print(f"Error: SELA run_experiment.py not found at expected location: {full_experiment_script_path}")
        wandb.finish(exit_code=1); sys.exit(1)

    command_experiment = [ "python", "-u", full_experiment_script_path, "--exp_mode", "mcts", "--task", args.dataset, "--rollouts", str(args.rollouts) ]
    try:
        return_code_experiment = await stream_subprocess(command_experiment, cwd=sela_code_base_dir, env=env_for_sela_scripts, log_prefix="[run_experiment.py] ")
        if return_code_experiment != 0:
            print(f"Error running SELA run_experiment.py, exited with code {return_code_experiment}")
            # Log stage 0 as the core SELA run failed to produce usable code/model
            log_inference_stage_and_metrics(0, message="SELA run_experiment.py failed.")
            wandb.finish(exit_code=1); sys.exit(1)
        print("SELA run_experiment.py finished. Expecting generated code in output directory.")
    except FileNotFoundError:
        print(f"Error: SELA run_experiment.py not found at {full_experiment_script_path} during execution attempt.")
        wandb.finish(exit_code=1); sys.exit(1)
    except Exception as e:
        print(f"An unexpected error occurred while running SELA run_experiment.py: {e}")
        traceback.print_exc()
        wandb.finish(exit_code=1); sys.exit(1)


    # --- Log generated SELA agent files and run inference ---
    generated_train_script_path = os.path.join(current_run_output_dir, "train.py")
    generated_inference_script_path = os.path.join(current_run_output_dir, "inference.py") # SELA might name it differently or put it elsewhere

    print(f"Checking for generated files: \n  Train script: {generated_train_script_path}\n  Inference script: {generated_inference_script_path}")
    
    # SELA might place outputs in <current_run_output_dir>/<dataset_name>/ or <current_run_output_dir>/storage/SELA/<dataset_name>/
    # Let's check a common alternative path pattern observed in some MetaGPT DI outputs
    sela_output_subdir = os.path.join(current_run_output_dir, args.dataset) 
    alternative_inference_script_path = os.path.join(sela_output_subdir, "inference.py")

    if not os.path.exists(generated_inference_script_path):
        print(f"Primary inference script path {generated_inference_script_path} not found. Checking alternative {alternative_inference_script_path}...")
        if os.path.exists(alternative_inference_script_path):
            generated_inference_script_path = alternative_inference_script_path
            print(f"Found inference script at alternative path: {generated_inference_script_path}")
            # Adjust train script path if structure is consistent
            if not os.path.exists(generated_train_script_path) and os.path.exists(os.path.join(sela_output_subdir, "train.py")):
                 generated_train_script_path = os.path.join(sela_output_subdir, "train.py")
        else:
            error_msg = f"SELA run did not generate the expected inference script in {current_run_output_dir} or {sela_output_subdir}."
            print(f"Error: {error_msg}")
            print(f"Contents of {current_run_output_dir}: {os.listdir(current_run_output_dir) if os.path.exists(current_run_output_dir) else 'N/A'}")
            if os.path.exists(sela_output_subdir): print(f"Contents of {sela_output_subdir}: {os.listdir(sela_output_subdir)}")
            log_inference_stage_and_metrics(0, message=error_msg)
            wandb.finish(exit_code=1); sys.exit(1)
            
    files_to_log = []
    if os.path.exists(generated_train_script_path): files_to_log.append(generated_train_script_path)
    else: print(f"Warning: Generated train script {generated_train_script_path} (or its alternative) not found.")
    if os.path.exists(generated_inference_script_path): files_to_log.append(generated_inference_script_path)
    # else: error already handled above

    if not files_to_log:
        error_msg = "No generated scripts found to log."
        print(f"Error: {error_msg}")
        log_inference_stage_and_metrics(0, message=error_msg)
        wandb.finish(exit_code=1); sys.exit(1)

    print("Generated SELA agent files found. Logging to WandB...")
    try:
        log_files(files=files_to_log, agent_id=run_id)
        print("Generated SELA agent files logged.")
    except Exception as e:
        print(f"Error during logging of generated SELA files: {e}")
        log_inference_stage_and_metrics(0, message=f"Error logging SELA files: {str(e)}")
        wandb.finish(exit_code=1); sys.exit(1)

    # Run the generated inference script
    predictions_output_file_path = os.path.join(current_run_output_dir, "predictions.csv") # SELA should write this here based on data.yaml
    conda_env_for_inference = "/tmp/sela_env" # SELA scripts run in this env

    # Input for inference should be the split file SELA's dataset.py was told about
    # SELA's dataset.py saves this as <datasets_dir>/<dataset_name>/split_test_wo_target.csv
    # The data.yaml's `datasets_dir` points to `/repository/datasets`
    # So, the path for inference input is /repository/datasets/<dataset_name>/split_test_wo_target.csv
    # However, the DI_INSTRUCTION in dataset.py receives `test_path = <output_dir>/split_test_wo_target.csv`
    # Let's assume dataset.py makes this file available in the SELA output_dir structure:
    inference_input_path = os.path.join(sela_output_subdir, "split_test_wo_target.csv") # if SELA created it in its own output dir
    if not os.path.exists(inference_input_path):
        # Fallback to the original location if SELA didn't copy/create it in its output.
        inference_input_path = test_csv_no_labels_path 
        print(f"Split test file not found in SELA output dir {sela_output_subdir}, using original: {inference_input_path}")


    cmd_inference_list = [
        "python", "-u", generated_inference_script_path,
        "--input", inference_input_path,
        "--output", predictions_output_file_path
    ]
    # The inference script needs to be run with its CWD set to where it might expect its model relative to.
    # Assuming SELA's generated train.py saves the model in its own CWD (which was sela_output_subdir or current_run_output_dir)
    # and inference.py loads it from a relative path.
    inference_cwd = os.path.dirname(generated_inference_script_path) 
    print(f"Executing generated inference script. CWD for inference: {inference_cwd}")
    
    conda_activation_cmd = f"source /opt/conda/etc/profile.d/conda.sh && conda activate {conda_env_for_inference} && "
    full_cmd_inference_str = conda_activation_cmd + " ".join(cmd_inference_list)

    try:
        # Using subprocess.run for simplicity here, can be converted to Popen if granular streaming is needed for inference too
        process_inference = subprocess.run(
            full_cmd_inference_str, capture_output=True, text=True, shell=True,
            executable="/bin/bash", check=True, cwd=inference_cwd
        )
        print(f"Inference script STDOUT:\n{process_inference.stdout}")
        if process_inference.stderr: print(f"Inference script STDERR:\n{process_inference.stderr}")
        print(f"Inference script finished. Predictions should be at {predictions_output_file_path}")
    except subprocess.CalledProcessError as e:
        print(f"Error running generated inference script: {e}\nSTDOUT:\n{e.stdout}\nSTDERR:\n{e.stderr}")
        log_inference_stage_and_metrics(1, message=f"Generated inference script failed: {str(e.stderr)[:250]}")
        wandb.finish(exit_code=1); sys.exit(1)

    # Evaluate metrics
    metrics_output_file_path = os.path.join(current_run_output_dir, "metrics.txt")
    # Ground truth for evaluation should be the split file SELA's dataset.py knows about
    evaluation_gt_path = os.path.join(sela_output_subdir, "split_test_target.csv") # if SELA created it
    if not os.path.exists(evaluation_gt_path):
         evaluation_gt_path = test_csv_path # Fallback to original with labels
         print(f"Split test target file not found in SELA output {sela_output_subdir}, using original: {evaluation_gt_path}")


    if not os.path.exists(predictions_output_file_path):
        print(f"Error: Predictions file {predictions_output_file_path} not found. Cannot evaluate metrics.")
        log_inference_stage_and_metrics(1, message="Predictions file missing for metrics.")
        wandb.finish(exit_code=1); sys.exit(1)
    if not os.path.exists(evaluation_gt_path): # Check again after potential fallback
        print(f"Error: Ground truth file for evaluation {evaluation_gt_path} not found.")
        log_inference_stage_and_metrics(1, message="Ground truth file missing for metrics.")
        wandb.finish(exit_code=1); sys.exit(1)

    try:
        print(f"Evaluating metrics: results={predictions_output_file_path}, ground_truth={evaluation_gt_path}")
        evaluate_log_metrics(
            results_file=predictions_output_file_path, test_file=evaluation_gt_path,
            output_file=metrics_output_file_path, label_to_scalar=label_to_scalar, class_col='target',
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
    parser.add_argument("--dataset", required=True, help="Dataset name (must match a folder in /repository/datasets/ and have a metadata.json)")
    parser.add_argument("--model", required=True, help="Model name for the LLM (as used by MetaGPT config)")
    parser.add_argument("--run_id", required=True, help="Unique Run ID for this execution (usually AGENT_ID from run.sh)")
    parser.add_argument("--tags", required=True, nargs='+', help="List of tags for Weights & Biases run")
    parser.add_argument("--rollouts", default="10", help="Number of rollouts for MCTS in SELA's run_experiment.py (default: 10)")
    return parser.parse_args()

if __name__ == "__main__":
    print(f"--- Starting run.py (with streaming) ---")
    print(f"CWD: {os.getcwd()}")
    print(f"Python executable: {sys.executable}")
    print(f"---")
    asyncio.run(main())