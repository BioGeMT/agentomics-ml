#!/usr/bin/env python3
"""
Enhanced one-shot LLM test script using Claude Code CLI.
This script:
1. Calls Claude Code to generate train.py and inference.py scripts in one shot
2. Extracts and saves the scripts with unique run numbers
3. Executes the train.py script to train the model
4. Executes the inference.py script to generate predictions
5. Executes the evaluation script to calculate metrics
6. Preserves all scripts for variability testing
"""

import os
import argparse
import re
import subprocess
import time
import datetime
import json
import shutil


def check_claude_code_installed():
    """
    Check if Claude Code CLI is installed in the system.

    Returns:
        bool: True if Claude Code is installed, False otherwise
    """
    try:
        result = subprocess.run(["claude", "--version"],
                                capture_output=True,
                                text=True)
        return result.returncode == 0
    except FileNotFoundError:
        return False


def extract_scripts(response_text):
    """
    Extract train.py and inference.py scripts from the Claude response.

    Args:
        response_text (str): The Claude's response text

    Returns:
        tuple: A tuple containing (train_script, inference_script)
    """
    # Look for scripts between Python code blocks
    python_blocks = re.findall(r'```python\s+(.*?)```', response_text, re.DOTALL)

    # If no Python blocks found, try looking for scripts with "# train.py" and "# inference.py" headers
    if not python_blocks or len(python_blocks) < 2:
        train_script = None
        inference_script = None

        # Try to find train.py by header
        train_match = re.search(r'# train\.py\s+(.*?)(?=# inference\.py|\Z)', response_text, re.DOTALL)
        if train_match:
            train_script = train_match.group(1).strip()

        # Try to find inference.py by header
        inference_match = re.search(r'# inference\.py\s+(.*?)(?=\Z|\n#)', response_text, re.DOTALL)
        if inference_match:
            inference_script = inference_match.group(1).strip()

        # If still not found, look for content between "```python" and "```" markers
        if not train_script or not inference_script:
            if len(python_blocks) >= 1:
                train_script = python_blocks[0]
            if len(python_blocks) >= 2:
                inference_script = python_blocks[1]
    else:
        # Assume first block is train.py and second is inference.py
        train_script = python_blocks[0]
        inference_script = python_blocks[1]

    # If still not found, try another approach with file path indicators
    if not train_script:
        train_indicators = ["def train", "def main(", "if __name__ == '__main__'"]
        for block in python_blocks:
            if any(indicator in block for indicator in train_indicators):
                train_script = block
                break

    if not inference_script:
        inference_indicators = ["argparse", "--input", "--output", "def inference", "load_model"]
        for block in python_blocks:
            if any(indicator in block for indicator in inference_indicators) and block != train_script:
                inference_script = block
                break

    return train_script, inference_script


def get_run_number(output_dir):
    """
    Get the next available run number for script files.

    Args:
        output_dir (str): The directory to check for existing run numbers

    Returns:
        int: The next available run number
    """
    # Get all existing train script files
    existing_files = []
    if os.path.exists(output_dir):
        existing_files = [f for f in os.listdir(output_dir) if f.startswith("train_") and f.endswith(".py")]

    # Extract run numbers from existing files
    run_numbers = []
    for file in existing_files:
        match = re.search(r'train_(\d+)\.py', file)
        if match:
            run_numbers.append(int(match.group(1)))

    # Return the next available run number
    return max(run_numbers, default=0) + 1


def save_scripts(train_script, inference_script, output_dir):
    """
    Save the extracted scripts to the specified output directory with unique run numbers.

    Args:
        train_script (str): The content of train.py
        inference_script (str): The content of inference.py
        output_dir (str): The directory to save the scripts

    Returns:
        tuple: Paths to the saved train and inference scripts
    """
    # Create output directory if it doesn't exist
    os.makedirs(output_dir, exist_ok=True)

    # Get a unique run number
    run_number = get_run_number(output_dir)
    timestamp = datetime.datetime.now().strftime("%Y%m%d_%H%M%S")

    # Define file paths with unique numbers
    train_path = os.path.join(output_dir, f"train_{run_number}.py")
    inference_path = os.path.join(output_dir, f"inference_{run_number}.py")

    # Also save current versions without run numbers for convenience
    current_train_path = os.path.join(output_dir, "train.py")
    current_inference_path = os.path.join(output_dir, "inference.py")

    # Save train.py files
    if train_script:
        with open(train_path, "w") as f:
            f.write(train_script)
        with open(current_train_path, "w") as f:
            f.write(train_script)
        print(f"Saved train script to {train_path}")
    else:
        print("Warning: Could not extract train.py script from Claude response")
        return None, None

    # Save inference.py files
    if inference_script:
        with open(inference_path, "w") as f:
            f.write(inference_script)
        with open(current_inference_path, "w") as f:
            f.write(inference_script)
        print(f"Saved inference script to {inference_path}")
    else:
        print("Warning: Could not extract inference.py script from Claude response")
        return train_path, None

    # Create a log file for this run
    log_path = os.path.join(output_dir, f"run_{run_number}_log.txt")
    with open(log_path, "w") as f:
        f.write(f"Run {run_number} generated on {timestamp}\n")
        f.write(f"Train script: {train_path}\n")
        f.write(f"Inference script: {inference_path}\n")

    return train_path, inference_path


def run_script(script_path, script_type, output_dir, dataset_name=None):
    """
    Run a Python script and wait for it to complete.

    Args:
        script_path (str): Path to the script to run
        script_type (str): Either 'train' or 'inference'
        output_dir (str): Directory for output files
        dataset_name (str, optional): Dataset name for inference paths

    Returns:
        int: Return code from the script execution
    """
    if not os.path.exists(script_path):
        print(f"Error: {script_path} does not exist")
        return 1

    print(f"\n{'=' * 40}")
    print(f"Running {script_type} script: {script_path}")
    print(f"{'=' * 40}\n")

    # Create command
    cmd = ["python", script_path]

    # Add arguments for inference script
    if script_type == 'inference' and dataset_name:
        test_path = f"/home/eddy/Documents/Agentomics-ML/datasets/{dataset_name}/human_nontata_promoters_test.no_label.csv"
        output_path = os.path.join(output_dir, "predictions.csv")
        cmd.extend(["--input", test_path, "--output", output_path])
        print(f"Inference input: {test_path}")
        print(f"Inference output: {output_path}")

    # Run the script
    start_time = time.time()
    process = subprocess.run(cmd, capture_output=True, text=True)
    end_time = time.time()

    # Print results
    print(f"\nExecution completed in {end_time - start_time:.2f} seconds")
    print(f"Return code: {process.returncode}")

    if process.stdout:
        print("\nSTDOUT:")
        print(process.stdout)

    if process.stderr:
        print("\nSTDERR:")
        print(process.stderr)

    # Log the execution results
    run_number = int(re.search(r'_(\d+)\.py', script_path).group(1))
    log_path = os.path.join(output_dir, f"run_{run_number}_log.txt")

    with open(log_path, "a") as f:
        f.write(f"\n{script_type.capitalize()} execution on {datetime.datetime.now().strftime('%Y-%m-%d %H:%M:%S')}\n")
        f.write(f"Command: {' '.join(cmd)}\n")
        f.write(f"Duration: {end_time - start_time:.2f} seconds\n")
        f.write(f"Return code: {process.returncode}\n")
        f.write("\nSTDOUT:\n")
        f.write(process.stdout)
        f.write("\nSTDERR:\n")
        f.write(process.stderr)

    return process.returncode


def run_evaluation(results_file, test_file, output_file, dataset_name, output_dir, inference_path):
    """
    Run the evaluation script to calculate metrics.

    Args:
        results_file (str): Path to the results file with predictions
        test_file (str): Path to the test file with true labels
        output_file (str): Path to save the evaluation metrics
        dataset_name (str): Dataset name for paths
        output_dir (str): Directory for output files
        inference_path (str): Path to the inference script (for getting run number)

    Returns:
        int: Return code from the evaluation script
    """
    print(f"\n{'=' * 40}")
    print(f"Running evaluation script")
    print(f"{'=' * 40}\n")

    # Define path to test file with labels
    test_labels_file = f"/home/eddy/Documents/Agentomics-ML/datasets/{dataset_name}/human_nontata_promoters_test.csv"

    # Run the evaluation script
    cmd = ["python", "/home/eddy/Documents/Agentomics-ML/src/eval/evaluate_result.py",
           "--results", results_file,
           "--test", test_labels_file,
           "--output", output_file]

    print(f"Command: {' '.join(cmd)}")

    start_time = time.time()
    process = subprocess.run(cmd, capture_output=True, text=True)
    end_time = time.time()

    # Print results
    print(f"\nEvaluation completed in {end_time - start_time:.2f} seconds")
    print(f"Return code: {process.returncode}")

    if process.stdout:
        print("\nSTDOUT:")
        print(process.stdout)

    if process.stderr:
        print("\nSTDERR:")
        print(process.stderr)

    # Log the execution results
    run_number = int(re.search(r'_(\d+)\.py', inference_path).group(1))
    log_path = os.path.join(output_dir, f"run_{run_number}_log.txt")

    with open(log_path, "a") as f:
        f.write(f"\nEvaluation execution on {datetime.datetime.now().strftime('%Y-%m-%d %H:%M:%S')}\n")
        f.write(f"Command: {' '.join(cmd)}\n")
        f.write(f"Duration: {end_time - start_time:.2f} seconds\n")
        f.write(f"Return code: {process.returncode}\n")
        f.write("\nSTDOUT:\n")
        f.write(process.stdout)
        f.write("\nSTDERR:\n")
        f.write(process.stderr)

    return process.returncode


def call_claude_code(prompt, model=None, temp=0.7, output_json=True):
    """
    Call Claude Code CLI to generate a response.

    Args:
        prompt (str): The prompt to send to Claude
        model (str, optional): The Claude model to use (set via environment if provided)
        temp (float): Temperature for generation
        output_json (bool): Whether to output in JSON format

    Returns:
        str: The response from Claude
    """
    # Create environment with optional model configuration
    env = os.environ.copy()
    if model:
        env["ANTHROPIC_MODEL"] = model

    # Build the command
    cmd = ["claude", "-p", prompt]
    if output_json:
        cmd.append("--json")

    # Call Claude Code with the prompt
    print("Sending request to Claude Code...")
    try:
        process = subprocess.run(
            cmd,
            capture_output=True,
            text=True,
            env=env
        )

        if process.returncode != 0:
            print(f"Claude Code returned error (code {process.returncode}):")
            print(process.stderr)
            return None

        # If JSON output was requested, parse it
        if output_json:
            try:
                response_data = json.loads(process.stdout)
                return response_data.get("message", {}).get("content", "")
            except json.JSONDecodeError:
                print("Error parsing JSON response from Claude Code")
                print(f"Raw output: {process.stdout}")
                return process.stdout

        return process.stdout

    except Exception as e:
        print(f"Error calling Claude Code: {e}")
        return None


def run_existing_scripts(output_dir, dataset_name):
    """
    Run the most recent scripts in the output directory.

    Args:
        output_dir (str): Directory containing the scripts
        dataset_name (str): Dataset name for inference paths
    """
    # Find the highest run number
    run_number = get_run_number(output_dir) - 1

    if run_number <= 0:
        print("No existing scripts found.")
        return

    train_path = os.path.join(output_dir, f"train_{run_number}.py")
    inference_path = os.path.join(output_dir, f"inference_{run_number}.py")

    # Run train.py
    train_result = run_script(train_path, 'train', output_dir)

    # If training was successful, run inference.py
    if train_result == 0:
        print("\nTraining completed successfully. Running inference...")
        inference_result = run_script(inference_path, 'inference', output_dir, dataset_name)

        # If inference was successful, run evaluation
        if inference_result == 0:
            print("\nInference completed successfully. Running evaluation...")

            # Paths for evaluation
            results_file = os.path.join(output_dir, "predictions.csv")
            metrics_file = os.path.join(output_dir, f"metrics_run_{run_number}.txt")

            run_evaluation(results_file, None, metrics_file, dataset_name, output_dir, inference_path)
        else:
            print("\nInference failed. Skipping evaluation.")
    else:
        print("\nTraining failed. Skipping inference and evaluation.")


def main():
    # Parse command line arguments
    parser = argparse.ArgumentParser(description="Generate ML code with one-shot LLM using Claude Code")
    parser.add_argument("--dataset", default="human_non_tata_promoters",
                        help="Dataset name")
    parser.add_argument("--model", default="claude-3-7-sonnet-20250219",
                        help="Claude model to use")
    parser.add_argument("--temp", type=float, default=0.7,
                        help="Temperature for LLM generation")
    parser.add_argument("--output-dir",
                        default="/home/eddy/Documents/Agentomics-ML/datasets/competitors/1-shot_llm_agent/claude",
                        help="Directory to save generated scripts")
    parser.add_argument("--skip-execution", action="store_true",
                        help="Skip running the generated scripts")
    parser.add_argument("--skip-generation", action="store_true",
                        help="Skip LLM generation and run existing scripts in the output directory")
    args = parser.parse_args()

    # Check if Claude Code is installed
    if not check_claude_code_installed():
        print("Error: Claude Code is not installed or not available in PATH")
        print("Please install Claude Code with: npm install -g @anthropic-ai/claude-code")
        print("For more information, visit: https://docs.anthropic.com/en/docs/claude-code")
        return

    # Check if we should skip generation and just run existing scripts
    if args.skip_generation:
        run_existing_scripts(args.output_dir, args.dataset)
        return

    # Get the next run number for the model file name
    run_number = get_run_number(args.output_dir)

    # Create the prompt
    prompt = f"""
You are an expert bioinformatics ML engineer. I need you to create a machine learning model for a classification task.

- Create a machine learning classifier for the dataset:
- Training file: human_nontata_promoters_train.csv
- Test file: human_nontata_promoters_test.no_label.csv

DATASET INFO:
- This dataset contains DNA sequences for classification
- Format: CSV with columns 'sequence' (DNA sequence) and 'class' (0 or 1)
- The sequences represent non-TATA promoters (class=1) or non-promoters (class=0)
- Each sequence is about 251 nucleotides long, specifically: the sequences have a uniform length of 251 characters
- The dataset is balanced (50% positive, 50% negative examples)
- The dataset contains sequences of nucleotides 'A', 'G', 'T', 'C' and 'N'

REQUIREMENTS:
1. You must write TWO Python scripts:
   - train.py: to train and save the model
   - inference.py: to load the model and make predictions

2. The inference.py script must:
   - Accept command-line arguments: 
        --input
        --output
   - Output a CSV file with a column named 'prediction'
   - Use ABSOLUTE PATHS for any file references
   - Handle input file format issues appropriately
   - Must compute and report the AUC (Area Under the ROC Curve) as the final score to evaluate the performance of the model.

3. You should create a machine learning model that:
   - Is appropriate for DNA sequence classification
   - Has good generalization ability
   - Uses appropriate sequence encoding (one-hot, k-mer, etc.)

4. IMPORTANT: The inference.py script must save raw probability scores (NOT binary classifications) to the prediction column.
   - DO NOT convert predictions to binary before saving
   - INCORRECT: output = pd.DataFrame({{'prediction': (predictions > 0.5).astype(int)}})
   - CORRECT: output = pd.DataFrame({{'prediction': predictions.flatten()}})
   - The raw probabilities are needed to calculate AUC and other metrics properly

5. Ensure that the final tensor shape after preprocessing matches the expected input dimension for the Dense layers. 
   If using one-hot encoding or a flattening layer, verify that the product of the sequence length and number of features 
   matches the Dense layer's input. If there is any discrepancy between the computed shape and the expected input dimension, 
   adjust the preprocessing pipeline or modify the model architecture to avoid dimension mismatches.

6. VERY IMPORTANT: Save the trained model file with a specific path and filename:
   - The model MUST be saved to: {args.output_dir}/model_{run_number}.h5
   - The inference.py script MUST load the model from this exact path
   - DO NOT save the model to the default models directory
   - Example code for saving: model.save("{args.output_dir}/model_{run_number}.h5")
   - Example code for loading: model = tf.keras.models.load_model("{args.output_dir}/model_{run_number}.h5")

First, explain your approach - what model architecture you'll use and why, how you'll encode the DNA sequences, etc.

Then, provide the complete code for both train.py and inference.py scripts.
Use clear headers like "# train.py" and "# inference.py" to separate the scripts.
"""

    # Call Claude Code
    print("Sending request to Claude Code...")
    response_content = call_claude_code(prompt, model=args.model, temp=args.temp)

    if not response_content:
        print("Error: No response received from Claude Code")
        return

    # Print the response for copying
    print("\n" + "=" * 80)
    print("CLAUDE RESPONSE - COPY FROM BELOW:")
    print("=" * 80 + "\n")
    print(response_content)
    print("\n" + "=" * 80)
    print("END OF RESPONSE - COPY UNTIL ABOVE")
    print("=" * 80)

    # Extract and save scripts
    train_script, inference_script = extract_scripts(response_content)
    train_path, inference_path = save_scripts(train_script, inference_script, args.output_dir)

    # Run the scripts if paths are valid and execution is not skipped
    if not args.skip_execution and train_path and inference_path:
        print("\nExecuting generated scripts...")

        # Run train.py
        train_result = run_script(train_path, 'train', args.output_dir)

        # If training was successful, run inference.py
        if train_result == 0:
            print("\nTraining completed successfully. Running inference...")
            inference_result = run_script(inference_path, 'inference', args.output_dir, args.dataset)

            # If inference was successful, run evaluation
            if inference_result == 0:
                print("\nInference completed successfully. Running evaluation...")

                # Paths for evaluation
                results_file = os.path.join(args.output_dir, "predictions.csv")
                run_number = int(re.search(r'_(\d+)\.py', inference_path).group(1))
                metrics_file = os.path.join(args.output_dir, f"metrics_run_{run_number}.txt")

                run_evaluation(results_file, None, metrics_file, args.dataset, args.output_dir, inference_path)
            else:
                print("\nInference failed. Skipping evaluation.")
        else:
            print("\nTraining failed. Skipping inference and evaluation.")


if __name__ == "__main__":
    main()