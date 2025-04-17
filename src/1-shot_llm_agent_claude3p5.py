#!/usr/bin/env python3
import os
import argparse
import dotenv
import re
import subprocess
import time
import datetime
from anthropic import Anthropic


def extract_scripts(response_text):
    python_blocks = re.findall(r'```python\s+(.*?)```', response_text, re.DOTALL)
    if len(python_blocks) < 2:
        raise ValueError("Could not find two Python code blocks in the response")
    train_script = python_blocks[0]
    inference_script = python_blocks[1]
    return train_script, inference_script


def save_scripts(train_script, inference_script, output_dir, run_name):
    os.makedirs(output_dir, exist_ok=True)

    train_path = os.path.join(output_dir, f"train_{run_name}.py")
    inference_path = os.path.join(output_dir, f"inference_{run_name}.py")

    with open(train_path, "w") as f:
        f.write(train_script)
    with open(os.path.join(output_dir, "train.py"), "w") as f:
        f.write(train_script)

    with open(inference_path, "w") as f:
        f.write(inference_script)
    with open(os.path.join(output_dir, "inference.py"), "w") as f:
        f.write(inference_script)

    return train_path, inference_path


def run_script(script_path, script_type, output_dir, dataset_name=None, base_path=None):
    cmd = ["python", script_path]

    if script_type == 'inference' and dataset_name:
        test_path = os.path.join(base_path, f"datasets/{dataset_name}/human_nontata_promoters_test.no_label.csv")
        output_path = os.path.join(output_dir, "predictions.csv")
        cmd.extend(["--input", test_path, "--output", output_path])

    process = subprocess.run(cmd, capture_output=True, text=True)

    if process.stdout:
        print(process.stdout)
    if process.stderr:
        print(process.stderr)

    return process.returncode


def run_evaluation(results_file, dataset_name, output_dir, run_name, base_path):
    test_labels_file = os.path.join(base_path, f"datasets/{dataset_name}/human_nontata_promoters_test.csv")
    metrics_file = os.path.join(output_dir, f"metrics_run_{run_name}.txt")

    eval_script_path = os.path.join(base_path, "src/eval/evaluate_result_no-logging.py")

    cmd = ["python", eval_script_path,
           "--results", results_file,
           "--test", test_labels_file,
           "--output", metrics_file]

    process = subprocess.run(cmd, capture_output=True, text=True)

    if process.stdout:
        print(process.stdout)
    if process.stderr:
        print(process.stderr)

    return process.returncode


def generate_and_run_scripts(client, model, temperature, output_dir, dataset_name,
                             base_run_name, attempt, base_path):
    run_name = f"{base_run_name}_attempt{attempt}"

    train_file_path = os.path.join(base_path, "datasets/human_non_tata_promoters/human_nontata_promoters_train.csv")
    test_file_path  = os.path.join(base_path,
                                   "datasets/human_non_tata_promoters/human_nontata_promoters_test.no_label.csv")

    prompt = f"""
You are an expert bioinformatics ML engineer. Create a machine learning model for DNA sequence classification.

DATASET:
- Training file: {train_file_path}
- Test file: {test_file_path}
- Format: CSV with columns 'sequence' (DNA sequence, 251 nucleotides long) and 'class' (0 or 1)
- Contains sequences of nucleotides 'A', 'G', 'T', 'C' and 'N'
- Classifies non-TATA promoters (class=1) vs non-promoters (class=0)

REQUIREMENTS:
1. Create TWO Python scripts: train.py and inference.py

2. For train.py:
   - Train a robust model suitable for DNA sequence classification
   - Handle encoding of DNA sequences appropriately
   - Use validation to ensure good generalization
   - Save the trained model to: {output_dir}/model_{run_name}.h5

3. For inference.py:
   - Accept arguments: --input and --output
   - Load the model from: {output_dir}/model_{run_name}.h5
   - Output a CSV with column 'prediction' containing RAW PROBABILITIES (not binary classes)
   - Use pd.DataFrame({{'prediction': predictions.flatten()}}) to save predictions
   - Compute and report AUC

Provide complete code for both scripts with "# train.py" and "# inference.py" headers.
"""

    # === Claude call (Anthropic) ===
    response = client.messages.create(
        model=model,
        messages=[{"role": "user", "content": prompt}],
        temperature=temperature,
        max_tokens=4000
    )
    response_content = "".join(block.text for block in response.content)
    # ===============================

    try:
        train_script, inference_script = extract_scripts(response_content)
        train_path, inference_path = save_scripts(train_script, inference_script, output_dir, run_name)

        print(f"\nRunning training for attempt {attempt}...")
        train_result = run_script(train_path, 'train', output_dir, dataset_name, base_path)
        if train_result != 0:
            raise Exception("Training script failed")

        print(f"\nRunning inference for attempt {attempt}...")
        inference_result = run_script(inference_path, 'inference', output_dir, dataset_name, base_path)
        if inference_result != 0:
            raise Exception("Inference script failed")

        print(f"\nRunning evaluation for attempt {attempt}...")
        results_file = os.path.join(output_dir, "predictions.csv")
        eval_result = run_evaluation(results_file, dataset_name, output_dir, run_name, base_path)
        if eval_result != 0:
            raise Exception("Evaluation failed")

        print(f"\nSuccess! Pipeline completed successfully for attempt {attempt}")
        return True

    except Exception as e:
        print(f"\nAttempt {attempt} failed: {e}")
        return False


def main():
    dotenv.load_dotenv()

    parser = argparse.ArgumentParser(description="Generate ML code with one-shot LLM and retry with new generations")
    parser.add_argument("--dataset", default="human_non_tata_promoters")
    parser.add_argument("--model", default="claude-3.5-sonnet-20240620")
    parser.add_argument("--temp", type=float, default=0.7)
    parser.add_argument("--api", default="anthropic", choices=["anthropic"])
    parser.add_argument("--base-path", default="/home/user/Documents/Agentomics-ML",
                        help="Base path to the Agentomics-ML directory")
    parser.add_argument("--output-dir", default=None)
    parser.add_argument("--run-name", default="ed_run")
    parser.add_argument("--max-attempts", type=int, default=5)
    args = parser.parse_args()

    if args.output_dir is None:
        args.output_dir = os.path.join(args.base_path,
                                       "datasets/competitors/1-shot_llm_agent/gpt4o")

    base_run_name = args.run_name

    api_key = os.getenv("ANTHROPIC_API_KEY")
    if not api_key:
        raise EnvironmentError("ANTHROPIC_API_KEY must be set in your .env file")
    client = Anthropic(api_key=api_key)

    print(f"Starting up to {args.max_attempts} independent LLM generations and executions")
    print(f"Using model: {args.model}, temperature: {args.temp}")
    print(f"Base path: {args.base_path}")
    print(f"Output directory: {args.output_dir}")

    successes = 0
    for attempt in range(1, args.max_attempts + 1):
        print(f"\n{'=' * 50}")
        print(f"ATTEMPT {attempt}/{args.max_attempts}")
        print(f"{'=' * 50}")

        success = generate_and_run_scripts(
            client,
            args.model,
            args.temp,
            args.output_dir,
            args.dataset,
            base_run_name,
            attempt,
            args.base_path
        )

        if success:
            successes += 1

    if successes > 0:
        print(f"\n{successes} out of {args.max_attempts} attempts completed successfully.")
    else:
        print(f"\nAll {args.max_attempts} attempts failed.")


if __name__ == "__main__":
    main()
