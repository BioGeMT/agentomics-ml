#!/usr/bin/env python3
import os
import sys
import argparse
import dotenv
import re
import subprocess
import wandb

# Add the parent directory to sys.path to import modules
sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

# Import existing functions
from run_logging.wandb import setup_logging
from run_logging.logging_helpers import log_inference_stage_and_metrics
from run_logging.evaluate_log_run import evaluate_log_metrics
from openai import OpenAI
from anthropic import Anthropic


def get_client(provider):
    """Create and return appropriate client based on provider."""
    if provider == "openai":
        api_key = os.getenv("OPENAI_API_KEY")
        if not api_key:
            raise EnvironmentError("OPENAI_API_KEY must be set in your .env file")
        return OpenAI(api_key=api_key)
    elif provider == "openrouter":
        api_key = os.getenv("OPENROUTER_API_KEY") or os.getenv("OPENAI_API_KEY")
        if not api_key:
            raise EnvironmentError("OPENROUTER_API_KEY must be set in your .env file")
        return OpenAI(api_key=api_key, base_url="https://openrouter.ai/api/v1")
    elif provider == "anthropic":
        api_key = os.getenv("ANTHROPIC_API_KEY")
        if not api_key:
            raise EnvironmentError("ANTHROPIC_API_KEY must be set in your .env file")
        return Anthropic(api_key=api_key)


def get_llm_response(client, provider, model, prompt, temperature, max_tokens):
    """Get response from LLM with unified handling."""
    response_kwargs = {
        "model": model,
        "messages": [{"role": "user", "content": prompt}],
        "temperature": temperature,
        "max_tokens": max_tokens
    }

    response = (client.chat.completions.create(**response_kwargs) if provider == "openai"
                else client.messages.create(**response_kwargs))

    return (response.choices[0].message.content if provider == "openai"
            else "".join(block.text for block in response.content))


def extract_scripts(response_text):
    python_blocks = re.findall(r'```python\s+(.*?)```', response_text, re.DOTALL)
    if len(python_blocks) < 2:
        raise ValueError("Expected at least two Python code blocks in the response")
    train_script = python_blocks[0]
    inference_script = python_blocks[1]
    return train_script, inference_script


def save_scripts(train_script, inference_script, output_dir, run_name):
    os.makedirs(output_dir, exist_ok=True)

    train_path = os.path.join(output_dir, f"train_{run_name}.py")
    inference_path = os.path.join(output_dir, f"inference_{run_name}.py")

    with open(train_path, "w") as f:
        f.write(train_script)

    with open(inference_path, "w") as f:
        f.write(inference_script)
    return train_path, inference_path


def run_script(script_path, script_type, output_dir, test_csv_path=None):
    cmd = ["python", script_path]

    if script_type == 'inference' and test_csv_path:
        output_path = os.path.join(output_dir, "predictions.csv")
        cmd.extend(["--input", test_csv_path, "--output", output_path])

    process = subprocess.run(cmd, capture_output=True, text=True)

    # Log the output to wandb
    if process.stdout:
        print(process.stdout)
        wandb.log({f"{script_type}_stdout": process.stdout})
    if process.stderr:
        print(process.stderr)
        wandb.log({f"{script_type}_stderr": process.stderr})

    return process.returncode


def parse_metrics_file(metrics_file):
    metrics = {}
    try:
        with open(metrics_file, 'r') as f:
            for line in f:
                if ':' in line:
                    key, value = line.strip().split(':', 1)
                    metrics[key.strip()] = float(value.strip())
    except Exception as e:
        print(f"Error parsing metrics file: {e}")
    return metrics


def run_evaluation(results_file, test_labels_file, output_dir, run_name, base_path):
    metrics_file = os.path.join(output_dir, f"metrics_run_{run_name}.txt")

    try:
        # Use the imported evaluate_log_metrics function
        metrics = evaluate_log_metrics(
            results_file=results_file,
            test_file=test_labels_file,
            output_file=metrics_file
        )
        return 0
    except Exception as e:
        print(f"Error during evaluation: {e}")
        wandb.log({"eval_error": str(e)})
        return 1


def generate_and_run_scripts(client, provider, model, temperature, output_dir, train_csv_path,
                             test_csv_path, base_run_name, attempt, base_path, max_tokens):
    run_name = f"{base_run_name}_attempt{attempt}"

    # Create specific attempt directory inside output_dir
    attempt_dir = os.path.join(output_dir, f"attempt_{attempt}")
    os.makedirs(attempt_dir, exist_ok=True)

    # Log attempt information
    wandb.log({
        "attempt": attempt,
        "attempt_dir": attempt_dir
    })

    train_file_path = train_csv_path
    test_file_path = test_csv_path

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
   - Save the trained model to: {attempt_dir}/model_{run_name}.h5

3. For inference.py:
   - Accept arguments: --input and --output
   - Load the model from: {attempt_dir}/model_{run_name}.h5
   - Output a CSV with column 'prediction' containing RAW PROBABILITIES (not binary classes)
   - Use pd.DataFrame({{'prediction': predictions.flatten()}}) to save predictions
   - Compute and report AUC

Provide complete code for both scripts with "# train.py" and "# inference.py" headers.
"""

    # Log the prompt
    wandb.log({"prompt": prompt})

    # Get LLM response using the unified function
    response_content = get_llm_response(client, provider, model, prompt, temperature, max_tokens)

    # Log the LLM response
    wandb.log({"llm_response": response_content})

    try:
        # Extract and save scripts
        train_script, inference_script = extract_scripts(response_content)
        train_path, inference_path = save_scripts(train_script, inference_script, attempt_dir, run_name)

        # Log the scripts
        wandb.log({
            "train_script": train_script,
            "inference_script": inference_script
        })

        # Run training
        print(f"\nRunning training for attempt {attempt}...")
        wandb.log({"stage": "training"})
        train_result = run_script(train_path, 'train', attempt_dir)

        if train_result != 0:
            log_inference_stage_and_metrics(0)
            raise Exception("Training script failed")

        # Run inference
        print(f"\nRunning inference for attempt {attempt}...")
        wandb.log({"stage": "inference"})
        inference_result = run_script(inference_path, 'inference', attempt_dir, test_file_path)

        if inference_result != 0:
            log_inference_stage_and_metrics(1)
            raise Exception("Inference script failed")

        # Run evaluation
        print(f"\nRunning evaluation for attempt {attempt}...")
        wandb.log({"stage": "evaluation"})
        results_file = os.path.join(attempt_dir, "predictions.csv")

        test_with_labels_path = test_file_path.replace(".no_label.csv", ".csv")

        eval_result = run_evaluation(results_file, test_with_labels_path, attempt_dir, run_name, base_path)

        if eval_result != 0:
            log_inference_stage_and_metrics(1)
            raise Exception("Evaluation failed")

        # Parse and log metrics directly
        metrics_file = os.path.join(attempt_dir, f"metrics_run_{run_name}.txt")
        if os.path.exists(metrics_file):
            metrics = parse_metrics_file(metrics_file)
            log_inference_stage_and_metrics(2, metrics)

        print(f"\nSuccess! Pipeline completed successfully for attempt {attempt}")
        return True

    except Exception as e:
        print(f"\nAttempt {attempt} failed: {e}")
        wandb.log({"error": str(e)})
        return False


def main():
    dotenv.load_dotenv()

    parser = argparse.ArgumentParser(description="Generate ML code with one-shot LLM and retry with new generations")
    parser.add_argument("--dataset", required=True, help="Dataset name")
    parser.add_argument("--provider", required=True, choices=["openai", "anthropic", "openrouter"],
                        help="The API provider to use")
    parser.add_argument("--model", required=True,
                        help="Model name (e.g., gpt-4o-2024-08-06, claude-3.5-sonnet-20240620)")
    parser.add_argument("--temp", type=float, required=True, help="Temperature for LLM generation")
    parser.add_argument("--max-tokens", type=int, required=True, help="Maximum tokens for LLM response")
    parser.add_argument("--base-path", required=True, help="Base path to the Agentomics-ML directory")
    parser.add_argument("--output-dir", required=True, help="Output directory path")
    parser.add_argument("--run-name", required=True, help="Base name for the run")
    parser.add_argument("--max-attempts", type=int, required=True, help="Maximum number of attempts")
    parser.add_argument("--train-csv", required=True, help="Path to the training CSV file")
    parser.add_argument("--test-csv", required=True, help="Path to the test CSV file without labels")
    parser.add_argument("--tags", nargs='+', default=["testing"], help="Tags for wandb run")

    args = parser.parse_args()

    # Create model-specific output directory
    model_name = args.model.replace("-", "_").replace(".", "_")
    output_dir = os.path.join(args.output_dir, f"competitors/{model_name}")
    os.makedirs(output_dir, exist_ok=True)

    # Setup config for wandb
    config = {
        "dataset": args.dataset,
        "model": args.model,
        "provider": args.provider,
        "temperature": args.temp,
        "max_tokens": args.max_tokens,
        "run_name": args.run_name,
        "max_attempts": args.max_attempts,
        "tags": args.tags,
        # Add agent_id to match what setup_logging expects
        "agent_id": f"{model_name}_{args.run_name}"
    }

    # Setup wandb logging
    wandb_key = os.getenv("WANDB_API_KEY")
    if not wandb_key:
        raise EnvironmentError("WANDB_API_KEY must be set in your .env file")
    setup_logging(config, api_key=wandb_key)

    # Get the appropriate client using the unified function
    client = get_client(args.provider)

    print(f"Starting up to {args.max_attempts} independent LLM generations and executions")
    print(f"Using provider: {args.provider}, model: {args.model}, temperature: {args.temp}")
    print(f"Base path: {args.base_path}")
    print(f"Output directory: {output_dir}")
    print(f"Training CSV: {args.train_csv}")
    print(f"Test CSV: {args.test_csv}")

    successes = 0
    for attempt in range(1, args.max_attempts + 1):
        print(f"\n{'=' * 50}")
        print(f"ATTEMPT {attempt}/{args.max_attempts}")
        print(f"{'=' * 50}")

        success = generate_and_run_scripts(
            client,
            args.provider,
            args.model,
            args.temp,
            output_dir,
            args.train_csv,
            args.test_csv,
            args.run_name,
            attempt,
            args.base_path,
            args.max_tokens
        )

        if success:
            successes += 1

    wandb.log({"successful_attempts": successes, "total_attempts": args.max_attempts})

    if successes > 0:
        print(f"\n{successes} out of {args.max_attempts} attempts completed successfully.")
    else:
        print(f"\nAll {args.max_attempts} attempts failed.")

    # Finish the wandb run
    wandb.finish()


if __name__ == "__main__":
    main()