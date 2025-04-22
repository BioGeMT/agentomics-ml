#!/usr/bin/env python3
import os
import argparse
import dotenv
import re
import subprocess
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


def get_llm_response(client, provider, model, prompt, temperature=0.7, max_tokens=4000):
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

    if process.stdout:
        print(process.stdout)
    if process.stderr:
        print(process.stderr)

    return process.returncode


def run_evaluation(results_file, test_labels_file, output_dir, run_name, base_path):
    metrics_file = os.path.join(output_dir, f"metrics_run_{run_name}.txt")

    eval_script_path = os.path.join(base_path, "src/eval/evaluate_result.py")

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


def generate_and_run_scripts(client, provider, model, temperature, output_dir, train_csv_path,
                             test_csv_path, base_run_name, attempt, base_path):
    run_name = f"{base_run_name}_attempt{attempt}"

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
   - Save the trained model to: {output_dir}/model_{run_name}.h5

3. For inference.py:
   - Accept arguments: --input and --output
   - Load the model from: {output_dir}/model_{run_name}.h5
   - Output a CSV with column 'prediction' containing RAW PROBABILITIES (not binary classes)
   - Use pd.DataFrame({{'prediction': predictions.flatten()}}) to save predictions
   - Compute and report AUC

Provide complete code for both scripts with "# train.py" and "# inference.py" headers.
"""

    # Get LLM response using the unified function
    response_content = get_llm_response(client, provider, model, prompt, temperature)

    try:
        # Extract and save scripts
        train_script, inference_script = extract_scripts(response_content)
        train_path, inference_path = save_scripts(train_script, inference_script, output_dir, run_name)

        # Run training
        print(f"\nRunning training for attempt {attempt}...")
        train_result = run_script(train_path, 'train', output_dir)

        if train_result != 0:
            raise Exception("Training script failed")

        # Run inference
        print(f"\nRunning inference for attempt {attempt}...")
        inference_result = run_script(inference_path, 'inference', output_dir, test_file_path)

        if inference_result != 0:
            raise Exception("Inference script failed")

        # Run evaluation
        print(f"\nRunning evaluation for attempt {attempt}...")
        results_file = os.path.join(output_dir, "predictions.csv")

        test_with_labels_path = test_file_path.replace(".no_label.csv", ".csv")

        eval_result = run_evaluation(results_file, test_with_labels_path, output_dir, run_name, base_path)

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
    parser.add_argument("--provider", default="openai", choices=["openai", "anthropic", "openrouter"],
                        help="The API provider to use")
    parser.add_argument("--model", default="gpt-4o-2024-08-06",
                        help="Model name (e.g., gpt-4o-2024-08-06, claude-3.5-sonnet-20240620)")
    parser.add_argument("--temp", type=float, default=0.7)
    parser.add_argument("--base-path", default="/home/user/Documents/Agentomics-ML",
                        help="Base path to the Agentomics-ML directory")
    parser.add_argument("--output-dir", default=None)
    parser.add_argument("--run-name", default="ed_run")
    parser.add_argument("--max-attempts", type=int, default=5)
    parser.add_argument("--train-csv",
                        default="/home/user/Documents/Agentomics-ML/datasets/human_non_tata_promoters/human_nontata_promoters_train.csv",
                        help="Path to the training CSV file (default: derived from dataset name)")
    parser.add_argument("--test-csv",
                        default="/home/user/Documents/Agentomics-ML/datasets/human_non_tata_promoters/human_nontata_promoters_test.no_label.csv",
                        help="Path to the test CSV file without labels (default: derived from dataset name)")

    args = parser.parse_args()

    # Set default output directory based on base path if not provided
    if args.output_dir is None:
        model_dir = "gpt4o" if "gpt" in args.model.lower() else "claude"
        args.output_dir = os.path.join(args.base_path, f"datasets/competitors/1-shot_llm_agent/{model_dir}")

    base_run_name = args.run_name

    # Get the appropriate client using the unified function
    client = get_client(args.provider)

    print(f"Starting up to {args.max_attempts} independent LLM generations and executions")
    print(f"Using provider: {args.provider}, model: {args.model}, temperature: {args.temp}")
    print(f"Base path: {args.base_path}")
    print(f"Output directory: {args.output_dir}")
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
            args.output_dir,
            args.train_csv,
            args.test_csv,
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