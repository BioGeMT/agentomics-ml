import os
import sys
import argparse
import dotenv
import re
import subprocess
import wandb
import json

# Add the repository src directory to sys.path to import modules
sys.path.append("/repository/src")

# Import existing functions
from run_logging.wandb import setup_logging
from run_logging.logging_helpers import log_inference_stage_and_metrics
from run_logging.evaluate_log_run import evaluate_log_metrics
from openai import OpenAI
from anthropic import Anthropic

# Import create_user function
sys.path.append("/repository/src/utils")
from create_user import create_new_user_and_rundir

def get_client(provider):
    if provider == "openai":
        return OpenAI(api_key=os.getenv("OPENAI_API_KEY"))
    if provider == "openrouter":
        return OpenAI(api_key=os.getenv("OPENROUTER_API_KEY"), base_url="https://openrouter.ai/api/v1")
    if provider == "anthropic":
        return Anthropic(api_key=os.getenv("ANTHROPIC_API_KEY"))

def get_llm_response(client, provider, model, prompt, temperature, max_tokens):
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

def run_script(script_path, script_type, output_dir, test_csv_no_labels_path=None):
    cmd = ["python", script_path]
    if script_type == 'inference' and test_csv_no_labels_path:
        output_path = os.path.join(output_dir, "predictions.csv")
        cmd.extend(["--input", test_csv_no_labels_path, "--output", output_path])

    print(f"Executing command: {' '.join(cmd)}")
    process = subprocess.run(cmd, capture_output=True, text=True)
    print(f"STDOUT: {process.stdout}")
    print(f"STDERR: {process.stderr}")
    return process.returncode

def run_evaluation(results_file, test_labels_file, output_dir):
    print(f"\nRunning evaluation...")
    metrics_file_path = os.path.join(output_dir, f"metrics.txt")
    return evaluate_log_metrics(
        results_file=results_file,
        test_file=test_labels_file,
        output_file=metrics_file_path
    )

#TODO promoter-specific prompt!
def generate_and_run_scripts(client, provider, model, temperature, train_csv_path,
                             test_csv_no_labels_path, test_csv_path, run_name, max_tokens):
    run_dir = os.path.join("/workspace/runs", run_name)
    prompt = f"""
You are an expert bioinformatics ML engineer. Create a machine learning model for DNA sequence classification.

DATASET:
- Training file: {train_csv_path}
- Test file: {test_csv_no_labels_path}
- Format: CSV with columns 'sequence' (DNA sequence, 251 nucleotides long) and 'class' (0 or 1)
- Contains sequences of nucleotides 'A', 'G', 'T', 'C' and 'N'
- Classifies non-TATA promoters (class=1) vs non-promoters (class=0)

REQUIREMENTS:
1. Create TWO Python scripts: train.py and inference.py

2. For train.py:
   - Train a robust model suitable for DNA sequence classification
   - Handle encoding of DNA sequences appropriately
   - Use validation to ensure good generalization
   - Save the trained model to: {run_dir}/model_{run_name}.pkl using joblib or pickle

3. For inference.py:
   - Accept arguments: --input and --output
   - Load the model from: {run_dir}/model_{run_name}.pkl
   - Output a CSV with column 'prediction' containing RAW PROBABILITIES (not binary classes)
   - Use pd.DataFrame({{'prediction': predictions.flatten()}}) to save predictions
   - Compute and report AUC

Provide complete code for both scripts with "# train.py" and "# inference.py" headers.
"""

    response_content = get_llm_response(client, provider, model, prompt, temperature, max_tokens)
    train_script, inference_script = extract_scripts(response_content)
    train_path, inference_path = save_scripts(train_script, inference_script, run_dir, run_name)

    error_code = run_script(train_path, 'train', run_dir)
    if error_code != 0:
        log_inference_stage_and_metrics(0)
        return

    error_code = run_script(inference_path, 'inference', run_dir, test_csv_no_labels_path)
    if error_code != 0:
        log_inference_stage_and_metrics(1)
        return

    try:
        results_file = os.path.join(run_dir, "predictions.csv")
        run_evaluation(results_file, test_csv_path, run_dir)
    except Exception as e:
        print(f"Error during evaluation: {e}")
        log_inference_stage_and_metrics(1)

def parse_args():
    parser = argparse.ArgumentParser(description="Generate ML code with one-shot LLM")
    parser.add_argument("--dataset", required=True, help="Dataset name")
    parser.add_argument("--provider", required=True, choices=["openai", "anthropic", "openrouter"],
                        help="The API provider to use")
    parser.add_argument("--model", required=True,
                        help="Model name (e.g., gpt-4o-2024-08-06, claude-3.5-sonnet-20240620)")
    parser.add_argument("--temp", type=float, required=True, help="Temperature for LLM generation")
    parser.add_argument("--max-tokens", type=int, required=True, help="Maximum tokens for LLM response")
    parser.add_argument("--tags", nargs='+', default=["testing"], help="Tags for wandb run")

    return parser.parse_args()

def main():
    dotenv.load_dotenv()
    args = parse_args()

    run_id = create_new_user_and_rundir()
    print(f"Generated unique run ID: {run_id}")

    config = {
        "dataset": args.dataset,
        "model": args.model,
        "provider": args.provider,
        "temperature": args.temp,
        "max_tokens": args.max_tokens,
        "run_id": run_id,
        "tags": args.tags,
        "agent_id": run_id,
    }

    wandb_key = os.getenv("WANDB_API_KEY")
    if not wandb_key:
        raise EnvironmentError("WANDB_API_KEY must be set in your .env file")
    setup_logging(config, api_key=wandb_key)

    client = get_client(config['provider'])

    with open(f"/repository/datasets/{config['dataset']}/metadata.json") as f:
        dataset_metadata = json.load(f)
    train_csv = dataset_metadata['train_split']
    test_csv_no_labels = dataset_metadata['test_split_no_labels']
    test_csv = dataset_metadata['test_split_with_labels']

    generate_and_run_scripts(
        client=client,
        provider=args.provider,
        model=args.model,
        temperature=args.temp,
        train_csv_path=train_csv,
        test_csv_no_labels_path=test_csv_no_labels,
        test_csv_path = test_csv,
        run_name=run_id,
        max_tokens=args.max_tokens
    )
    wandb.finish()

if __name__ == "__main__":
    main()