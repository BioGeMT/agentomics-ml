import os
import sys
import argparse
import re
import subprocess
import json

from openai import OpenAI
import wandb
import dotenv

sys.path.append("/repository/src")

from run_logging.wandb import setup_logging
from run_logging.logging_helpers import log_inference_stage_and_metrics
from run_logging.evaluate_log_run import evaluate_log_metrics

sys.path.append("/repository/src/utils")
from create_user import create_new_user_and_rundir

def get_llm_response(client, model, prompt, temperature, max_tokens):
    response_kwargs = {
        "model": model,
        "messages": [{"role": "user", "content": prompt}],
        "temperature": temperature,
        "max_tokens": max_tokens
    }
    response = client.chat.completions.create(**response_kwargs)
    return (response.choices[0].message.content)

def extract_scripts(response_text):
    python_blocks = re.findall(r'```python\s+(.*?)```', response_text, re.DOTALL)
    if len(python_blocks) != 2:
        raise ValueError("Expected at two Python code blocks in the response")
    train_script = python_blocks[0]
    inference_script = python_blocks[1]
    
    yaml_blocks = re.findall(r'```(?:yaml|yml)\s+(.*?)```', response_text, re.DOTALL)
    if len(yaml_blocks) != 1:
        raise ValueError("Expected one YAML code block in the response")
    env_yaml = yaml_blocks[0]
    
    return train_script, inference_script, env_yaml

def save_scripts(train_script, inference_script, env_yaml, output_dir, run_name):
    os.makedirs(output_dir, exist_ok=True)
    train_path = os.path.join(output_dir, f"train.py")
    inference_path = os.path.join(output_dir, f"inference.py")
    env_yaml_path = os.path.join(output_dir, f"environment.yaml")

    # Ensure environment.yaml has the name set to run_name_env
    env_name = run_name + '_env'
    if not re.search(f'^name:\\s*{env_name}\\s*$', env_yaml, re.MULTILINE):
        # Replace existing name line if present
        env_yaml = re.sub(r'^name:.*$', f'name: {env_name}', env_yaml, flags=re.MULTILINE)
        # If no name line exists, add it at the beginning
        if not re.search(r'^name:', env_yaml, re.MULTILINE):
            env_yaml = f"name: {env_name}\n{env_yaml}"

    with open(train_path, "w") as f:
        f.write(train_script)
    with open(inference_path, "w") as f:
        f.write(inference_script)
    with open(env_yaml_path, "w") as f:
        f.write(env_yaml)

    return train_path, inference_path, env_yaml_path

def run_script(script_path, script_type, output_dir, run_name, test_csv_no_labels_path=None):
    if script_type == 'inference':
        cmd = f"source activate {run_name}_env && python {script_path} --input {test_csv_no_labels_path} --output {output_dir}/eval_predictions.csv"
    if script_type == 'train':
        cmd = f"source activate {run_name}_env && python {script_path}"

    print(f"Executing command: {' '.join(cmd)}")
    process = subprocess.run(cmd, capture_output=True, text=True, shell=True, executable="/usr/bin/bash")
    print(f"STDOUT: {process.stdout}")
    print(f"STDERR: {process.stderr}")
    return process.returncode

def run_evaluation(results_file, test_labels_file, output_dir, label_to_scalar, class_col):
    print(f"\nRunning evaluation...")
    metrics_file_path = os.path.join(output_dir, f"metrics.txt")
    return evaluate_log_metrics(
        results_file=results_file,
        test_file=test_labels_file,
        output_file=metrics_file_path,
        label_to_scalar=label_to_scalar,
        class_col=class_col,
    )

def generate_and_run_scripts(client, model, dataset, temperature, run_name, max_tokens):
    run_dir = os.path.join("/workspace/runs", run_name)
    with open(f"/repository/datasets/{dataset}/metadata.json") as f:
        dataset_metadata = json.load(f)
    train_csv_path = dataset_metadata['train_split']
    test_csv_no_labels_path = dataset_metadata['test_split_no_labels']
    test_csv_path = dataset_metadata['test_split_with_labels']
    label_to_scalar = dataset_metadata['label_to_scalar']
    class_col = dataset_metadata['class_col']
    dataset_knowledge_path = dataset_metadata['dataset_knowledge']
    with open(dataset_knowledge_path) as f:
        dataset_knowledge = f.read()

    dataset_to_hints = {
        "human_nontata_promoters": 
            """
            - Format: CSV with columns 'sequence' (DNA sequence, 251 nucleotides long) and 'numeric_label' (0 or 1)
            - Contains sequences of nucleotides 'A', 'G', 'T', 'C' and 'N'
            """,
        "drosophila_enhancers_stark":
            """
            - Format: CSV with columns 'sequence' (DNA sequence, varying lengths) and 'numeric_label' (0 or 1)
            - Contains sequences of nucleotides 'A', 'G', 'T', 'C'
            """,
        "human_enhancers_cohn":
            """
            - Format: CSV with columns 'sequence' (DNA sequence, 500 nucleotides long) and 'numeric_label' (0 or 1)
            - Contains sequences of nucleotides 'A', 'G', 'T', 'C'
            """,
        "human_enhancers_ensembl":
            """
            - Format: CSV with columns 'sequence' (DNA sequence, varying lengths) and 'numeric_label' (0 or 1)
            - Contains sequences of nucleotides 'A', 'G', 'T', 'C' and 'N'
            """,
        "human_ensembl_regulatory":
            """
            - Format: CSV with columns 'sequence' (DNA sequence, varying lengths) and 'numeric_label' (0, 1, or 2),
            - Contains sequences of nucleotides 'A', 'G', 'T', 'C' and 'N'
            """,
        "human_ocr_ensembl":
            """
            - Format: CSV with columns 'sequence' (DNA sequence, varying lengths) and 'numeric_label' (0 or 1)
            - Contains sequences of nucleotides 'A', 'G', 'T', 'C' and 'N'
            """,
        "AGO2_CLASH_Hejret2023":
            """
            - Format: CSV with columns 'gene' (DNA sequence, 50 nucleotides long), 'noncodingRNA' (varying length) and 'numeric_label' (0 or 1)
            - Contains sequences of nucleotides 'A', 'G', 'T', 'C'
            """,

    }
    prompt = f"""
        Create the best possible classifier that will generalize to new unseen data.
        You are using a linux system.
        You have access to both CPU and GPU resources.

        DATASET:
        - Training file: {train_csv_path}
        - Test file: {test_csv_no_labels_path}
        {dataset_to_hints[dataset]}

        Dataset knowledge:
        {dataset_knowledge}

        REQUIREMENTS:
        1. Create three files:
           - train.py
           - inference.py
           - environment.yaml

        2. For train.py:
        - Train a robust model suitable for the given dataset
        - Save the trained model to: {run_dir}/model.pkl using joblib or pickle

        3. For inference.py:
        - Accept arguments: --input and --output
        - Load the model from: {run_dir}/model.pkl
        - Output a CSV with column 'prediction' containing a score from 0 to 1

        4. For environment.yaml:
        - Create a conda environment file with all necessary packages
        - Include all libraries used in both train.py and inference.py

        Provide complete code for all files with headers "# train.py", "# inference.py", and "# environment.yaml".
        """

    response_content = get_llm_response(client, model, prompt, temperature, max_tokens)
    print(response_content)
    try:
        train_script, inference_script, env_yaml = extract_scripts(response_content)
    except Exception as e:
        print(e)
        log_inference_stage_and_metrics(0)
        return
    train_path, inference_path, env_yaml_path = save_scripts(train_script, inference_script, env_yaml, run_dir, run_name)

    # Create conda environment
    env_result = subprocess.run(f"conda env create -f {env_yaml_path}", shell=True, capture_output=True, text=True)
    if env_result.returncode != 0:
        log_inference_stage_and_metrics(0)
        wandb.log({"conda_creation_failed": True})
        print(f"Error creating conda environment: {env_result.stderr}")
        return env_result.returncode

    error_code = run_script(script_path=train_path, script_type='train', output_dir=run_dir, run_name=run_name)
    if error_code != 0:
        log_inference_stage_and_metrics(0)
        return

    error_code = run_script(script_path=inference_path, script_type='inference', output_dir=run_dir, run_name=run_name, test_csv_no_labels_path=test_csv_no_labels_path)
    if error_code != 0:
        log_inference_stage_and_metrics(1)
        return

    try:
        results_file = os.path.join(run_dir, "eval_predictions.csv")
        run_evaluation(results_file, test_csv_path, run_dir, label_to_scalar, class_col)
    except Exception as e:
        print(f"Error during evaluation: {e}")
        log_inference_stage_and_metrics(1)

def parse_args():
    parser = argparse.ArgumentParser(description="Generate ML code with one-shot LLM")
    parser.add_argument("--dataset", required=True, help="Dataset name")
    parser.add_argument("--model", required=True,
                        help="Model name (e.g., gpt-4o-2024-08-06, claude-3.5-sonnet-20240620)")
    parser.add_argument("--temp", type=float, required=True, help="Temperature for LLM generation")
    parser.add_argument("--max-tokens", type=int, required=True, help="Maximum tokens for LLM response")
    parser.add_argument("--tags", required=True, nargs='+', help="List of tags for wandb run")

    return parser.parse_args()

def main():
    dotenv.load_dotenv()
    args = parse_args()

    run_id = create_new_user_and_rundir()
    print(f"Generated unique run ID: {run_id}")

    config = {
        "dataset": args.dataset,
        "model": args.model,
        "temperature": args.temp,
        "max_tokens": args.max_tokens,
        "run_id": run_id,
        "tags": args.tags,
        "agent_id": run_id,
    }

    wandb_key = os.getenv("WANDB_API_KEY")
    setup_logging(config, api_key=wandb_key)

    client = OpenAI(base_url="https://openrouter.ai/api/v1", api_key=os.getenv("OPENROUTER_API_KEY"))

    generate_and_run_scripts(
        client=client,
        model=args.model,
        dataset=args.dataset,
        temperature=args.temp,
        run_name=run_id,
        max_tokens=args.max_tokens,
    )
    wandb.finish()

if __name__ == "__main__":
    main()