#!/usr/bin/env python3
"""
Zero-Shot LLM Agent for BioMLBench
"""

import os
import sys
import argparse
import re
import subprocess
import json
from pathlib import Path
from openai import OpenAI

def parse_args():
    parser = argparse.ArgumentParser(description="Zero-shot LLM agent")
    parser.add_argument('--data-dir', required=True, help="Path to /home/data")
    parser.add_argument('--submission-dir', required=True, help="Path to /home/submission")
    parser.add_argument('--code-dir', required=True, help="Path to /home/code")
    parser.add_argument('--logs-dir', required=True, help="Path to /home/logs")
    parser.add_argument('--model', required=True, help="Model name via OpenRouter")
    parser.add_argument('--temperature', type=float, default=1.0)
    return parser.parse_args()

def get_llm_response(client, model, prompt, temperature):
    """Get LLM response - same as original"""
    response_kwargs = {
        "model": model,
        "messages": [{"role": "user", "content": prompt}],
        "temperature": temperature
    }
    response = client.chat.completions.create(**response_kwargs)
    return response.choices[0].message.content

def extract_scripts(response_text):
    """Extract scripts - same as original"""
    python_blocks = re.findall(r'```python\s+(.*?)```', response_text, re.DOTALL)
    if len(python_blocks) != 2:
        raise ValueError("Expected two Python code blocks in the response")
    train_script = python_blocks[0]
    inference_script = python_blocks[1]

    yaml_blocks = re.findall(r'```(?:yaml|yml)\s+(.*?)```', response_text, re.DOTALL)
    if len(yaml_blocks) != 1:
        raise ValueError("Expected one YAML code block in the response")
    env_yaml = yaml_blocks[0]

    return train_script, inference_script, env_yaml

def save_scripts(train_script, inference_script, env_yaml, output_dir, run_name):
    """Save scripts - same as original"""
    os.makedirs(output_dir, exist_ok=True)
    train_path = os.path.join(output_dir, "train.py")
    inference_path = os.path.join(output_dir, "inference.py")
    env_yaml_path = os.path.join(output_dir, "environment.yaml")

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

def run_script(script_path, script_type, output_path, run_name, test_csv_no_labels_path=None):
    """Run script - same as original"""
    if script_type == 'inference':
        cmd = f"source activate {run_name}_env && python {script_path} --input {test_csv_no_labels_path} --output {output_path}"
    if script_type == 'train':
        cmd = f"source activate {run_name}_env && python {script_path}"

    print(f"Executing command: {cmd}")
    process = subprocess.run(cmd, capture_output=True, text=True, shell=True, executable="/bin/bash")
    print(f"STDOUT: {process.stdout}")
    print(f"STDERR: {process.stderr}")
    return process.returncode

def delete_conda_env(run_name):
    """Delete conda environment - same as original"""
    env_name = run_name + '_env'
    subprocess.run(f"conda env remove -n {env_name} -y", shell=True, check=True)

def generate_and_run_scripts(client, model, data_dir, work_dir, run_name, temperature, test_features_path, submission_path, submission_dir):
    """Main function - adapted from original"""

    # Load dataset description
    desc_path = Path(data_dir) / "description.md"
    dataset_knowledge = desc_path.read_text() if desc_path.exists() else "No description available"

    # Get train CSV path
    train_csv_path = str(Path(data_dir) / "train.csv")

    # Infer dataset hints from data structure
    import pandas as pd
    train_df = pd.read_csv(train_csv_path)
    columns = list(train_df.columns)

    # Build dataset hints (simplified - adapt based on actual data)
    dataset_hints = f"""
        - Format: CSV with columns {columns}
        - Training samples: {len(train_df)}
        """

    # Use EXACT prompt from original
    prompt = f"""
        Create the best possible classifier that will generalize to new unseen data.
        You are using a linux system.
        You have access to both CPU and GPU resources (1 cuda device).

        DATASET:
        - Training file: {train_csv_path}
        {dataset_hints}

        Dataset knowledge:
        {dataset_knowledge}

        REQUIREMENTS:
        1. Create three files:
           - train.py
           - inference.py
           - environment.yaml

        2. For train.py:
        - Train a robust model suitable for the given dataset
        - Save the trained model to: {submission_dir}/model.pkl using joblib or pickle
        - Save all model artifacts to {submission_dir}/
        - Split the train file to train and validation to optimize during training.

        3. For inference.py:
        - Accept arguments: --input and --output
        - Load the model from: {submission_dir}/model.pkl
        - Output a CSV with all original input columns and 'numeric_label' column containing a score from 0 to 1

        4. For environment.yaml:
        - Create a conda environment file with all necessary packages
        - Include all libraries used in both train.py and inference.py
        - Pin all package versions (e.g., numpy<2, pytorch=2.1.*, scikit-learn>=1.3) to ensure compatibility

        OUTPUT FORMAT (CRITICAL):
        Provide EXACTLY 2 Python code blocks followed by EXACTLY 1 YAML code block in this order:

        # train.py
        ```python
        [train.py code here]
        ```

        # inference.py
        ```python
        [inference.py code here]
        ```

        # environment.yaml
        ```yaml
        [environment.yaml content here]
        ```

        Do not include any other code blocks. Use exactly this format.
        """

    response_content = get_llm_response(client, model, prompt, temperature)
    print(response_content)

    try:
        train_script, inference_script, env_yaml = extract_scripts(response_content)
    except Exception as e:
        print(f"Failed to extract scripts: {e}")
        return 1

    train_path, inference_path, env_yaml_path = save_scripts(
        train_script, inference_script, env_yaml, work_dir, run_name
    )

    # Create conda environment
    env_result = subprocess.run(
        f"conda env create -f {env_yaml_path}",
        shell=True, capture_output=True, text=True
    )
    if env_result.returncode != 0:
        print(f"Error creating conda environment: {env_result.stderr}")
        return 1

    # Run training
    error_code = run_script(
        script_path=train_path,
        script_type='train',
        output_path=None,
        run_name=run_name
    )
    if error_code != 0:
        delete_conda_env(run_name=run_name)
        return 1

    # Run inference
    error_code = run_script(
        script_path=inference_path,
        script_type='inference',
        output_path=submission_path,
        run_name=run_name,
        test_csv_no_labels_path=test_features_path
    )
    if error_code != 0:
        delete_conda_env(run_name=run_name)
        return 1


    delete_conda_env(run_name=run_name)
    return 0


def main():
    args = parse_args()

    client = OpenAI(
        base_url=os.getenv("OPENAI_BASE_URL"),
        api_key=os.getenv("OPENAI_API_KEY")
    )

    # Paths
    data_dir = args.data_dir
    submission_dir = args.submission_dir
    submission_path = os.path.join(submission_dir, "submission.csv")
    work_dir = submission_dir  # Use submission dir as workspace so all files end up there
    test_features_path = os.path.join(data_dir, "test_features.csv")

    run_name = "zeroshot"

    # Generate and run scripts
    result = generate_and_run_scripts(
        client=client,
        model=args.model,
        data_dir=data_dir,
        work_dir=work_dir,
        run_name=run_name,
        temperature=args.temperature,
        test_features_path=test_features_path,
        submission_path=submission_path,
        submission_dir=submission_dir
    )

    if result != 0:
        print("Zero-shot agent failed")
        sys.exit(1)

    print("✓ Zero-shot agent completed successfully!")

if __name__ == "__main__":
    main()
