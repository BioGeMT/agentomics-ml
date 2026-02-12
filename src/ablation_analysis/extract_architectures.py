#!/usr/bin/env python3
"""Extract model architectures from ablation study run reports.

For runs with model_architecture step: parses the [MODEL ARCHITECTURE] section from reports.
For runs that skipped model_architecture: fetches the training script from W&B artifacts
(using the filename parsed from [MODEL TRAINING] path_to_train_file in each report).
For runs that skipped all steps: fetches all *train*.py files from W&B artifacts.
"""

import os
import re
import fnmatch
import wandb
import pandas as pd
from . import config
from .list_runs import list_runs


DEFAULT_RESULTS_DIR = "remote_outputs"

# Configs that need artifact-based extraction
NEEDS_ARTIFACT_SINGLE = "no_model_architecture"
NEEDS_ARTIFACT_ALL = "no_all_steps"


def find_report_files(agent_id, results_dir):
    """Find all report files for a given agent run.

    Returns list of (iteration, filepath) tuples sorted by iteration.
    """
    reports_dir = os.path.join(results_dir, agent_id, "reports", agent_id)
    if not os.path.isdir(reports_dir):
        return []

    results = []
    for fname in os.listdir(reports_dir):
        match = re.search(r'run_report_iter_(\d+)', fname)
        if match and fname.endswith('.txt'):
            iteration = int(match.group(1))
            results.append((iteration, os.path.join(reports_dir, fname)))

    return sorted(results, key=lambda x: x[0])


def extract_architecture_section(report_path):
    """Extract the full [MODEL ARCHITECTURE] section from a report file.

    Returns the text between [MODEL ARCHITECTURE] and the next section
    ([MODEL TRAINING] or [FINAL_OUTCOME]), or None if not found.
    """
    try:
        with open(report_path, 'r') as f:
            content = f.read()
    except Exception as e:
        print(f"    Error reading {report_path}: {e}")
        return None

    pattern = r'\[MODEL ARCHITECTURE\]\n(.*?)(?=\n\[MODEL TRAINING\]|\n\[FINAL_OUTCOME\]|\Z)'
    match = re.search(pattern, content, re.DOTALL)

    if not match:
        return None

    return match.group(1).strip()


def extract_train_filename(report_path):
    """Extract the training file name from [MODEL TRAINING] path_to_train_file in a report."""
    try:
        with open(report_path, 'r') as f:
            content = f.read()
    except Exception as e:
        print(f"    Error reading {report_path}: {e}")
        return None

    match = re.search(r'path_to_train_file:\s*(\S+)', content)
    if match:
        return os.path.basename(match.group(1))
    return None


def _get_artifact(agent_id, iteration, api):
    """Download a W&B code artifact and return the local directory, or None."""
    artifact_path = f"{config.WANDB_ENTITY}/{config.WANDB_PROJECT}/{agent_id}_iteration_{iteration}:latest"
    try:
        artifact = api.artifact(artifact_path, type='code')
        return artifact.download()
    except wandb.errors.CommError:
        print(f"    Artifact not found: {artifact_path}")
        return None
    except Exception as e:
        print(f"    Error fetching artifact: {e}")
        return None


def fetch_train_file_from_artifact(agent_id, iteration, train_filename, api):
    """Fetch a specific training file from a W&B code artifact."""
    artifact_dir = _get_artifact(agent_id, iteration, api)
    if not artifact_dir:
        return None

    train_file_path = os.path.join(artifact_dir, train_filename)
    if os.path.isfile(train_file_path):
        with open(train_file_path, 'r', errors='replace') as f:
            return f.read()
    else:
        print(f"    {train_filename} not found in artifact (has: {os.listdir(artifact_dir)})")
        return None


def fetch_all_train_files_from_artifact(agent_id, iteration, api):
    """Fetch all *train*.py files from a W&B code artifact, concatenated.

    Returns the concatenated content with file separators, or None if no train files found.
    """
    artifact_dir = _get_artifact(agent_id, iteration, api)
    if not artifact_dir:
        return None

    # Find all files matching *train*.py
    train_files = sorted(
        f for f in os.listdir(artifact_dir)
        if fnmatch.fnmatch(f.lower(), '*train*.py')
    )

    if not train_files:
        print(f"    No *train*.py files in artifact (has: {os.listdir(artifact_dir)})")
        return None

    # Concatenate all train files with separators
    parts = []
    for fname in train_files:
        fpath = os.path.join(artifact_dir, fname)
        with open(fpath, 'r', errors='replace') as f:
            code = f.read()
        parts.append(f"# === {fname} ===\n{code}")

    return "\n\n".join(parts)


def _init_wandb_api():
    """Initialize and return W&B API client."""
    api_key = os.getenv("WANDB_API_KEY")
    if api_key:
        wandb.login(key=api_key, relogin=True)
    return wandb.Api()


def extract_architectures(results_dir=DEFAULT_RESULTS_DIR, output_file=None):
    """Extract architectures from all ablation runs.

    - Runs with arch step: extracts [MODEL ARCHITECTURE] section from reports
    - no_model_architecture: fetches specific training file from W&B artifacts
    - no_all_steps: fetches all *train*.py files from W&B artifacts

    Returns a DataFrame with columns: agent_id, ablation_config, model, dataset, iteration, architecture.
    """
    runs = list_runs()

    if not runs:
        print("\nNo runs found.")
        return None

    if not os.path.isdir(results_dir):
        print(f"\nError: results directory not found: {results_dir}")
        return None

    # Split runs into three groups
    runs_with_arch = []
    runs_no_arch = []
    runs_no_all = []

    for run_info in runs:
        if run_info['ablation_config'] == NEEDS_ARTIFACT_ALL:
            runs_no_all.append(run_info)
        elif run_info['ablation_config'] == NEEDS_ARTIFACT_SINGLE:
            runs_no_arch.append(run_info)
        else:
            runs_with_arch.append(run_info)

    print(f"\n  Runs with arch step: {len(runs_with_arch)}")
    print(f"  Runs without arch step (no_model_architecture): {len(runs_no_arch)}")
    print(f"  Runs without any steps (no_all_steps): {len(runs_no_all)}")

    rows = []
    skipped_no_reports = 0

    # --- Phase 1: Extract from reports (runs with model_architecture step) ---
    print(f"\n--- Extracting [MODEL ARCHITECTURE] from reports ---\n")

    for run_info in sorted(runs_with_arch, key=lambda r: r['name']):
        agent_id = run_info['name']

        report_files = find_report_files(agent_id, results_dir)
        if not report_files:
            skipped_no_reports += 1
            continue

        iterations_found = 0
        for iteration, report_path in report_files:
            architecture = extract_architecture_section(report_path)
            if architecture is None:
                continue

            rows.append({
                'agent_id': agent_id,
                'ablation_config': run_info['ablation_config'],
                'model': run_info['model'],
                'dataset': run_info['dataset'],
                'iteration': iteration,
                'architecture': architecture,
            })
            iterations_found += 1

        if iterations_found:
            print(f"  {agent_id}: {iterations_found} iterations")
        else:
            print(f"  {agent_id}: no [MODEL ARCHITECTURE] section found")

    report_rows = len(rows)

    # --- Phase 2: Fetch from W&B artifacts (no_model_architecture) ---
    needs_artifacts = runs_no_arch or runs_no_all
    api = _init_wandb_api() if needs_artifacts else None

    if runs_no_arch:
        print(f"\n--- Fetching training scripts from W&B artifacts (no_model_architecture) ---\n")

        for run_info in sorted(runs_no_arch, key=lambda r: r['name']):
            agent_id = run_info['name']

            report_files = find_report_files(agent_id, results_dir)
            if not report_files:
                skipped_no_reports += 1
                continue

            iterations_found = 0
            for iteration, report_path in report_files:
                train_filename = extract_train_filename(report_path)
                if not train_filename:
                    print(f"  {agent_id} iter {iteration}: no path_to_train_file in report")
                    continue

                train_code = fetch_train_file_from_artifact(agent_id, iteration, train_filename, api)
                if not train_code:
                    continue

                rows.append({
                    'agent_id': agent_id,
                    'ablation_config': run_info['ablation_config'],
                    'model': run_info['model'],
                    'dataset': run_info['dataset'],
                    'iteration': iteration,
                    'architecture': train_code,
                })
                iterations_found += 1

            if iterations_found:
                print(f"  {agent_id}: {iterations_found} iterations (from artifacts)")
            else:
                print(f"  {agent_id}: no training files found")

    no_arch_rows = len(rows) - report_rows

    # --- Phase 3: Fetch from W&B artifacts (no_all_steps) ---
    if runs_no_all:
        print(f"\n--- Fetching all *train*.py from W&B artifacts (no_all_steps) ---\n")

        for run_info in sorted(runs_no_all, key=lambda r: r['name']):
            agent_id = run_info['name']

            # Use report files to get iteration numbers
            report_files = find_report_files(agent_id, results_dir)
            if not report_files:
                skipped_no_reports += 1
                continue

            iterations_found = 0
            for iteration, _ in report_files:
                train_code = fetch_all_train_files_from_artifact(agent_id, iteration, api)
                if not train_code:
                    continue

                rows.append({
                    'agent_id': agent_id,
                    'ablation_config': run_info['ablation_config'],
                    'model': run_info['model'],
                    'dataset': run_info['dataset'],
                    'iteration': iteration,
                    'architecture': train_code,
                })
                iterations_found += 1

            if iterations_found:
                print(f"  {agent_id}: {iterations_found} iterations (all train files)")
            else:
                print(f"  {agent_id}: no training files found")

    no_all_rows = len(rows) - report_rows - no_arch_rows

    # --- Summary ---
    print(f"\n{'='*50}")
    print(f"Extracted {len(rows)} total architecture entries")
    print(f"  From reports: {report_rows}")
    print(f"  From artifacts (no_model_architecture): {no_arch_rows}")
    print(f"  From artifacts (no_all_steps): {no_all_rows}")
    print(f"  Skipped (no reports dir): {skipped_no_reports}")

    if not rows:
        print("\nNo architecture data found!")
        return None

    df = pd.DataFrame(rows)
    df = df.sort_values(['agent_id', 'iteration'])

    if output_file:
        os.makedirs(os.path.dirname(output_file) or ".", exist_ok=True)
        df.to_csv(output_file, index=False)
        print(f"\nSaved to {output_file}")
        print(f"  Total rows: {len(df)}")
        print(f"  Unique runs: {df['agent_id'].nunique()}")

    return df
