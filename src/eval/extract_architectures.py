#!/usr/bin/env python3
"""
Extract model architectures from ablation study run reports.

This script parses local report files to retrieve ModelArchitecture decisions
across all iterations for ablation study runs and outputs them to a CSV file.
"""

import pandas as pd
import os
import json
import re
import argparse
from typing import Dict, Any, Optional


def get_run_metadata(agent_id: str, path: str) -> Optional[Dict[str, Any]]:
    """Get metadata for a run from local config file"""
    config_path = f"{path}/{agent_id}/extras/config.json"

    if not os.path.exists(config_path):
        print(f"  Warning: Config not found at {config_path}")
        return None

    with open(config_path, 'r') as f:
        config = json.load(f)

    return {
        'model_name': config.get('model_name', 'unknown'),
        'steps_to_skip': config.get('steps_to_skip', [])
    }


def extract_architecture_from_report(report_path: str) -> Optional[str]:
    """
    Extract architecture description from a report file.

    Parses the [MODEL ARCHITECTURE] section and extracts the text
    after 'architecture:' until the next field.
    """
    try:
        with open(report_path, 'r') as f:
            content = f.read()

        # Find the [MODEL ARCHITECTURE] section
        arch_match = re.search(r'\[MODEL ARCHITECTURE\](.*?)(?:\[|$)', content, re.DOTALL)
        if not arch_match:
            return None

        arch_section = arch_match.group(1)

        # Extract the architecture field
        # Look for "architecture:" followed by text until "hyperparameters:" or end
        arch_field_match = re.search(r'architecture:\s*(.*?)(?:hyperparameters:|reasoning:|$)',
                                      arch_section, re.DOTALL)
        if not arch_field_match:
            return None

        # Clean up the extracted text
        architecture = arch_field_match.group(1).strip()

        return architecture

    except Exception as e:
        print(f"  Error reading {report_path}: {e}")
        return None


def get_iteration_from_filename(filename: str) -> Optional[int]:
    """Extract iteration number from report filename (e.g., run_report_iter_3.txt -> 3)"""
    match = re.search(r'run_report_iter_(\d+)', filename)
    if match:
        return int(match.group(1))
    return None


def main():
    """Main function to extract architectures and create CSV"""

    parser = argparse.ArgumentParser(description="Extract architectures from ablation study reports")
    parser.add_argument(
        '--ablation-results-dir',
        type=str,
        default='outputs/ablation_results',
        help='Path to the ablation results directory (default: outputs/ablation_results)'
    )
    parser.add_argument(
        '--output_file',
        type=str,
        default='outputs/ablation_architectures.csv',
    )
    args = parser.parse_args()

    # Check if ablation results directory exists
    ablation_results_dir = args.ablation_results_dir
    output_file = args.output_file
    if not os.path.exists(ablation_results_dir):
        print(f"Error: {ablation_results_dir} not found")
        return

    # Get all ablation study runs
    agent_ids = [d for d in os.listdir(ablation_results_dir)
                 if os.path.isdir(os.path.join(ablation_results_dir, d))]

    print(f"Found {len(agent_ids)} ablation runs: {agent_ids}\n")

    # Collect data for each run
    rows = []

    for agent_id in agent_ids:
        print(f"Processing {agent_id}...")

        # Get metadata
        metadata = get_run_metadata(agent_id, ablation_results_dir)
        if metadata is None:
            print(f"  Skipping {agent_id} - no config found")
            continue

        # Skip if model_architecture was skipped
        if 'model_architecture' in metadata['steps_to_skip']:
            print(f"  Skipping {agent_id} - model_architecture step was skipped")
            continue

        # Find report files
        reports_dir = f"outputs/ablation_results/{agent_id}/reports/{agent_id}"
        if not os.path.exists(reports_dir):
            print(f"  Warning: Reports directory not found at {reports_dir}")
            continue

        report_files = [f for f in os.listdir(reports_dir)
                       if f.startswith('run_report_iter_') and f.endswith('.txt')]

        if not report_files:
            print(f"  Warning: No report files found")
            continue

        # Process each report file
        iterations_found = 0
        for report_file in sorted(report_files):
            report_path = os.path.join(reports_dir, report_file)
            iteration = get_iteration_from_filename(report_file)

            if iteration is None:
                continue

            # Extract architecture
            architecture = extract_architecture_from_report(report_path)

            if architecture:
                # Prepare ablation configuration string
                ablation_config = ','.join(metadata['steps_to_skip']) if metadata['steps_to_skip'] else 'none'

                # Add row
                rows.append({
                    'agent_id': agent_id,
                    'ablation_configuration': ablation_config,
                    'model': metadata['model_name'],
                    'iteration': iteration,
                    'architecture': architecture
                })
                iterations_found += 1

        print(f"  ✓ Extracted {iterations_found} iterations")

    # Create DataFrame
    if not rows:
        print("\n⚠ No architecture data found!")
        return

    df = pd.DataFrame(rows)

    # Sort by agent_id and iteration
    df = df.sort_values(['agent_id', 'iteration'])

    # Save to CSV
    df.to_csv(output_file, index=False)

    print(f"\n✓ Saved architectures to {output_file}")
    print(f"  Total rows: {len(df)}")
    print(f"  Unique runs: {df['agent_id'].nunique()}")
    print(f"  Columns: {list(df.columns)}")
    print(f"\nFirst few rows:")
    print(df.head())


if __name__ == "__main__":
    main()
