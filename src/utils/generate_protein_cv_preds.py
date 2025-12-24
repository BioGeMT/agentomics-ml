"""
Protein dataset cross-validation evaluation script for stealth test.
Handles retraining and evaluation with cross-validation folds similar to run_agent_biomlbench.py
"""
import os
import sys
import pandas as pd
import subprocess
import shutil
import tempfile
from pathlib import Path
import argparse


def run_command(command, description=""):
    """Run a shell command and return the result."""
    print(f"Running: {description}")
    print(f"Command: {command}")
    result = subprocess.run(command, shell=True, executable="/bin/bash", capture_output=True)
    if result.returncode != 0:
        print(f"Error during {description}:")
        print(result.stderr.decode())
        print(result.stdout.decode())
        return False
    print(f"Success: {description}")
    return True


def protein_cv_retrain_and_evaluate(
    iteration_dir,
    train_csv_path,
    output_csv_path,
    agent_name=None
):
    """
    Retrain a protein model using cross-validation and generate predictions.

    Args:
        iteration_dir: Path to the iteration directory containing train.py and inference.py
        train_csv_path: Path to the original training CSV with fold columns
        output_csv_path: Path where final predictions CSV should be saved
        agent_name: Name for the conda environment (optional, derived from iteration_dir if not provided)

    Returns:
        bool: True if successful, False otherwise
    """
    iteration_dir = Path(iteration_dir)
    train_script_path = iteration_dir / "train.py"
    inference_script_path = iteration_dir / "inference.py"

    if not train_script_path.exists():
        print(f"Error: train.py not found at {train_script_path}")
        return False
    if not inference_script_path.exists():
        print(f"Error: inference.py not found at {inference_script_path}")
        return False

    # Determine conda env path (conda env should already be created by compute_stealth_test.sh)
    if agent_name is None:
        agent_name = iteration_dir.name

    # Use parent directory for conda env (shared across iterations)
    agent_dir = iteration_dir.parent
    conda_env_path = agent_dir / ".conda" / "envs" / f"{agent_name}_env"

    if not conda_env_path.exists():
        print(f"Error: Conda environment not found at {conda_env_path}")
        print("Conda environment should be created by compute_stealth_test.sh before calling this script")
        return False

    print(f"Using conda environment at {conda_env_path}")

    # Read original training data with fold columns
    og_train_data = pd.read_csv(train_csv_path)

    # Filter out rows with fold_random_5 == -1 (test set rows)
    if 'fold_random_5' in og_train_data.columns:
        og_train_data = og_train_data[og_train_data['fold_random_5'] != -1]

    # Determine the label column name (either 'fitness_score' or 'numeric_label')
    if 'fitness_score' in og_train_data.columns:
        label_col = 'fitness_score'
    elif 'numeric_label' in og_train_data.columns:
        label_col = 'numeric_label'
    else:
        print(f"Error: Neither 'fitness_score' nor 'numeric_label' found in columns: {og_train_data.columns.tolist()}")
        return False

    # Columns to keep for training/validation/test
    cols_to_keep = ['id', 'sequence', label_col]

    # Ensure all required columns exist
    if not all(col in og_train_data.columns for col in cols_to_keep):
        print(f"Error: Missing required columns. Expected {cols_to_keep}, got {og_train_data.columns.tolist()}")
        return False

    # Find all fold columns
    fold_col_types = [col for col in og_train_data.columns if col.startswith('fold_')]
    if not fold_col_types:
        print("Error: No fold columns found in training data")
        return False

    print(f"Found fold columns: {fold_col_types}")

    # Prepare command prefix
    command_prefix = f"conda run -p {conda_env_path} --no-capture-output"

    # Create temporary directory for intermediate files
    with tempfile.TemporaryDirectory() as temp_dir:
        temp_dir = Path(temp_dir)
        fold_col_to_preds = []

        # Process each fold column type
        for fold_col in fold_col_types:
            fold_predictions_dfs = []
            unique_folds = og_train_data[fold_col].unique()
            num_folds = len(unique_folds)

            print(f"\nProcessing fold column: {fold_col} with {num_folds} folds")

            # Train and predict for each fold
            for current_test_fold_value in unique_folds:
                print(f"  Processing test fold {current_test_fold_value}/{num_folds-1}")

                # Pick validation fold (next fold in rotation)
                validation_fold_value = (current_test_fold_value + 1) % num_folds

                # Split data
                test_df = og_train_data[og_train_data[fold_col] == current_test_fold_value][cols_to_keep].copy()
                valid_df = og_train_data[og_train_data[fold_col] == validation_fold_value][cols_to_keep].copy()
                train_df = og_train_data[
                    (og_train_data[fold_col] != current_test_fold_value) &
                    (og_train_data[fold_col] != validation_fold_value)
                ][cols_to_keep].copy()

                # Rename columns to numeric_label if needed
                if label_col != 'numeric_label':
                    test_df = test_df.rename(columns={label_col: 'numeric_label'})
                    valid_df = valid_df.rename(columns={label_col: 'numeric_label'})
                    train_df = train_df.rename(columns={label_col: 'numeric_label'})

                # Validate no intersection
                train_ids = set(train_df['id'])
                valid_ids = set(valid_df['id'])
                test_ids = set(test_df['id'])
                assert len(train_ids & valid_ids) == 0, "Train/valid intersection"
                assert len(train_ids & test_ids) == 0, "Train/test intersection"
                assert len(valid_ids & test_ids) == 0, "Valid/test intersection"

                # Save temporary CSV files
                fold_prefix = f"{fold_col}_{current_test_fold_value}"
                train_csv = temp_dir / f"{fold_prefix}_train.csv"
                valid_csv = temp_dir / f"{fold_prefix}_valid.csv"
                test_csv = temp_dir / f"{fold_prefix}_test.csv"
                artifacts_dir = temp_dir / f"{fold_prefix}_artifacts"
                predictions_csv = temp_dir / f"{fold_prefix}_predictions.csv"

                train_df.to_csv(train_csv, index=False)
                valid_df.to_csv(valid_csv, index=False)
                test_df.to_csv(test_csv, index=False)
                artifacts_dir.mkdir(exist_ok=True)

                # Training command
                train_cmd_dir = f"cd {iteration_dir} && "
                train_cmd = f'{train_cmd_dir} {command_prefix} python "{train_script_path}" --train-data "{train_csv}" --validation-data "{valid_csv}" --artifacts-dir "{artifacts_dir}"'

                if not run_command(train_cmd, f"Training {fold_prefix}"):
                    print(f"Warning: Training failed for {fold_prefix}")
                    continue

                # Inference command
                inference_cmd_dir = f"cd {iteration_dir} && "
                inference_cmd = f'{inference_cmd_dir} {command_prefix} python "{inference_script_path}" --input "{test_csv}" --output "{predictions_csv}" --artifacts-dir "{artifacts_dir}"'

                if not run_command(inference_cmd, f"Inference {fold_prefix}"):
                    print(f"Warning: Inference failed for {fold_prefix}")
                    continue

                # Read predictions
                if not predictions_csv.exists():
                    print(f"Warning: Predictions file not created for {fold_prefix}")
                    continue

                preds_df = pd.read_csv(predictions_csv)
                fold_predictions_dfs.append(preds_df)

            # Concatenate all fold predictions
            if fold_predictions_dfs:
                fold_preds = pd.concat(fold_predictions_dfs, ignore_index=True)

                # Rename prediction column
                if 'prediction' in fold_preds.columns:
                    if len(fold_col_types) == 1:
                        fold_preds = fold_preds.rename(columns={'prediction': 'fitness_score'})
                    else:
                        fold_preds = fold_preds.rename(columns={'prediction': f'fitness_score_{fold_col}'})

                fold_col_to_preds.append(fold_preds)
            else:
                print(f"Warning: No predictions generated for {fold_col}")
                return False

        # Combine all predictions
        if not fold_col_to_preds:
            print("Error: No predictions were generated")
            return False

        final_predictions_df = fold_col_to_preds[0]
        for fold_preds_df in fold_col_to_preds[1:]:
            final_predictions_df = final_predictions_df.merge(fold_preds_df, on='id', how='inner')

        # Save final predictions
        final_predictions_df.to_csv(output_csv_path, index=False)
        print(f"\nFinal predictions saved to {output_csv_path}")
        print(f"Total predictions: {len(final_predictions_df)}")

    return True


def main():
    parser = argparse.ArgumentParser(
        description="Retrain and evaluate protein models with cross-validation"
    )
    parser.add_argument('--iteration-dir', required=True, help='Path to iteration directory with train.py and inference.py')
    parser.add_argument('--train-csv', required=True, help='Path to training CSV with fold columns')
    parser.add_argument('--output-csv', required=True, help='Path for output predictions CSV')
    parser.add_argument('--agent-name', help='Name for conda environment (optional, defaults to iteration dir name)')

    args = parser.parse_args()

    success = protein_cv_retrain_and_evaluate(
        iteration_dir=args.iteration_dir,
        train_csv_path=args.train_csv,
        output_csv_path=args.output_csv,
        agent_name=args.agent_name
    )

    sys.exit(0 if success else 1)


if __name__ == "__main__":
    main()
