#!/usr/bin/env python3
"""
Evaluate classification metrics for a model's predictions.

This script computes the following metrics:
  - Average Precision Score (AUPRC): Area under the Precision-Recall curve.
  - ROC AUC Score (AUROC): Area under the Receiver Operating Characteristic curve.

Usage:
    python evaluate_result.py --results <results_file> --test <test_file> --output <output_file> --model-name <model_name>
           [--pred-col <predicted_column_name>] [--class-col <true_label_column_name>]

Arguments:
    -r, --results      Path to the results CSV file. This file must include a column containing predicted probability scores.
                        Default prediction column is 'prediction', changeable via --pred-col.
    -t, --test         Path to the test CSV file. This file must include a column containing the true binary labels.
                        Default true label column is 'class', changeable via --class-col.
    -o, --output       Path to the output file where the computed metrics will be saved.
    -m, --model-name   The name of the model being evaluated.
    --pred-col         (Optional) Name of the prediction column in the results file (default: 'prediction').
    --class-col        (Optional) Name of the class/label column in the test file (default: 'class').
"""

import argparse
import sys
import pandas as pd
from sklearn.metrics import average_precision_score, roc_auc_score

def main():
    # Parse the arguments
    parser = argparse.ArgumentParser(
        description="Evaluate model predictions using AUPRC and AUROC metrics."
    )
    parser.add_argument("-r", "--results",
                        help="Path to the results CSV file (must contain prediction column).",
                        required=True)
    parser.add_argument("-t", "--test",
                        help="Path to the test CSV file (must contain class column with true labels).",
                        required=True)
    parser.add_argument("-o", "--output",
                        help="Path to the output file where metrics will be saved.",
                        required=True)
    parser.add_argument("-m", "--model-name",
                        help="Name of the model being evaluated.",
                        required=True)
    parser.add_argument("--pred-col",
                        help="Name of the prediction column in the results file (default: 'prediction').",
                        default="prediction")
    parser.add_argument("--class-col",
                        help="Name of the class/label column in the test file (default: 'class').",
                        default="class")

    args = parser.parse_args()

    # Load the results and test files
    try:
        results = pd.read_csv(args.results)
    except Exception as e:
        sys.exit(f"Error reading results file '{args.results}': {e}")

    try:
        test = pd.read_csv(args.test)
    except Exception as e:
        sys.exit(f"Error reading test file '{args.test}': {e}")

    # Check if the required columns exist in the respective files
    if args.pred_col not in results.columns:
        sys.exit(f"Error: The prediction column '{args.pred_col}' was not found in the results file '{args.results}'. "
                 f"Available columns: {results.columns.tolist()}")
    if args.class_col not in test.columns:
        sys.exit(f"Error: The class column '{args.class_col}' was not found in the test file '{args.test}'. "
                 f"Available columns: {test.columns.tolist()}")

    # Merge the results and test files on the index
    merged = pd.merge(results, test, left_index=True, right_index=True)

    # Calculate metrics using the specified columns
    try:
        auprc = average_precision_score(merged[args.class_col], merged[args.pred_col])
    except Exception as e:
        sys.exit(f"Error calculating AUPRC: {e}")

    try:
        auroc = roc_auc_score(merged[args.class_col], merged[args.pred_col])
    except Exception as e:
        sys.exit(f"Error calculating AUROC: {e}")

    # Print the computed metrics
    print(f"Model: {args.model_name}")
    print(f"AUPRC: {auprc}")
    print(f"AUROC: {auroc}")

    # Save the results to the output file
    try:
        with open(args.output, "w") as f:
            f.write(f"Model: {args.model_name}\n")
            f.write(f"AUPRC: {auprc}\n")
            f.write(f"AUROC: {auroc}\n")
    except Exception as e:
        sys.exit(f"Error writing to output file '{args.output}': {e}")

if __name__ == '__main__':
    main()
