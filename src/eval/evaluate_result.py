#!/usr/bin/env python3
"""
Evaluate classification metrics for a model's predictions.

This script computes the following metrics:
  - Average Precision Score (AUPRC): Area under the Precision-Recall curve.
  - ROC AUC Score (AUROC): Area under the Receiver Operating Characteristic curve.

Usage:
    python evaluate_result.py --results <results_file> --test <test_file> --output <output_file>
           [--pred-col <predicted_column_name>] [--class-col <true_label_column_name>]

Arguments:
    -r, --results      Path to the results CSV file. This file must include a column containing predicted probability scores.
                        Default prediction column is 'prediction', changeable via --pred-col.
    -t, --test         Path to the test CSV file. This file must include a column containing the true binary labels.
                        Default true label column is 'class', changeable via --class-col.
    -o, --output       Path to the output file where the computed metrics will be saved.
    --pred-col         (Optional) Name of the prediction column in the results file (default: 'prediction').
    --class-col        (Optional) Name of the class/label column in the test file (default: 'class').
"""

import argparse
import sys
import pandas as pd
from sklearn.metrics import average_precision_score, roc_auc_score, accuracy_score
from run_logging.logging_helpers import log_inference_stage_and_metrics

def evaluate_log_metrics(results_file, test_file, label_to_scalar, output_file=None, 
                    pred_col="prediction", class_col="class"):
    """
    Evaluate classification metrics for a model's predictions.
    
    Args:
        results_file (str): Path to the results CSV file with predictions.
        test_file (str): Path to the test CSV file with true labels.
        output_file (str, optional): Path to save the metrics. If None, metrics are not saved to a file.
        pred_col (str): Name of the prediction column in the results file.
        class_col (str): Name of the class/label column in the test file.
        
    Returns:
        dict: Dictionary containing the computed metrics (AUPRC and AUROC).
    """
    # Load the results and test files
    try:
        results = pd.read_csv(results_file)
    except Exception as e:
        raise ValueError(f"Error reading results file '{results_file}': {e}")

    try:
        test = pd.read_csv(test_file)
    except Exception as e:
        raise ValueError(f"Error reading test file '{test_file}': {e}")

    # Check if the required columns exist in the respective files
    if pred_col not in results.columns:
        raise ValueError(f"The prediction column '{pred_col}' was not found in the results file. "
                        f"Available columns: {results.columns.tolist()}")
    if class_col not in test.columns:
        raise ValueError(f"The class column '{class_col}' was not found in the test file. "
                        f"Available columns: {test.columns.tolist()}")

    # Merge the results and test files on the index
    merged = pd.merge(results, test, left_index=True, right_index=True)
    merged['class_numeric'] = merged[class_col].map(lambda x: int(label_to_scalar[x]))
    merged['prediction_numeric'] = merged[pred_col].map(lambda x: int(label_to_scalar[x]))

    # Calculate metrics using the specified columns
    try:
        auprc = average_precision_score(merged['class_numeric'], merged['prediction_numeric'])
    except Exception as e:
        raise ValueError(f"Error calculating AUPRC: {e}")

    try:
        auroc = roc_auc_score(merged['class_numeric'], merged['prediction_numeric'])
    except Exception as e:
        raise ValueError(f"Error calculating AUROC: {e}")
    
    try:
        accuracy = accuracy_score(merged['class_numeric'], merged['prediction_numeric'])
    except Exception as e:
        raise ValueError(f"Error calculating accuracy: {e}")

    # Create metrics dictionary
    metrics = {
        "AUPRC": auprc,
        "AUROC": auroc,
        "ACC": accuracy,
    }

    # Log the metrics
    log_inference_stage_and_metrics(2, metrics=metrics)

    # Save the results to the output file if specified
    if output_file:
        try:
            with open(output_file, "w") as f:
                for key, value in metrics.items():
                    f.write(f"{key}: {value}\n")
        except Exception as e:
            raise ValueError(f"Error writing to output file '{output_file}': {e}")

    return metrics

def main():
    parser = argparse.ArgumentParser(description="Evaluate classification metrics for model predictions.")
    parser.add_argument("-r", "--results", required=True, help="Path to the results CSV file.")
    parser.add_argument("-t", "--test", required=True, help="Path to the test CSV file.")
    parser.add_argument("-o", "--output", required=True, help="Path to the output file.")
    parser.add_argument("--pred-col", default="prediction", help="Name of the prediction column in the results file.")
    parser.add_argument("--class-col", default="class", help="Name of the class column in the test file.")
    
    args = parser.parse_args()
    
    try:
        evaluate_log_metrics(
            results_file=args.results,
            test_file=args.test,
            output_file=args.output,
            pred_col=args.pred_col,
            class_col=args.class_col
        )
    except Exception as e:
        sys.exit(f"Error: {e}")

if __name__ == '__main__':
    main()