import argparse
import sys
import pandas as pd
import numpy as np
from utils.metrics import get_classification_metrics_functions, get_regression_metrics_functions
import os

def get_metrics(results_file, test_file, task_type, output_file=None, numeric_label_col="numeric_label", 
                    pred_col="prediction", prob_col_prefix = 'probability_', delete_preds=False):
    
    results = pd.read_csv(results_file)
    test = pd.read_csv(test_file)
    merged = pd.merge(results, test, left_index=True, right_index=True)
    metrics = {}
    
    if task_type == "classification":
        # For classification, ensure predictions and labels are integers
        merged[pred_col] = merged[pred_col].astype(int)
        merged[numeric_label_col] = merged[numeric_label_col].astype(int)
        # Extract probability columns if they exist
        prob_cols = [col for col in results.columns if col.startswith(prob_col_prefix)]
        assert len(prob_cols) > 0, f"No probability columns found with prefix '{prob_col_prefix}'."
        # Sort probability columns numerically
        prob_cols_sorted = sorted(prob_cols, key=lambda x: int(x.split('_')[1]))

        metric_to_fn = get_classification_metrics_functions()
        for metric_name, metric_fn in metric_to_fn.items():
            if metric_fn.needs_probabilities:
                y_prob = merged[prob_cols_sorted].astype(float).values
                metrics[metric_name] = metric_fn(merged[numeric_label_col], y_prob)
            else:
                # For metrics that need class predictions (like ACC), use the prediction column
                metrics[metric_name] = metric_fn(merged[numeric_label_col], merged[pred_col])

    if task_type == "regression":
        # For regression, ensure predictions and labels are floats
        merged[numeric_label_col] = merged[numeric_label_col].astype(float)
        merged[pred_col] = merged[pred_col].astype(float)

        metric_to_fn = get_regression_metrics_functions()
        for metric_name, metric_fn in metric_to_fn.items():
            metrics[metric_name] = metric_fn(merged[numeric_label_col], merged[pred_col])

    # Save the results to the output file if specified
    if output_file:
        try:
            with open(output_file, "w") as f:
                for key, value in metrics.items():
                    f.write(f"{key}: {value}\n")
        except Exception as e:
            message = f"FAIL DURING WRITING METRICS TO A FILE {output_file}."
            raise Exception(message) from e
    
    if delete_preds:
        os.remove(results_file)

    return metrics

def main():
    parser = argparse.ArgumentParser(description="Evaluate classification metrics for model predictions.")
    parser.add_argument("-r", "--results", required=True, help="Path to the results CSV file.")
    parser.add_argument("-t", "--test", required=True, help="Path to the CSV file with labels.")
    parser.add_argument("-o", "--output", required=True, help="Path to the output file.")
    parser.add_argument("--pred-col", default="prediction", help="Name of the prediction column in the results file.")
    parser.add_argument("--numeric-label-col", default="numeric_label", help="Name of the numeric label column in the file.")
    parser.add_argument("--delete-preds", action="store_true", help="Delete the predictions file after evaluation.")
    parser.add_argument("--task-type", choices=["classification", "regression"], required=True, help="Type of task: classification or regression.")

    args = parser.parse_args()
    
    try:
        get_metrics(
            results_file=args.results,
            test_file=args.test,
            output_file=args.output,
            pred_col=args.pred_col,
            numeric_label_col=args.numeric_label_col,
            delete_preds=args.delete_preds,
            task_type=args.task_type
        )
    except Exception as e:
        sys.exit(f"Error: {e}")

if __name__ == '__main__':
    main()