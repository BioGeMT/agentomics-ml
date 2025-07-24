import argparse
import sys
import pandas as pd
from sklearn.metrics import average_precision_score, roc_auc_score, accuracy_score, mean_squared_error
import os

def get_metrics(results_file, test_file, task_type, output_file=None, numeric_label_col="numeric_label", 
                    pred_col="prediction", acc_threshold=0.5, delete_preds=False):
    
    results = pd.read_csv(results_file)
    test = pd.read_csv(test_file)

    merged = pd.merge(results, test, left_index=True, right_index=True)

    merged[numeric_label_col] = merged[numeric_label_col].astype(float)
    merged[pred_col] = merged[pred_col].astype(float)

    if task_type == "classification":
        auprc = average_precision_score(merged[numeric_label_col], merged[pred_col])
        auroc = roc_auc_score(merged[numeric_label_col], merged[pred_col])
        accuracy = accuracy_score(merged[numeric_label_col], (merged[pred_col] >= acc_threshold).astype(int))

        metrics = {
            "AUPRC": auprc,
            "AUROC": auroc,
            "ACC": accuracy,
        }
    else:
        mse = mean_squared_error(merged[numeric_label_col], merged[pred_col])
        metrics = {
            "MSE": mse,
        }

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