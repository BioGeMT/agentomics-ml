import argparse
import sys
import pandas as pd
from sklearn.metrics import average_precision_score, roc_auc_score, accuracy_score
from run_logging.logging_helpers import log_inference_stage_and_metrics

def evaluate_log_metrics(results_file, test_file, label_to_scalar, output_file=None, 
                    pred_col="prediction", class_col="class", acc_threshold=0.5):
    
    results = pd.read_csv(results_file)
    test = pd.read_csv(test_file)

    merged = pd.merge(results, test, left_index=True, right_index=True)
    merged['class_numeric'] = merged[class_col].map(lambda x: int(label_to_scalar[x]))

    auprc = average_precision_score(merged['class_numeric'], merged[pred_col])
    auroc = roc_auc_score(merged['class_numeric'], merged[pred_col])
    accuracy = accuracy_score(merged['class_numeric'], (merged[pred_col] >= acc_threshold).astype(int))

    metrics = {
        "AUPRC": auprc,
        "AUROC": auroc,
        "ACC": accuracy,
    }

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