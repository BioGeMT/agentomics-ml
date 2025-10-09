import json
import subprocess
import traceback

from pydantic_ai import ModelRetry
from eval.evaluate_result import get_metrics
from run_logging.logging_helpers import log_inference_stage_and_metrics, log_serial_metrics

def run_inference_and_log(config, iteration, evaluation_stage, use_best_snapshot=False):
    with open(config.prepared_dataset_dir / "metadata.json") as f:
        dataset_metadata = json.load(f)

    run_dir = config.runs_dir / config.agent_id
    snapshot_dir = config.snapshots_dir / config.agent_id
    source_folder = 'snapshot' if use_best_snapshot else 'run'
    if use_best_snapshot:
        print('USING BEST SNAPSHOT')
    conda_path = {
        'run': run_dir / ".conda" / "envs" / f"{config.agent_id}_env",
        'snapshot': snapshot_dir / ".conda"/ "envs" / f"{config.agent_id}_env",
    }
    inference_path = {
        'run': run_dir / "inference.py",
        'snapshot': snapshot_dir / "inference.py",
    }
    # For validation and train sets, check if agent created them or if they were provided
    if config.explicit_valid_set_provided:
        validation_input = config.agent_dataset_dir / "validation.csv"
        train_input = config.agent_dataset_dir / "train.csv"
    else:
        validation_input = run_dir / "validation.csv"
        train_input = run_dir / "train.csv"

    stage_to_input = {
        'dry_run': config.prepared_dataset_dir / "train.no_label.csv",
        'validation': validation_input,
        'test': config.prepared_dataset_dir / "test.no_label.csv",
        'train': train_input
    }
    stage_to_output = {
        'dry_run': run_dir / "eval_predictions_dry_run.csv",
        'validation': run_dir / "eval_predictions_validation.csv",
        'test': run_dir / "eval_predictions_test.csv",
        'train': run_dir / "eval_predictions_train.csv"
    }
    stage_to_metrics_file = {
        'dry_run': run_dir / "dry_run_metrics.txt",
        'validation': run_dir / "validation_metrics.txt",
        'test': run_dir / "test_metrics.txt",
        'train': run_dir / "train_metrics.txt"
    }

    command_prefix=f"conda run -p {conda_path[source_folder]} --no-capture-output"
    command = f"{command_prefix} python {inference_path[source_folder]} --input {stage_to_input[evaluation_stage]} --output {stage_to_output[evaluation_stage]}"
    inference_out = subprocess.run(command, shell=True, executable="/bin/bash", capture_output=True)
    if evaluation_stage == 'test':
        test_file_path = config.prepared_dataset_dir / "test.csv"
        if not test_file_path.exists():
            print('TEST EVAL SKIPPED - NO TEST SET')
            return
        print('RUNNING TEST EVAL')
        if inference_out.returncode != 0:
            print('TEST EVAL FAIL', str(inference_out))
            log_inference_stage_and_metrics(1, task_type=config.task_type)
            raise Exception('Inference on TEST script failed:', str(inference_out))
        try:
            test_metrics = get_metrics(
                results_file=stage_to_output[evaluation_stage],
                test_file=test_file_path,
                output_file=stage_to_metrics_file[evaluation_stage],
                numeric_label_col=dataset_metadata['numeric_label_col'],
                delete_preds=True,
                task_type=dataset_metadata['task_type']
            )
            log_inference_stage_and_metrics(2, metrics=test_metrics, task_type=config.task_type)
        except Exception as e:
            print('TEST EVAL FAIL', {traceback.format_exc()})
            log_inference_stage_and_metrics(1, task_type=config.task_type)
            return
        print('TEST EVAL SUCCESS')
    if evaluation_stage == 'dry_run':
        print('RUNNING DRY RUN EVAL')
        if inference_out.returncode != 0:
            print('DRY RUN EVAL FAIL during inference:', inference_out.stderr)
            raise ModelRetry(f'Inference script validation failed: {str(inference_out)}')
        try:
            _ = get_metrics(
                results_file=stage_to_output[evaluation_stage],
                test_file=config.prepared_dataset_dir / "train.csv",
                output_file=stage_to_metrics_file[evaluation_stage],
                numeric_label_col=dataset_metadata['numeric_label_col'],
                delete_preds=True,
                task_type=dataset_metadata['task_type']
            )
        except Exception as e:
            message = f"FAIL DURING DRY RUN METRICS COMPUTATION. {traceback.format_exc()}"
            print(message)
            raise ModelRetry(message)
        print('DRY RUN EVAL SUCCESS')
    if evaluation_stage == 'validation':
        print('RUNNING VALIDATION EVAL')
        if inference_out.returncode != 0:
            print('VALIDATION EVAL FAIL during inference:', inference_out.stderr)
            log_serial_metrics(prefix=evaluation_stage, metrics=None, iteration=iteration, task_type=config.task_type)
            raise Exception('Inference script validation failed:', str(inference_out))
        try:
            _ = get_metrics_and_serial_log(
                results_file=stage_to_output[evaluation_stage],
                test_file=stage_to_input[evaluation_stage],
                output_file=stage_to_metrics_file[evaluation_stage],
                numeric_label_col=dataset_metadata['numeric_label_col'],
                iteration=iteration,
                prefix=evaluation_stage,
                delete_preds=True,
                task_type=dataset_metadata['task_type']
            )
        except Exception as e:
            log_serial_metrics(prefix="validation", metrics=None, iteration=iteration, task_type=config.task_type)
            message = "FAIL DURING VALIDATION METRICS COMPUTATION."
            raise Exception(message) from e
        print('VALIDATION EVAL SUCCESS')
    if evaluation_stage == 'train':
        print('RUNNING TRAIN EVAL')
        if inference_out.returncode != 0:
            print('TRAIN EVAL FAIL during inference:', inference_out.stderr)
            raise Exception('Inference script validation failed:', str(inference_out))
        try:
            _ = get_metrics_and_serial_log(
                results_file=stage_to_output[evaluation_stage],
                test_file=stage_to_input[evaluation_stage],
                output_file=stage_to_metrics_file[evaluation_stage],
                numeric_label_col=dataset_metadata['numeric_label_col'],
                iteration=iteration,
                prefix=evaluation_stage,
                delete_preds=True,
                task_type=dataset_metadata['task_type']
            )
        except Exception as e:
            log_serial_metrics(prefix=evaluation_stage, metrics=None, iteration=iteration, task_type=config.task_type)
            message = "FAIL DURING TRAIN METRICS COMPUTATION."
            raise Exception(message) from e
        print('TRAIN EVAL SUCCESS')

def get_metrics_and_serial_log(results_file, test_file, output_file, numeric_label_col, iteration, prefix, delete_preds, task_type):
    metrics = get_metrics(
        results_file=results_file,
        test_file=test_file,
        output_file=output_file,
        numeric_label_col=numeric_label_col,
        delete_preds=delete_preds,
        task_type=task_type
    )
    log_serial_metrics(prefix=prefix, metrics=metrics, iteration=iteration, task_type=task_type)
    return metrics
