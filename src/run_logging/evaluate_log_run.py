import json
import subprocess
import traceback

from pydantic_ai import ModelRetry
from eval.evaluate_result import get_metrics
from run_logging.logging_helpers import log_inference_stage_and_metrics, log_serial_metrics

def run_inference_and_log(config, iteration, evaluation_stage, use_best_snapshot=False):
    with open(f"/repository/datasets/{config['dataset']}/metadata.json") as f:
        dataset_metadata = json.load(f)

    run_dir = f"/workspace/runs/{config['agent_id']}"
    snapshot_dir = f"/snapshots/{config['agent_id']}"
    source_folder = 'snapshots' if use_best_snapshot else 'workspace'
    if(use_best_snapshot):
        print('USING BEST SNAPSHOT')
    conda_path = {
        'workspace': f"{run_dir}/.conda/envs/{config['agent_id']}_env",
        'snapshots': f"{snapshot_dir}/.conda/envs/{config['agent_id']}_env",
    }
    inference_path = {
        'workspace': f"{run_dir}/inference.py",
        'snapshots': f"{snapshot_dir}/inference.py",
    }
    stage_to_input = {
        'dry_run': dataset_metadata['train_split'],
        'validation': run_dir + "/validation.csv",
        'test': dataset_metadata['test_split_no_labels'],
        'train': run_dir + "/train.csv",
        'stealth_test': dataset_metadata['test_split_no_labels'],
    }
    stage_to_output = {
        'dry_run': run_dir + "/eval_predictions_dry_run.csv",
        'validation': run_dir + "/eval_predictions_validation.csv",
        'test': run_dir + "/eval_predictions_test.csv",
        'train': run_dir + "/eval_predictions_train.csv",
        'stealth_test': snapshot_dir + "/eval_predictions_stealth_test.csv",
    }
    stage_to_metrics_file = {
        'dry_run': f"{run_dir}/dry_run_metrics.txt",
        'validation': f"{run_dir}/validation_metrics.txt",
        'test': f"{run_dir}/test_metrics.txt",
        'train': f"{run_dir}/train_metrics.txt",
        'stealth_test': f"{snapshot_dir}/stealth_test_metrics.txt",
    }

    command_prefix=f"source /opt/conda/etc/profile.d/conda.sh && conda activate {conda_path[source_folder]}"
    command = f"{command_prefix} && python {inference_path[source_folder]} --input {stage_to_input[evaluation_stage]} --output {stage_to_output[evaluation_stage]}"
    inference_out = subprocess.run(command, shell=True, executable="/bin/bash", capture_output=True)
    if(evaluation_stage == 'stealth_test'):
        print('RUNNING STEALTH TEST EVAL')
        try:
            _ = get_metrics_and_serial_log(
                results_file=stage_to_output[evaluation_stage],
                test_file=f"{dataset_metadata['test_split_with_labels']}",
                output_file=None,
                numeric_label_col=dataset_metadata['numeric_label_col'],
                iteration=iteration,
                prefix=evaluation_stage,
                delete_preds=True,
            )
        except Exception as e:
            print('STEALTH TEST EVAL FAIL')
            log_serial_metrics(prefix=evaluation_stage, metrics=None, iteration=iteration)
            return
        print('STEALTH TEST EVAL SUCCESS')
        return
    if(evaluation_stage == 'test'):
        print('RUNNING TEST EVAL')
        if(inference_out.returncode != 0):
            print('TEST EVAL FAIL', str(inference_out))
            log_inference_stage_and_metrics(1)
            raise Exception('Inference on TEST script failed:', str(inference_out))
        try:
            test_metrics = get_metrics(
                results_file=stage_to_output[evaluation_stage],
                test_file=f"{dataset_metadata['test_split_with_labels']}",
                output_file=stage_to_metrics_file[evaluation_stage],
                numeric_label_col=dataset_metadata['numeric_label_col'],
                delete_preds=True,
            )
            log_inference_stage_and_metrics(2, metrics=test_metrics)
        except Exception as e:
            print('TEST EVAL FAIL', {traceback.format_exc()})
            log_inference_stage_and_metrics(1)
            return
        print('TEST EVAL SUCCESS')
    if(evaluation_stage == 'dry_run'):
        print('RUNNING DRY RUN EVAL')
        if(inference_out.returncode != 0):
            print('DRY RUN EVAL FAIL during inference:', inference_out.stderr)
            raise ModelRetry(f'Inference script validation failed: {str(inference_out)}')
        try:
            _ = get_metrics(
                results_file=stage_to_output[evaluation_stage],
                test_file=stage_to_input[evaluation_stage],
                output_file=stage_to_metrics_file[evaluation_stage],
                numeric_label_col=dataset_metadata['numeric_label_col'],
                delete_preds=True,
            )
        except Exception as e:
            message = f"FAIL DURING DRY RUN METRICS COMPUTATION. {traceback.format_exc()}"
            print(message)
            raise ModelRetry(message)
        print('DRY RUN EVAL SUCCESS')
    if(evaluation_stage == 'validation'):
        print('RUNNING VALIDATION EVAL')
        if(inference_out.returncode != 0):
            print('VALIDATION EVAL FAIL during inference:', inference_out.stderr)
            log_serial_metrics(prefix=evaluation_stage, metrics=None, iteration=iteration)
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
            )
        except Exception as e:
            log_serial_metrics(prefix="validation", metrics=None, iteration=iteration)
            message = "FAIL DURING VALIDATION METRICS COMPUTATION."
            raise Exception(message) from e
        print('VALIDATION EVAL SUCCESS')
    if(evaluation_stage == 'train'):
        print('RUNNING TRAIN EVAL')
        if(inference_out.returncode != 0):
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
            )
        except Exception as e:
            log_serial_metrics(prefix=evaluation_stage, metrics=None, iteration=iteration)
            message = "FAIL DURING TRAIN METRICS COMPUTATION."
            raise Exception(message) from e
        print('TRAIN EVAL SUCCESS')

def get_metrics_and_serial_log(results_file, test_file, output_file, numeric_label_col, iteration, prefix, delete_preds):
    metrics = get_metrics(
        results_file=results_file,
        test_file=test_file,
        output_file=output_file,
        numeric_label_col=numeric_label_col,
        delete_preds=delete_preds,
    )
    log_serial_metrics(prefix=prefix, metrics=metrics, iteration=iteration)
    return metrics
