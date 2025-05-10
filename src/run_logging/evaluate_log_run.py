import json
import subprocess

from pydantic_ai import ModelRetry
from eval.evaluate_result import get_metrics
from run_logging.logging_helpers import log_inference_stage_and_metrics, log_serial_metrics

def run_inference_and_log(config, iteration, evaluation_stage):
    with open(f"/repository/datasets/{config['dataset']}/metadata.json") as f:
        dataset_metadata = json.load(f)

    run_dir = f"/workspace/runs/{config['agent_id']}"
    stage_to_input = {
        'dry_run': dataset_metadata['train_split'],
        'validation': run_dir + "/validation.csv",
        'test': dataset_metadata['test_split_no_labels'],
        'train': run_dir + "/train.csv",
    }
    stage_to_output = {
        'dry_run': "/dev/null",
        'validation': run_dir + "/eval_predictions_validation.csv",
        'test': run_dir + "/eval_predictions_test.csv",
        'train': run_dir + "/eval_predictions_train.csv",
    }

    agent_env_name = f"{run_dir}/.conda/envs/{config['agent_id']}_env"
    command_prefix=f"source /opt/conda/etc/profile.d/conda.sh && conda activate {agent_env_name}"
    command = f"{command_prefix} && python {run_dir}/inference.py --input {stage_to_input[evaluation_stage]} --output {stage_to_output[evaluation_stage]}"
    inference_out = subprocess.run(command, shell=True, executable="/bin/bash", capture_output=True)

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
                output_file=f"{run_dir}/metrics.txt",
                numeric_label_col=dataset_metadata['numeric_label_col'],
            )
            log_inference_stage_and_metrics(2, metrics=test_metrics)
        except Exception as e:
            print('TEST EVAL FAIL', e)
            log_inference_stage_and_metrics(1)
            return
        print('TEST EVAL SUCCESS')
    if(evaluation_stage == 'dry_run'):
        print('RUNNING DRY RUN EVAL')
        if(inference_out.returncode != 0):
            print('DRY RUN EVAL FAIL', inference_out.stderr)
            raise ModelRetry(f'Inference script validation failed: {str(inference_out)}')
        print('DRY RUN EVAL SUCCESS')
    if(evaluation_stage == 'validation'):
        print('RUNNING VALIDATION EVAL')
        if(inference_out.returncode != 0):
            print('VALIDATION EVAL FAIL during inference:', inference_out.stderr)
            log_serial_metrics(prefix="validation", metrics=None, iteration=iteration)
            raise Exception('Inference script validation failed:', str(inference_out))
        try:
            _ = get_metrics_and_serial_log(
                results_file=stage_to_output[evaluation_stage],
                test_file=stage_to_input[evaluation_stage],
                output_file=f"{run_dir}/validation_metrics.txt",
                numeric_label_col=dataset_metadata['numeric_label_col'],
                iteration=iteration,
                prefix="validation",
            )
        except Exception as e:
            # add message to the exception
            log_serial_metrics(prefix="validation", metrics=None, iteration=iteration)
            message = "FAIL DURING VALIDATION METRICS COMPUTATION."
            raise type(e)(f"{message} {str(e)}").with_traceback(e.__traceback__)
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
                output_file=f"{run_dir}/train_metrics.txt",
                numeric_label_col=dataset_metadata['numeric_label_col'],
                iteration=iteration,
                prefix="train",
            )
        except Exception as e:
            log_serial_metrics(prefix="train", metrics=None, iteration=iteration)
            message = "FAIL DURING TRAIN METRICS COMPUTATION."
            raise type(e)(f"{message} {str(e)}").with_traceback(e.__traceback__)
        print('TRAIN EVAL SUCCESS')

def get_metrics_and_serial_log(results_file, test_file, output_file, numeric_label_col, iteration, prefix):
    metrics = get_metrics(
        results_file=results_file,
        test_file=test_file,
        output_file=output_file,
        numeric_label_col=numeric_label_col,
    )
    log_serial_metrics(prefix=prefix, metrics=metrics, iteration=iteration)
    return metrics
