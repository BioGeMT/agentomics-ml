import json
import subprocess

from pydantic_ai import ModelRetry
from eval.evaluate_result import get_metrics
from run_logging.logging_helpers import log_inference_stage_and_metrics, log_validation_metrics

def run_inference_and_log(config, iteration, evaluation_stage):
    with open(f"/repository/datasets/{config['dataset']}/metadata.json") as f:
        dataset_metadata = json.load(f)

    run_dir = f"/workspace/runs/{config['agent_id']}"
    stage_to_input = {
        'dry_run': dataset_metadata['train_split'],
        'validation': run_dir + "/validation.csv",
        'test': dataset_metadata['test_split_no_labels']
    }
    stage_to_output = {
        'dry_run': "/dev/null",
        'validation': run_dir + "/eval_predictions.csv", #TODO
        'test': run_dir + "/eval_predictions.csv"
    }

    agent_env_name = f"{run_dir}/.conda/envs/{config['agent_id']}_env"
    command_prefix=f"source /opt/conda/etc/profile.d/conda.sh && conda activate {agent_env_name}"
    command = f"{command_prefix} && python {run_dir}/inference.py --input {stage_to_input[evaluation_stage]} --output {stage_to_output[evaluation_stage]}"
    inference_out = subprocess.run(command, shell=True, executable="/bin/bash", capture_output=True)

    if(evaluation_stage == 'test'):
        print('RUNNING TEST EVAL')
        try:
            test_metrics = get_metrics(
                results_file=stage_to_output[evaluation_stage],
                test_file=f"{dataset_metadata['test_split_with_labels']}",
                output_file=f"{run_dir}/metrics.txt",
                label_to_scalar=dataset_metadata['label_to_scalar'],
                class_col=dataset_metadata['class_col'],
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
            raise ModelRetry('Inference script validation failed:', str(inference_out))
        print('DRY RUN EVAL SUCCESS')
    if(evaluation_stage == 'validation'):
        print('RUNNING VALIDATION EVAL')
        if(inference_out.returncode != 0):
            print('VALIDATION EVAL FAIL', inference_out.stderr)
            log_validation_metrics(metrics=None, iteration=iteration)
            return #TODO return something to add to next iteration context
        try:
            valid_metrics = get_metrics(
                results_file=stage_to_output[evaluation_stage],
                test_file=stage_to_input[evaluation_stage],
                output_file=f"{run_dir}/validation_metrics.txt",
                label_to_scalar=dataset_metadata['label_to_scalar'],
                class_col=dataset_metadata['class_col'],
            )
        except Exception as e:
            print('VALIDATION EVAL FAIL', e)
            log_validation_metrics(metrics=None, iteration=iteration)
            return #TODO return something to add to next iteration context
        
        log_validation_metrics(metrics={
            f"validation/{k}": v for k, v in valid_metrics.items()
        }, iteration=iteration)

        valid_metric = valid_metrics[config["best_metric"]]
        #TODO implement snapshoting
        print('VALIDATION EVAL SUCCESS')

