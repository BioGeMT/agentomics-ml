from pathlib import Path
import textwrap
from openai import AsyncOpenAI
import os
from utils.snapshots import get_best_iteration

def wrap_text(text, width=100):
    return '\n'.join(textwrap.fill(line, width) for line in str(text).split('\n'))


async def generate_summary(config, report_content):
    client = AsyncOpenAI(
        base_url='https://openrouter.ai/api/v1',
        api_key=openrouter_api_key
    )
    
    response = await client.chat.completions.create(
        model=config.model_name,
        messages=[
            {"role": "system", "content": "Summarize this ML experiment report in 5-10 lines. Focus on key decisions, approaches, and outcomes."},
            {"role": "user", "content": report_content}
        ]
    )
    
    return response.choices[0].message.content.strip()


async def add_summary_to_report(config, iteration):
    reports_dir = Path(config.reports_dir)
    reports_dir.mkdir(parents=True, exist_ok=True)
    report_file = reports_dir / config.agent_id / f"run_report_iter_{iteration}.txt"
    
    with open(report_file, 'r') as f:
        content = f.read()
    
    summary = await generate_summary(config, content)
    wrapped_summary = wrap_text(summary, 100)
    
    with open(report_file, 'w') as f:
        f.write(f"SUMMARY\n{wrapped_summary}\n\n{content}")


def save_step_output(config, step_name, step_data, iteration):
    reports_dir = config.reports_dir
    reports_dir.mkdir(parents=True, exist_ok=True)
    
    step_titles = {
        'data_exploration': '[DATA EXPLORATION]',
        'data_representation': '[DATA REPRESENTATION]', 
        'model_architecture': '[MODEL ARCHITECTURE]',
        'model_training': '[MODEL TRAINING]'
    }
    
    title = step_titles.get(step_name, f'[{step_name.upper()}]')
    
    with open(reports_dir / config.agent_id /f"run_report_iter_{iteration}.txt", 'a') as f:
        f.write(f"{title}\n")
        for key, value in step_data.model_dump().items():
            wrapped_value = wrap_text(value)
            f.write(f"{key}: {wrapped_value}\n")
        f.write("\n")


def add_metrics_to_report(config, iteration):
    output_dir = Path(config.runs_dir) / config.agent_id
    reports_dir = config.reports_dir
    reports_dir.mkdir(parents=True, exist_ok=True)
    
    metrics = [
        ('train_metrics.txt', 'Train Metrics', output_dir),
        ('validation_metrics.txt', 'Validation Metrics', output_dir)
    ]
    
    with open(reports_dir / config.agent_id / f"run_report_iter_{iteration}.txt", 'a') as f:
        f.write("[METRICS]\n\n")
        for metric_file, display_name, file_dir in metrics:
            with open(file_dir / metric_file, 'r') as file:
                content = file.read().strip()
                f.write(f"{display_name}:\n{content}\n\n")


def add_final_test_metrics_to_best_report(config):
    best_iteration = get_best_iteration(config)

    output_dir = Path(config.runs_dir) / config.agent_id
    reports_dir = config.reports_dir
    best_report = reports_dir / config.agent_id / f"run_report_iter_{best_iteration}.txt"
    #TODO we write this at final test stage to the agent's run dir - change to a better location for clarity
    test_metrics_file = output_dir / "test_metrics.txt"
    
    with open(test_metrics_file, 'r') as f:
        test_metrics = f.read().strip()
    
    with open(best_report, 'r') as f:
        content = f.read()
    
    lines = content.split('\n')
    lines.append(f"Test Metrics:\n{test_metrics}")
    
    with open(best_report, 'w') as f:
        f.write('\n'.join(lines))

def rename_and_snapshot_best_iteration_report(config):
    best_iteration = get_best_iteration(config)
    reports_dir = config.reports_dir
    snapshot_dir = Path(config.snapshots_dir) / config.agent_id
    
    old_file = reports_dir / config.agent_id / f"run_report_iter_{best_iteration}.txt"
    new_file = reports_dir / config.agent_id / f"run_report_iter_{best_iteration}_BEST.txt"
    snapshot_file = snapshot_dir / config.agent_id / f"run_report_iter_{best_iteration}_BEST.txt"
    
    old_file.rename(new_file)
    
    snapshot_dir.mkdir(parents=True, exist_ok=True)
    with open(new_file, 'r') as src:
        with open(snapshot_file, 'w') as dst:
            dst.write(src.read())