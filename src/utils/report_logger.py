from pathlib import Path
import textwrap
from pydantic_ai.messages import ModelRequest, SystemPromptPart, UserPromptPart
from pydantic_ai.models import ModelRequestParameters
from utils.snapshots import get_best_iteration

def wrap_text(text, width=100):
    return '\n'.join(textwrap.fill(line, width) for line in str(text).split('\n'))


async def generate_summary(model, report_content):
    #TODO parametrize summary model    
    messages = [
        ModelRequest(parts=[
            SystemPromptPart(content="Summarize this ML experiment report in 5-10 lines. Focus on key decisions, approaches, and outcomes."),
            UserPromptPart(content=report_content)
        ])
    ]

    response = await model.request(
        messages=messages,
        model_settings=None,
        model_request_parameters=ModelRequestParameters(
            allow_text_output= True,
        )
    )
    return response.parts[0].content
    
async def add_summary_to_report(model, config, iteration):
    report_dir = config.reports_dir / config.agent_id
    report_dir.mkdir(parents=True, exist_ok=True)
    report_file = report_dir / f"run_report_iter_{iteration}.txt"
    
    with open(report_file, 'r') as f:
        content = f.read()
    
    summary = await generate_summary(model, content)
    wrapped_summary = wrap_text(summary, 100)
    
    with open(report_file, 'w') as f:
        f.write(f"SUMMARY\n{wrapped_summary}\n\n{content}")


def save_step_output(config, step_name, step_data, iteration):
    report_dir = config.reports_dir / config.agent_id
    report_dir.mkdir(parents=True, exist_ok=True)
    
    step_titles = {
        'data_exploration': '[DATA EXPLORATION]',
        'data_representation': '[DATA REPRESENTATION]', 
        'model_architecture': '[MODEL ARCHITECTURE]',
        'model_training': '[MODEL TRAINING]'
    }
    
    title = step_titles.get(step_name, f'[{step_name.upper()}]')
    
    with open(report_dir /f"run_report_iter_{iteration}.txt", 'a') as f:
        f.write(f"{title}\n")
        for key, value in step_data.model_dump().items():
            wrapped_value = wrap_text(value)
            f.write(f"{key}: {wrapped_value}\n")
        f.write("\n")


def add_metrics_to_report(config, iteration):
    output_dir = Path(config.runs_dir) / config.agent_id
    report_dir = config.reports_dir / config.agent_id
    report_dir.mkdir(parents=True, exist_ok=True)
    
    metrics = [
        ('train_metrics.txt', 'Train Metrics', output_dir),
        ('validation_metrics.txt', 'Validation Metrics', output_dir)
    ]
    
    with open(report_dir / f"run_report_iter_{iteration}.txt", 'a') as f:
        f.write("[METRICS]\n\n")
        for metric_file, display_name, file_dir in metrics:
            with open(file_dir / metric_file, 'r') as file:
                content = file.read().strip()
                f.write(f"{display_name}:\n{content}\n\n")


def add_final_test_metrics_to_best_report(config):
    best_iteration = get_best_iteration(config)

    output_dir = Path(config.runs_dir) / config.agent_id
    agent_reports_dir = config.reports_dir / config.agent_id
    best_report = agent_reports_dir / f"run_report_iter_{best_iteration}_BEST.txt"
    
    test_metrics_file = output_dir / "test_metrics.txt"
    if not test_metrics_file.exists():
        print('No test metrics file found to add to best report.')
        return

    with open(test_metrics_file, 'r') as f:
        test_metrics = f.read().strip()
    
    with open(best_report, 'r') as f:
        content = f.read()
    
    lines = content.split('\n')
    lines.append(f"Test Metrics:\n{test_metrics}")
    
    with open(best_report, 'w') as f:
        f.write('\n'.join(lines))

def rename_best_iteration_report(config):
    best_iteration = get_best_iteration(config)
    report_dir = config.reports_dir / config.agent_id

    old_file = report_dir / f"run_report_iter_{best_iteration}.txt"
    new_file = report_dir / f"run_report_iter_{best_iteration}_BEST.txt"

    if old_file.exists():
        old_file.rename(new_file)
    else:
        print(f"WARNING: No report found at {old_file} to rename")
