from pathlib import PurePosixPath
from pathlib import Path
import textwrap
import datetime

from pydantic_ai.messages import ModelRequest, SystemPromptPart, UserPromptPart
from pydantic_ai.models import ModelRequestParameters
from utils.snapshots import get_best_iteration

# Utilities
def clean_path(iteration, text: str, config=None) -> str:
    s = str(text).replace("\\", "/")
    s = s.replace("/workspace/", "")
    if config is None:
        return s
    root_files = {"train.csv", "test.csv", "validation.csv"}
    name = PurePosixPath(s).name
    is_root_file = name in root_files

    if iteration is None:
        replacement = "run_files/"
    else:
        replacement = "run_files/" if is_root_file else f"run_files/iteration_{iteration}/"
    s = s.replace(f"runs/{config.agent_id}/", replacement)
    return s


def wrap_text(text, width=100):
    return "\n".join(textwrap.fill(line, width) for line in str(text).split("\n"))


def _md_block(value: str) -> str:
    s = str(value).replace("\r\n", "\n").strip()
    if "\n" in s or len(s) > 120:
        return "\n".join([f"> {line}" if line.strip() else ">" for line in s.split("\n")])
    return s


def humanize_step_title(step_name: str) -> str:
    """
    Make headings readable in MD/PDF.
    Examples: dataexploration -> Data exploration, modeltraining -> Model training
    """
    s = step_name.replace("_", " ").strip()
    fixes = {
        "dataexploration": "Data Exploration",
        "predictionexploration": "Prediction Exploration",
        "modelarchitecture": "Model Architecture",
        "modeltraining": "Model Training",
        "modelinference": "Model Inference",
        "datarepresentation": "Data Representation",
        "datasplit": "Data Split",
    }
    key = s.replace(" ", "").lower()
    return fixes.get(key, s.title())


# Markdown helpers
def _md_header_if_missing(config, iteration: int):
    report_dir = config.reports_dir / config.agent_id
    report_dir.mkdir(parents=True, exist_ok=True)
    md_path = report_dir / f"run_report_iter_{iteration}.md"

    if md_path.exists():
        return

    header = (
        f"# Run Report - Iteration {iteration}\n\n"
        f"**Agent ID:** `{config.agent_id}`  \n"
        f"**Dataset:** `{config.dataset}`  \n"
        f"**Model:** `{config.model_name}`  \n"
        f"**Task:** `{config.task_type}`  \n"
        f"**Validation metric:** `{config.val_metric}`  \n"
        f"**Generated:** {datetime.datetime.now().strftime('%Y-%m-%d %H:%M:%S')}\n\n"
        f"---\n\n"
    )
    md_path.write_text(header, encoding="utf-8")


def _append_md_section(config, iteration: int, title: str, body: str):
    md_path = config.reports_dir / config.agent_id / f"run_report_iter_{iteration}.md"
    with open(md_path, "a", encoding="utf-8") as f:
        f.write(f"## {title}\n\n{body}\n\n")


def _prepend_md_section(config, iteration: int, title: str, body: str):
    md_path = config.reports_dir / config.agent_id / f"run_report_iter_{iteration}.md"
    content = md_path.read_text(encoding="utf-8") if md_path.exists() else ""
    section = f"## {title}\n\n{body}\n\n"
    md_path.write_text(section + content, encoding="utf-8")


async def generate_summary(model, report_content):
    messages = [
        ModelRequest(
            parts=[
                SystemPromptPart(
                    content=(
                        "Summarize this ML experiment report in 5–10 lines. "
                        "Focus on decisions, methodology, and outcomes."
                    )
                ),
                UserPromptPart(content=report_content),
            ]
        )
    ]
    response = await model.request(
        messages=messages,
        model_settings=None,
        model_request_parameters=ModelRequestParameters(allow_text_output=True),
    )
    return response.parts[0].content

async def add_summary_to_report(model, config, iteration):
    _md_header_if_missing(config, iteration)
    md_path = config.reports_dir / config.agent_id / f"run_report_iter_{iteration}.md"
    report_content = md_path.read_text(encoding="utf-8") if md_path.exists() else ""

    summary = await generate_summary(model, report_content)

    bullets = "\n".join(f"- {line.strip()}" for line in summary.split("\n") if line.strip())
    _prepend_md_section(config, iteration, "Summary", bullets or "_No summary._")

def save_step_output(config, step_name, step_data, iteration):
    report_dir = config.reports_dir / config.agent_id
    report_dir.mkdir(parents=True, exist_ok=True)

    # MD report
    _md_header_if_missing(config, iteration)
    dump = step_data.model_dump()

    # Detect skipped step
    text_values = [
        str(v).lower()
        for v in dump.values()
        if isinstance(v, str) and v.strip()
    ]
    is_skipped = bool(text_values) and all("skipped" in v for v in text_values)
    if is_skipped:
        if step_name.replace("_", "").lower() == "dataexploration":
            return

        _append_md_section(
            config,
            iteration,
            humanize_step_title(step_name),
            "Step skipped for this iteration.",
        )
        return

    body = []
    for k, v in dump.items():
        if k == "files_created":
            continue
        if k == "unresolved_issues":
            continue

        if isinstance(v, str) and ("path" in k.lower() or "dir" in k.lower()):
            v = clean_path(iteration, v, config)
        body.append(f"**{k.replace('_',' ').title()}:**\n\n{_md_block(v)}\n")

    files = dump.get("files_created")
    if files:
        nice = [Path(clean_path(iteration, x, config)).name for x in files]
        body.append("**Files created:**\n\n" + "\n".join(f"- `{x}`" for x in nice))

    _append_md_section(
        config,
        iteration,
        humanize_step_title(step_name),
        "\n".join(body).strip(),
    )

# Metrics
def add_metrics_to_report(config, iteration, metrics_dict):
    report_dir = config.reports_dir / config.agent_id
    report_dir.mkdir(parents=True, exist_ok=True)

    _md_header_if_missing(config, iteration)
    lines = [f"- **{k}**: {v}" for k, v in metrics_dict.items()]
    _append_md_section(config, iteration, "Metrics", "\n".join(lines) or "_No metrics._")


# Final test metrics (best iteration only)
def add_final_test_metrics_to_best_report(config):
    best_iter = get_best_iteration(config)
    if best_iter is None:
        return

    test_metrics_path = Path(config.runs_dir) / config.agent_id / "test_metrics.txt"
    if not test_metrics_path.exists():
        return

    test_metrics = test_metrics_path.read_text().strip()

    _md_header_if_missing(config, best_iter)
    _append_md_section(config, best_iter, "Test metrics", f"```\n{test_metrics}\n```")
