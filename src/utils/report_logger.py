from pathlib import Path
import textwrap
import datetime

from pydantic_ai.messages import ModelRequest, SystemPromptPart, UserPromptPart
from pydantic_ai.models import ModelRequestParameters

# Utilities
def wrap_text(text, width=100):
    return "\n".join(textwrap.fill(line, width) for line in str(text).split("\n"))

def _md_escape(text: str) -> str:
    return str(text).replace("\r\n", "\n")

def _md_block(value: str) -> str:
    s = _md_escape(value).strip()
    if "\n" in s or len(s) > 120:
        return "\n".join([f"> {line}" if line.strip() else ">" for line in s.split("\n")])
    return s

# Markdown helpers
def _md_header_if_missing(config, iteration: int):
    report_dir = config.reports_dir / config.agent_id
    report_dir.mkdir(parents=True, exist_ok=True)
    md_path = report_dir / f"run_report_iter_{iteration}.md"

    if md_path.exists():
        return

    header = (
        f"# Agentomics Run Report — Iteration {iteration}\n\n"
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

# Summary
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
    report_path = config.reports_dir / config.agent_id / f"run_report_iter_{iteration}.txt"
    content = report_path.read_text()

    summary = await generate_summary(model, content)
    report_path.write_text(f"SUMMARY\n{wrap_text(summary)}\n\n{content}")

    _md_header_if_missing(config, iteration)
    bullets = "\n".join(f"- {line.strip()}" for line in summary.split("\n") if line.strip())
    _append_md_section(config, iteration, "Summary", bullets or "_No summary._")

# Step logging
def save_step_output(config, step_name, step_data, iteration):
    report_dir = config.reports_dir / config.agent_id
    report_dir.mkdir(parents=True, exist_ok=True)

    # TXT
    txt_path = report_dir / f"run_report_iter_{iteration}.txt"
    with open(txt_path, "a") as f:
        f.write(f"[{step_name.upper()}]\n")
        for k, v in step_data.model_dump().items():
            f.write(f"{k}: {wrap_text(v)}\n")
        f.write("\n")

    # MD
    _md_header_if_missing(config, iteration)
    body = []

    dump = step_data.model_dump()
    for k, v in dump.items():
        if k == "files_created":
            continue
        body.append(f"**{k.replace('_',' ').title()}:**\n\n{_md_block(v)}\n")

    files = dump.get("files_created")
    if files:
        body.append(
            "<details>\n<summary><b>Files created</b></summary>\n\n"
            + "\n".join(f"- `{x}`" for x in files)
            + "\n\n</details>"
        )

    _append_md_section(
        config,
        iteration,
        step_name.replace("_", " ").title(),
        "\n".join(body).strip(),
    )

# Metrics
def add_metrics_to_report(config, iteration, metrics_dict):
    report_dir = config.reports_dir / config.agent_id
    report_dir.mkdir(parents=True, exist_ok=True)

    txt_path = report_dir / f"run_report_iter_{iteration}.txt"
    with open(txt_path, "a") as f:
        f.write("[METRICS]\n")
        for k, v in metrics_dict.items():
            f.write(f"{k}: {v}\n")
        f.write("\n")

    _md_header_if_missing(config, iteration)
    lines = [f"- **{k}**: {v}" for k, v in metrics_dict.items()]
    _append_md_section(config, iteration, "Metrics", "\n".join(lines) or "_No metrics._")

# Final test metrics (best iteration only)
def add_final_test_metrics_to_best_report(config):
    from utils.snapshots import get_best_iteration

    best_iter = get_best_iteration(config)
    if best_iter is None:
        return

    test_metrics_path = Path(config.runs_dir) / config.agent_id / "test_metrics.txt"
    if not test_metrics_path.exists():
        return

    test_metrics = test_metrics_path.read_text().strip()

    txt_path = config.reports_dir / config.agent_id / f"run_report_iter_{best_iter}.txt"
    txt_path.write_text(txt_path.read_text() + f"\n[Test Metrics]\n{test_metrics}\n")

    _md_header_if_missing(config, best_iter)
    _append_md_section(config, best_iter, "Test metrics", f"```\n{test_metrics}\n```")
