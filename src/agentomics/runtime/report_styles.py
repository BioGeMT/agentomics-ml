from __future__ import annotations

from reportlab.lib import colors
from reportlab.lib.units import cm
from reportlab.platypus import Paragraph, Table, TableStyle
from reportlab.lib.styles import getSampleStyleSheet, ParagraphStyle

from agentomics.runtime.report_data import RunMeta


REPORT_PALETTE = {
    "navy": "#17324D",
    "teal": "#007F86",
    "green": "#00876C",
    "orange": "#E69F00",
    "vermilion": "#D55E00",
    "slate": "#526575",
    "grid": "#D5DEE5",
    "pale_blue": "#EAF1F5",
    "pale_teal": "#DCEFEA",
    "pale_gold": "#FFF1CC",
    "pale_gray": "#F3F6F8",
    "white": "#FFFFFF",
}


def pdf_color(name: str):
    return colors.HexColor(REPORT_PALETTE[name])


def build_styles():
    styles = getSampleStyleSheet()
    styles.add(
        ParagraphStyle(
            name="H1",
            parent=styles["Heading1"],
            fontName="Helvetica-Bold",
            fontSize=18,
            textColor=pdf_color("navy"),
            spaceAfter=12,
        )
    )
    styles.add(
        ParagraphStyle(
            name="H2",
            parent=styles["Heading2"],
            fontName="Helvetica-Bold",
            fontSize=13,
            textColor=pdf_color("teal"),
            spaceBefore=12,
            spaceAfter=6,
        )
    )
    styles.add(
        ParagraphStyle(
            name="Body",
            parent=styles["BodyText"],
            fontName="Helvetica",
            fontSize=10,
            leading=14,
            textColor=pdf_color("slate"),
            spaceAfter=6,
        )
    )
    styles.add(
        ParagraphStyle(
            name="Muted",
            parent=styles["BodyText"],
            fontName="Helvetica",
            fontSize=9.5,
            leading=12,
            textColor=pdf_color("slate"),
            spaceAfter=6,
        )
    )
    styles.add(
        ParagraphStyle(
            name="Quote",
            parent=styles["BodyText"],
            fontName="Helvetica-Oblique",
            fontSize=9.8,
            leading=13,
            textColor=pdf_color("navy"),
            leftIndent=14,
            spaceBefore=4,
            spaceAfter=8,
        )
    )
    styles.add(
        ParagraphStyle(
            name="MiniHeader",
            parent=styles["Body"],
            fontName="Helvetica-Bold",
            fontSize=10.5,
            textColor=pdf_color("teal"),
            spaceBefore=8,
            spaceAfter=4,
        )
    )
    styles.add(
        ParagraphStyle(
            name="CodeBlock",
            parent=styles["BodyText"],
            fontName="Courier",
            fontSize=9,
            leading=11,
            textColor=pdf_color("navy"),
            backColor=pdf_color("pale_gray"),
            borderPadding=6,
            spaceBefore=4,
            spaceAfter=8,
        )
    )
    return styles


def escape_report_text(value: str) -> str:
    return value.replace("&", "&amp;").replace("<", "&lt;").replace(">", "&gt;")


def format_metric(value) -> str:
    if value is None:
        return "—"
    if isinstance(value, (int, float)):
        if abs(value) >= 1e6 or (abs(value) > 0 and abs(value) < 1e-3):
            return f"{value:.3g}"
        return f"{value:.6g}"
    return str(value)


def run_meta_flowables(meta: RunMeta):
    rows = [
        ("Agent ID", meta.agent_id),
        ("Model", meta.model_name),
        ("Dataset", meta.dataset),
        ("Task", meta.task_type),
        ("Optimized Metric", meta.val_metric or "—"),
        (
            "Split Allowed Iterations",
            meta.split_allowed_iterations if meta.split_allowed_iterations is not None else "—",
        ),
        (
            "Exploration Iterations",
            meta.exploration_iterations if meta.exploration_iterations is not None else "—",
        ),
    ]
    data = [[key, str(value)] for key, value in rows]

    table = Table(data, colWidths=[5 * cm, 10 * cm], hAlign="LEFT")
    table.setStyle(
        TableStyle(
            [
                ("FONTNAME", (0, 0), (0, -1), "Helvetica-Bold"),
                ("FONTSIZE", (0, 0), (-1, -1), 9),
                ("BACKGROUND", (0, 0), (0, -1), pdf_color("pale_blue")),
                ("TEXTCOLOR", (0, 0), (0, -1), pdf_color("navy")),
                ("BACKGROUND", (1, 0), (1, -1), pdf_color("white")),
                ("GRID", (0, 0), (-1, -1), 0.35, pdf_color("grid")),
                ("LEFTPADDING", (0, 0), (-1, -1), 6),
                ("RIGHTPADDING", (0, 0), (-1, -1), 6),
                ("TOPPADDING", (0, 0), (-1, -1), 4),
                ("BOTTOMPADDING", (0, 0), (-1, -1), 4),
            ]
        )
    )
    return table
