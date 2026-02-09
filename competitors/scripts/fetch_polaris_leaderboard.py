#!/usr/bin/env python3
from __future__ import annotations

import argparse
import re
from io import StringIO
from pathlib import Path

import pandas as pd
import requests
import yaml
from bs4 import BeautifulSoup

NAME_COLUMN_HINTS = ("team", "model", "name", "method")
DATE_COLUMN_HINTS = ("date", "time")

METRIC_ALIASES = {
    "pr_auc": ("pr_auc", "prauc", "averageprecision", "auprc"),
    "roc_auc": ("roc_auc", "rocauc", "auroc"),
    "pearsonr": ("pearsonr", "pearson"),
    "spearmanr": ("spearmanr", "spearman"),
    "mean_squared_error": ("mean_squared_error", "meansquarederror", "mse", "rmse"),
    "mean_absolute_error": ("mean_absolute_error", "meanabsoluteerror", "mae"),
}


def _normalize_name(value: str) -> str:
    return re.sub(r"[^a-z0-9]+", "", value.lower())


def _get_task_paths(task_id: str, competitors_dir: Path) -> tuple[Path, Path]:
    task_name = task_id.split("/", 1)[1]
    task_dir = competitors_dir / "biomlbench" / "biomlbench" / "tasks" / "polarishub" / task_name
    assert task_dir.is_dir(), f"Missing task directory: {task_dir}"
    config_path = task_dir / "config.yaml"
    assert config_path.is_file(), f"Missing task config: {config_path}"
    return task_dir, config_path


def _pick_name_column(df: pd.DataFrame) -> str:
    for col in df.columns:
        normalized = _normalize_name(str(col))
        if any(hint in normalized for hint in NAME_COLUMN_HINTS):
            return str(col)
    raise AssertionError(f"Could not find team/model column in columns: {list(df.columns)}")


def _pick_metric_column(df: pd.DataFrame, main_metric: str) -> str:
    aliases = METRIC_ALIASES.get(main_metric)
    assert aliases is not None, f"Unsupported Polaris metric '{main_metric}'"

    normalized_to_original = {_normalize_name(str(col)): str(col) for col in df.columns}
    for alias in aliases:
        alias_norm = _normalize_name(alias)
        for normalized_name, original_name in normalized_to_original.items():
            if alias_norm in normalized_name:
                parsed = pd.to_numeric(df[original_name], errors="coerce")
                if parsed.notna().any():
                    return original_name

    raise AssertionError(
        f"Could not find score column for metric '{main_metric}' in columns: {list(df.columns)}"
    )


def _pick_date_column(df: pd.DataFrame) -> str | None:
    for col in df.columns:
        normalized = _normalize_name(str(col))
        if any(hint in normalized for hint in DATE_COLUMN_HINTS):
            return str(col)
    return None


def _extract_table(benchmark_id: str) -> pd.DataFrame:
    url = f"https://polarishub.io/benchmarks/{benchmark_id}"
    response = requests.get(
        url,
        timeout=30,
        headers={
            "User-Agent": (
                "Mozilla/5.0 (X11; Linux x86_64) "
                "AppleWebKit/537.36 (KHTML, like Gecko) Chrome/131.0.0.0 Safari/537.36"
            )
        },
    )
    response.raise_for_status()

    soup = BeautifulSoup(response.content, "html.parser")
    table = soup.select_one('table[data-slot="table"]')
    assert table is not None, f"No leaderboard table found on {url}"

    tables = pd.read_html(StringIO(str(table)), header=0)
    assert len(tables) > 0, f"Failed to parse leaderboard table on {url}"
    df = tables[0]
    assert not df.empty, f"Parsed leaderboard table is empty on {url}"

    # Drop auto-generated "Unnamed" columns from HTML parsing.
    df = df.loc[:, ~df.columns.astype(str).str.contains(r"^Unnamed", case=False)]
    assert not df.empty, f"Leaderboard table has no usable columns on {url}"
    return df


def _build_leaderboard(df: pd.DataFrame, main_metric: str) -> pd.DataFrame:
    name_col = _pick_name_column(df)
    score_col = _pick_metric_column(df, main_metric)
    date_col = _pick_date_column(df)

    scores = pd.to_numeric(df[score_col], errors="coerce")
    names = df[name_col].astype(str).str.strip()
    valid = names.ne("") & names.str.lower().ne("nan") & scores.notna()
    assert valid.any(), "No valid leaderboard rows after filtering"

    if date_col is not None:
        submission_dates = df[date_col].astype(str)
    else:
        submission_dates = pd.Series(["2024-01-01"] * len(df), index=df.index, dtype="object")

    out = pd.DataFrame(
        {
            "teamName": names[valid].values,
            "score": scores[valid].values,
            "submissionDate": submission_dates[valid].values,
        }
    )
    assert not out.empty, "Normalized leaderboard is empty"
    return out


def main() -> None:
    parser = argparse.ArgumentParser(description="Fetch Polaris leaderboard into task leaderboard.csv")
    parser.add_argument("--task-id", required=True, help="Task id like polarishub/polaris-pkis2-egfr-wt-c-1")
    parser.add_argument(
        "--competitors-dir",
        default=str(Path(__file__).resolve().parents[1]),
        help="Path to competitors directory",
    )
    args = parser.parse_args()

    task_id = args.task_id
    assert task_id.startswith("polarishub/"), f"Expected polarishub task id, got: {task_id}"

    competitors_dir = Path(args.competitors_dir).resolve()
    task_dir, config_path = _get_task_paths(task_id, competitors_dir)
    config = yaml.safe_load(config_path.read_text())

    benchmark_id = config["data_source"]["benchmark_id"]
    main_metric = config["biomedical_metadata"]["polaris_main_metric"]

    raw_table = _extract_table(benchmark_id)
    leaderboard = _build_leaderboard(raw_table, main_metric=main_metric)

    leaderboard_path = task_dir / "leaderboard.csv"
    leaderboard.to_csv(leaderboard_path, index=False)
    print(
        f"[fetch_polaris_leaderboard] {task_id}: "
        f"rows={len(leaderboard)} metric={main_metric} -> {leaderboard_path}"
    )


if __name__ == "__main__":
    main()
