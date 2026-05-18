#!/usr/bin/env python3
"""Pull per-run info from W&B into a single CSV.

Columns:
    experiment, dataset, llm, configuration, replicate_id, success,
    selected_iteration, train_metric, validation_metric, test_metric,
    metric_name, runtime_minutes

`configuration` is a composite of the ablation case and the knowledge
integration mode, joined with '+', e.g. "baseline+static",
"no_data_split+rag_tool". Filter by substring to slice on either dimension.

  ablation case  : from the `ablation:<name>` wandb tag (baseline, no_data_*, ...)
  knowledge mode : from config.knowledge_mode (mapped: icl->static,
                   rag->rag_step, rag_od->rag_tool, none->none)
"""

import argparse
import json
import os
from datetime import datetime

import pandas as pd
import wandb

from . import config as ab_config


KNOWLEDGE_MODE_RENAME = {
    "icl": "static",
    "rag": "rag_step",
    "rag_od": "rag_tool",
    "none": "none",
}


def _unwrap(v):
    return v.get("value") if isinstance(v, dict) else v


def _load_config(run):
    try:
        return json.loads(run.config) if isinstance(run.config, str) else dict(run.config)
    except Exception:
        return {}


def _load_summary(run):
    try:
        return dict(run.summary)
    except Exception:
        return {}


def _experiment_tag(run, allowed_tags):
    for t in run.tags:
        if t in allowed_tags:
            return t
    return "unknown"


def _ablation_case(run):
    for t in run.tags:
        if t.startswith("ablation:"):
            return t[len("ablation:"):]
    return "baseline"


def _knowledge_mode(run, cfg):
    # Prefer the wandb tag (set at submission), fall back to run config.
    for t in run.tags:
        if t.startswith("knowledge:"):
            raw = t[len("knowledge:"):]
            return KNOWLEDGE_MODE_RENAME.get(raw, raw)
    raw = _unwrap(cfg.get("knowledge_mode")) or "none"
    return KNOWLEDGE_MODE_RENAME.get(raw, raw)


def _runtime_minutes(run, summary):
    rt = summary.get("_runtime")
    if rt is not None:
        try:
            return round(float(rt) / 60.0, 2)
        except Exception:
            pass
    try:
        created = datetime.fromisoformat(run.created_at.replace("Z", ""))
        heartbeat = datetime.fromisoformat(run.heartbeatAt.replace("Z", ""))
        return round((heartbeat - created).total_seconds() / 60.0, 2)
    except Exception:
        return None


def _selected_iter_and_metrics(run, metric_name):
    """Scan run history for validation/{metric}; pick iteration with max value."""
    if not metric_name:
        return None, None, None
    val_key = f"validation/{metric_name}"
    train_key = f"train/{metric_name}"
    try:
        hist = run.history(keys=[val_key, train_key], pandas=True, samples=10000)
    except Exception:
        hist = None
    if hist is None or len(hist) == 0 or val_key not in hist.columns:
        return None, None, None

    df = hist.dropna(subset=[val_key])
    df = df[df[val_key] != -1]
    if len(df) == 0:
        return None, None, None

    best = df.loc[df[val_key].idxmax()]
    selected_iter = int(best["_step"]) if "_step" in best and pd.notna(best["_step"]) else None
    val_val = float(best[val_key])
    train_val = None
    if train_key in df.columns:
        tv = best.get(train_key)
        if pd.notna(tv) and tv != -1:
            train_val = float(tv)
    return selected_iter, train_val, val_val


def extract(experiments=None, output_file="run_info.csv"):
    api_key = os.getenv("WANDB_API_KEY")
    if api_key:
        wandb.login(key=api_key, relogin=True)

    api = wandb.Api()
    allowed_tags = experiments or ab_config.ABLATION_TAGS
    print(f"Fetching runs from {ab_config.WANDB_ENTITY}/{ab_config.WANDB_PROJECT}")
    print(f"  experiment tags: {allowed_tags}")

    runs = api.runs(
        f"{ab_config.WANDB_ENTITY}/{ab_config.WANDB_PROJECT}",
        filters={"tags": {"$in": allowed_tags}},
    )

    rows = []
    for run in runs:
        cfg = _load_config(run)
        summary = _load_summary(run)

        dataset = _unwrap(cfg.get("dataset"))
        llm = _unwrap(cfg.get("model_name")) or _unwrap(cfg.get("model"))
        metric_name = _unwrap(cfg.get("val_metric"))

        ablation_case = _ablation_case(run)
        knowledge_mode = _knowledge_mode(run, cfg)
        configuration = f"{ablation_case}+{knowledge_mode}"

        inference_stage = summary.get("inference_stage", 0)
        success = inference_stage == 2

        test_metric = summary.get(metric_name) if metric_name else None
        if test_metric == -1:
            test_metric = None

        selected_iter, train_val, val_val = _selected_iter_and_metrics(run, metric_name)

        rows.append({
            "experiment": _experiment_tag(run, allowed_tags),
            "dataset": dataset,
            "llm": llm,
            "configuration": configuration,
            "replicate_id": run.name or run.id,
            "success": success,
            "selected_iteration": selected_iter,
            "train_metric": train_val,
            "validation_metric": val_val,
            "test_metric": test_metric,
            "metric_name": metric_name,
            "runtime_minutes": _runtime_minutes(run, summary),
        })
        print(f"  {run.name:40} {configuration:35} success={success} iter={selected_iter}")

    df = pd.DataFrame(rows)
    df.to_csv(output_file, index=False)
    print(f"\nSaved {len(df)} runs to {output_file}")
    return df


def main():
    p = argparse.ArgumentParser()
    p.add_argument("--experiments", nargs="+", default=None,
                   help="W&B tags to include (defaults to ABLATION_TAGS in config.py)")
    p.add_argument("--output", default="run_info.csv")
    args = p.parse_args()
    extract(experiments=args.experiments, output_file=args.output)


if __name__ == "__main__":
    main()
