#!/usr/bin/env python3
"""List ablation study run names from W&B, optionally saving to file."""

import os
import json
import wandb
from collections import defaultdict
from . import config


def list_runs(output_file=None):
    """Fetch all ablation run names from W&B and optionally save to file.

    Returns a list of dicts with run metadata: name, id, model, dataset, ablation_config.
    """
    # Login to W&B
    api_key = os.getenv("WANDB_API_KEY")
    if api_key:
        wandb.login(key=api_key, relogin=True)

    api = wandb.Api()

    print(f"Fetching runs from {config.WANDB_ENTITY}/{config.WANDB_PROJECT}...")

    filters = {"tags": {"$in": config.ABLATION_TAGS}}
    all_runs = api.runs(
        f"{config.WANDB_ENTITY}/{config.WANDB_PROJECT}",
        filters=filters
    )

    runs = []
    runs_by_config = defaultdict(list)

    for run in all_runs:
        # Find ablation config tag
        ablation_config = None
        for tag in run.tags:
            if tag.startswith("ablation:"):
                ablation_config = tag.replace("ablation:", "")
                break

        if not ablation_config:
            ablation_config = "unknown"

        # Parse run config for dataset and model
        dataset = None
        model = None
        try:
            cfg = json.loads(run.config) if isinstance(run.config, str) else dict(run.config)

            if "dataset" in cfg:
                val = cfg["dataset"]
                dataset = val.get("value") if isinstance(val, dict) else val

            if "model_name" in cfg:
                val = cfg["model_name"]
                model = val.get("value") if isinstance(val, dict) else val
            elif "model" in cfg:
                val = cfg["model"]
                model = val.get("value") if isinstance(val, dict) else val
        except Exception:
            pass

        run_info = {
            'name': run.name,
            'id': run.id,
            'model': model or 'unknown',
            'dataset': dataset or 'unknown',
            'ablation_config': ablation_config,
        }
        runs.append(run_info)
        runs_by_config[ablation_config].append(run_info)

    # Print summary
    print(f"\nFound {len(runs)} runs across {len(runs_by_config)} ablation configs:\n")
    for ablation_cfg in sorted(runs_by_config):
        print(f"  {ablation_cfg:30} {len(runs_by_config[ablation_cfg]):3} runs")

    # Save to file if requested
    if output_file:
        os.makedirs(os.path.dirname(output_file) or ".", exist_ok=True)
        with open(output_file, 'w') as f:
            for run_info in sorted(runs, key=lambda r: r['name']):
                f.write(run_info['name'] + '\n')
        print(f"\nSaved {len(runs)} run names to {output_file}")

    return runs


if __name__ == "__main__":
    list_runs()
