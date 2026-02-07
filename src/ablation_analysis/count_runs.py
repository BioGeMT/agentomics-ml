#!/usr/bin/env python3
"""Count runs for each ablation study tag."""

import os
import wandb
from collections import defaultdict
from . import config


def count_runs_by_tag():
    """Count how many runs exist for each ablation tag."""
    # Login to W&B
    api_key = os.getenv("WANDB_API_KEY")
    if api_key:
        wandb.login(key=api_key, relogin=True)

    api = wandb.Api()

    print(f"Fetching runs from {config.WANDB_ENTITY}/{config.WANDB_PROJECT}...\n")

    # Fetch all runs with any of our tags
    filters = {"tags": {"$in": config.ABLATION_TAGS}}
    all_runs = api.runs(
        f"{config.WANDB_ENTITY}/{config.WANDB_PROJECT}",
        filters=filters
    )

    # Track runs by main tag and ablation config
    runs_by_tag = {tag: [] for tag in config.ABLATION_TAGS}
    ablation_configs_by_main_tag = {tag: defaultdict(list) for tag in config.ABLATION_TAGS}
    all_ablation_configs = set()

    # Track total runs per ablation config across all main tags
    runs_per_ablation_config = defaultdict(int)

    for run in all_runs:
        run_id = run.id
        run_name = run.name

        # Find main tag
        main_tag = None
        for tag in config.ABLATION_TAGS:
            if tag in run.tags:
                main_tag = tag
                break

        if not main_tag:
            continue

        # Find ablation config tag (starts with "ablation:")
        ablation_config = None
        for tag in run.tags:
            if tag.startswith("ablation:"):
                ablation_config = tag.replace("ablation:", "")
                all_ablation_configs.add(ablation_config)
                break

        if not ablation_config:
            ablation_config = "unknown"

        runs_by_tag[main_tag].append(run_id)
        ablation_configs_by_main_tag[main_tag][ablation_config].append({
            'id': run_id,
            'name': run_name
        })
        runs_per_ablation_config[ablation_config] += 1

    # Print breakdown by main tag
    print("="*70)
    print("RUNS BY MAIN TAG AND ABLATION CONFIG")
    print("="*70)

    total_runs = 0
    for main_tag in config.ABLATION_TAGS:
        configs = ablation_configs_by_main_tag[main_tag]
        tag_total = len(runs_by_tag[main_tag])
        total_runs += tag_total

        print(f"\n{main_tag}: {tag_total} runs")
        print("-" * 70)

        for ablation_config in sorted(configs.keys()):
            runs = configs[ablation_config]
            print(f"  {ablation_config:30} {len(runs):3} runs")

    # Print sum by ablation config
    print("\n" + "="*70)
    print("TOTAL RUNS PER ABLATION CONFIG (across all main tags)")
    print("="*70)

    for ablation_config in sorted(runs_per_ablation_config.keys()):
        count = runs_per_ablation_config[ablation_config]
        print(f"{ablation_config:30} {count:3} runs")

    # Summary
    print("\n" + "="*70)
    print("SUMMARY")
    print("="*70)
    print(f"Total unique runs:              {total_runs}")
    print(f"Unique main tags:               {len(config.ABLATION_TAGS)}")
    print(f"Unique ablation configs found:  {len(all_ablation_configs)}")

    # Check if all ablation configs have the same count
    counts = list(runs_per_ablation_config.values())
    if len(set(counts)) == 1:
        print(f"All ablation configs have:      {counts[0]} runs each")
        print(f"Expected total:                 {len(all_ablation_configs)} × {counts[0]} = {len(all_ablation_configs) * counts[0]}")
    else:
        min_count = min(counts)
        max_count = max(counts)
        print(f"Ablation config counts vary:    {min_count} to {max_count} runs")
        print(f"If all had {min_count} runs:         {len(all_ablation_configs)} × {min_count} = {len(all_ablation_configs) * min_count}")
        print(f"If all had {max_count} runs:         {len(all_ablation_configs)} × {max_count} = {len(all_ablation_configs) * max_count}")

    return {
        'total_runs': total_runs,
        'runs_by_tag': {tag: len(runs) for tag, runs in runs_by_tag.items()},
        'ablation_configs': sorted(all_ablation_configs),
        'runs_per_config': dict(runs_per_ablation_config),
        'breakdown': {
            tag: {cfg: len(runs) for cfg, runs in configs.items()}
            for tag, configs in ablation_configs_by_main_tag.items()
        }
    }


if __name__ == "__main__":
    count_runs_by_tag()
