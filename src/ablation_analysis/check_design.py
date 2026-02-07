#!/usr/bin/env python3
"""Check if runs match expected experimental design: 2 datasets × 3 models × 8 configs × 3 replicates."""

import os
import json
import wandb
from collections import defaultdict
from . import config


def check_experimental_design():
    """Check if runs match the expected 2×3×8×3 design."""
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

    # Track runs by (dataset, model, ablation_config)
    runs_by_combination = defaultdict(list)
    datasets = set()
    models = set()
    ablation_configs = set()

    for run in all_runs:
        # Get ablation config
        ablation_config = None
        for tag in run.tags:
            if tag.startswith("ablation:"):
                ablation_config = tag.replace("ablation:", "")
                ablation_configs.add(ablation_config)
                break

        if not ablation_config:
            continue

        # Get dataset and model from config
        try:
            if isinstance(run.config, str):
                cfg = json.loads(run.config)
            else:
                cfg = dict(run.config)

            # Extract dataset
            dataset = None
            if "dataset" in cfg:
                val = cfg["dataset"]
                dataset = val.get("value") if isinstance(val, dict) else val

            # Extract model
            model = None
            if "model_name" in cfg:
                val = cfg["model_name"]
                model = val.get("value") if isinstance(val, dict) else val
            elif "model" in cfg:
                val = cfg["model"]
                model = val.get("value") if isinstance(val, dict) else val

            if dataset and model:
                datasets.add(dataset)
                models.add(model)
                key = (dataset, model, ablation_config)
                runs_by_combination[key].append({
                    'id': run.id,
                    'name': run.name
                })

        except Exception as e:
            print(f"Warning: Could not parse config for run {run.name}: {e}")
            continue

    # Expected design
    EXPECTED_REPLICATES = 3
    expected_total = len(datasets) * len(models) * len(ablation_configs) * EXPECTED_REPLICATES

    print("="*70)
    print("EXPERIMENTAL DESIGN")
    print("="*70)
    print(f"Datasets:  {len(datasets)} → {sorted(datasets)}")
    print(f"Models:    {len(models)} → {sorted(models)}")
    print(f"Ablation configs: {len(ablation_configs)} → {sorted(ablation_configs)}")
    print(f"Expected replicates: {EXPECTED_REPLICATES}")
    print(f"\nExpected total: {len(datasets)} × {len(models)} × {len(ablation_configs)} × {EXPECTED_REPLICATES} = {expected_total}")
    print(f"Actual total:   {sum(len(runs) for runs in runs_by_combination.values())}")

    # Find combinations with wrong number of replicates
    print("\n" + "="*70)
    print("CHECKING REPLICATES PER COMBINATION")
    print("="*70)

    missing = []
    extra = []

    for dataset in sorted(datasets):
        for model in sorted(models):
            print(f"\n{dataset} × {model}:")
            print("-" * 70)
            for ablation_config in sorted(ablation_configs):
                key = (dataset, model, ablation_config)
                runs = runs_by_combination.get(key, [])
                count = len(runs)

                status = "✓" if count == EXPECTED_REPLICATES else "✗"
                print(f"  {ablation_config:30} {count} runs {status}")

                if count < EXPECTED_REPLICATES:
                    missing.append((dataset, model, ablation_config, count))
                elif count > EXPECTED_REPLICATES:
                    extra.append((dataset, model, ablation_config, count, runs))

    # Summary of issues
    print("\n" + "="*70)
    print("ISSUES FOUND")
    print("="*70)

    if missing:
        print(f"\nCombinations with MISSING runs ({len(missing)}):")
        print("-" * 70)
        for dataset, model, ablation_config, count in missing:
            missing_count = EXPECTED_REPLICATES - count
            print(f"  {dataset} × {model} × {ablation_config}")
            print(f"    → Has {count}, needs {EXPECTED_REPLICATES} (missing {missing_count})")

    if extra:
        print(f"\nCombinations with EXTRA runs ({len(extra)}):")
        print("-" * 70)
        for dataset, model, ablation_config, count, runs in extra:
            extra_count = count - EXPECTED_REPLICATES
            print(f"  {dataset} × {model} × {ablation_config}")
            print(f"    → Has {count}, should be {EXPECTED_REPLICATES} (extra {extra_count})")
            for run in runs:
                print(f"      - {run['name']}")

    if not missing and not extra:
        print("\n✓ All combinations have exactly 3 replicates!")

    total_missing = sum(EXPECTED_REPLICATES - count for _, _, _, count in missing)
    total_extra = sum(count - EXPECTED_REPLICATES for _, _, _, count, _ in extra)

    print(f"\nTotal missing runs: {total_missing}")
    print(f"Total extra runs:   {total_extra}")
    print(f"Net difference:     {total_extra - total_missing}")

    return {
        'datasets': sorted(datasets),
        'models': sorted(models),
        'ablation_configs': sorted(ablation_configs),
        'expected_total': expected_total,
        'actual_total': sum(len(runs) for runs in runs_by_combination.values()),
        'missing': missing,
        'extra': extra
    }


if __name__ == "__main__":
    check_experimental_design()
