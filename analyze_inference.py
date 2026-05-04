import yaml
from pathlib import Path

"""
Assumes remote_outputs folder containing all the output folder runs exists
Check out of the run that did not skip inference, where they defined the inference script. This is checked only for the 1st iter of each run
When you skip inference, where is it produced? 
"""

def find_log_and_config(agent_dir):
    extras = agent_dir / "extras" / "run_logs" / "wandb"
    if not extras.exists():
        return None, None
    run_dirs = list(extras.glob("run-*/files"))
    if not run_dirs:
        return None, None
    files_dir = run_dirs[0]
    log_path = files_dir / "output.log"
    config_path = files_dir / "config.yaml"
    if not log_path.exists() or not config_path.exists():
        return None, None
    return log_path, config_path


def get_ablation_config(config_path):
    with open(config_path, "r") as f:
        config = yaml.safe_load(f)
    steps_to_skip = config.get("steps_to_skip", {}).get("value", [])
    tags = config.get("tags", {}).get("value", [])
    ablation_tag = next((t for t in tags if t.startswith("ablation:")), None)
    return steps_to_skip, ablation_tag


def find_first_line(lines, pattern):
    for i, line in enumerate(lines, 1):
        if pattern in line:
            return i
    return None


def find_inference_written(lines):
    for i, line in enumerate(lines):
        if "write_python" in line:
            window = lines[i:i+4]
            if any("inference.py" in l for l in window):
                return i + 1  # 1-indexed
    return None


def analyze_log(log_path):
    with open(log_path, "r", errors="replace") as f:
        lines = f.readlines()

    exploration_end = find_first_line(lines, "Next task: choose the model architecture")
    training_start = find_first_line(lines, "Next task: implement any necessary code for training")
    inference_written = find_inference_written(lines)
    has_inference_step = find_first_line(lines, "Next task: create inference.py") is not None

    return {
        "exploration_end_line": exploration_end,
        "training_start_line": training_start,
        "inference_written_line": inference_written,
        "has_inference_step_prompt": has_inference_step,
        "total_lines": len(lines),
    }


def classify(inf_line, exploration_end, training_start):
    if inf_line is None:
        return "NOT_WRITTEN"
    if exploration_end and inf_line < exploration_end:
        return "IN_EXPLORATION"
    if training_start and inf_line < training_start:
        return "BETWEEN_EXPLORATION_AND_TRAINING"
    if training_start and inf_line >= training_start:
        return "IN_OR_AFTER_TRAINING"
    return "INCONCLUSIVE"


def main():
    remote_outputs = Path("remote_outputs")
    if not remote_outputs.exists():
        print("remote_outputs/ not found. Run from its parent directory.")
        return

    results = []

    for agent_dir in sorted(remote_outputs.iterdir()):
        if not agent_dir.is_dir():
            continue

        log_path, config_path = find_log_and_config(agent_dir)
        if log_path is None:
            continue

        steps_to_skip, ablation_tag = get_ablation_config(config_path)

        if steps_to_skip != ["final_outcome"]:
            continue

        if ablation_tag and ablation_tag != "ablation:no_final_outcome":
            print(f"  [{agent_dir.name}] WARNING: unexpected tag — {ablation_tag}")

        stats = analyze_log(log_path)
        inf_line = stats["inference_written_line"]
        exp_end = stats["exploration_end_line"]
        train_line = stats["training_start_line"]

        category = classify(inf_line, exp_end, train_line)

        results.append({
            "agent_id": agent_dir.name,
            "inference_written_line": inf_line,
            "exploration_end_line": exp_end,
            "training_start_line": train_line,
            "category": category,
        })

        print(f"  [{agent_dir.name}] inf@{inf_line}, exp_end@{exp_end}, train@{train_line} → {category}")

    # Summary
    from collections import Counter
    counts = Counter(r["category"] for r in results)

    total = len(results)
    print(f"\n{'='*50}")
    print(f"no_final_outcome runs found:           {total}")
    for cat, n in sorted(counts.items()):
        print(f"  {cat:<40} {n}")

    # Conclusive = everything except NOT_WRITTEN with no training anchor
    written = [r for r in results if r["category"] != "NOT_WRITTEN"]
    in_exp = counts["IN_EXPLORATION"]
    print(f"\nOf {len(written)} runs where inference.py was written:")
    print(f"  Written during Data Exploration:     {in_exp} / {len(written)}")


if __name__ == "__main__":
    main()