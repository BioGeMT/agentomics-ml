import argparse
import shutil
from importlib import util
from pathlib import Path

import pandas as pd
import yaml


def infer_target(train_df: pd.DataFrame) -> str:
    return "target" if "target" in train_df.columns else train_df.columns[-1]


def generate_task(clone_dir: Path, dataset_root: Path, templates_dir: Path, competitors_dir: Path, name: str) -> None:
    src = dataset_root / name
    train_df = pd.read_csv(src / "train.csv")
    test_df = pd.read_csv(src / "test.csv")
    target_col = infer_target(train_df)

    tasks_pkg = clone_dir / "biomlbench/tasks/agentomics"
    tasks_pkg.mkdir(parents=True, exist_ok=True)
    (tasks_pkg / "__init__.py").write_text("# Agentomics task package\n")

    task_dir = tasks_pkg / name
    task_dir.mkdir(parents=True, exist_ok=True)
    (task_dir / "__init__.py").write_text("")

    desc_src = src / "dataset_description.md"
    shutil.copy(desc_src, task_dir / "description.md")

    prepare_template = (templates_dir / "prepare_template.py").read_text()
    config_template = (templates_dir / "config_template.yaml").read_text()
    grade_template = (templates_dir / "grade_template.py").read_text()
    leaderboard_template = templates_dir / "leaderboard_template.csv"

    (task_dir / "prepare.py").write_text(prepare_template.format(target=target_col))
    (task_dir / "config.yaml").write_text(config_template.format(name=name))
    (task_dir / "grade.py").write_text(grade_template)
    shutil.copy(leaderboard_template, task_dir / "leaderboard.csv")

    data_dir = competitors_dir / "data"
    raw_dir = data_dir / "agentomics" / name / "raw"
    public_dir = data_dir / "agentomics" / name / "prepared/public"
    private_dir = data_dir / "agentomics" / name / "prepared/private"
    raw_dir.mkdir(parents=True, exist_ok=True)
    public_dir.mkdir(parents=True, exist_ok=True)
    private_dir.mkdir(parents=True, exist_ok=True)

    shutil.copy(src / "train.csv", raw_dir / "train.csv")
    shutil.copy(src / "test.csv", raw_dir / "test.csv")

    spec = util.spec_from_file_location("agentomics_prepare", task_dir / "prepare.py")
    module = util.module_from_spec(spec)
    spec.loader.exec_module(module)
    module.prepare(raw_dir, public_dir, private_dir)
    shutil.copy(desc_src, public_dir / "description.md")


def main() -> None:
    # Find directories relative to this script location
    script_dir = Path(__file__).resolve().parent
    competitors_dir = script_dir.parent
    clone_dir = competitors_dir / "biomlbench"
    dataset_root = competitors_dir.parent / "datasets"
    templates_dir = competitors_dir / "templates"

    # Prepare ALL datasets found in datasets directory
    dataset_names = [d.name for d in dataset_root.iterdir() if d.is_dir()]

    for name in dataset_names:
        print(f"[setup_tasks] Preparing dataset: {name}")
        generate_task(clone_dir, dataset_root, templates_dir, competitors_dir, name)

    print(f"[setup_tasks] Generated {len(dataset_names)} Agentomics tasks")


if __name__ == "__main__":
    main()
