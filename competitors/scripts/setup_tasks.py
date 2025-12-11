import argparse
import shutil
from importlib import util
from pathlib import Path

import pandas as pd
import yaml


def infer_target(train_df: pd.DataFrame) -> str:
    if "numeric_label" not in train_df.columns:
        raise ValueError(f"Expected 'numeric_label' column in train dataframe, but found columns: {list(train_df.columns)}")
    return "numeric_label"


def generate_task(clone_dir: Path, prepared_datasets_dir: Path, prepared_test_sets_dir: Path, templates_dir: Path, competitors_dir: Path, name: str) -> None:
    """
    Convert an Agentomics dataset to BioMLBench task format by creating task directory structure,
    populates it with templated files (prepare.py, config.yaml, grade.py), copies prepared data,
    and executes the prepare script to generate public/private data splits for agent run.
    """
    src_train = prepared_datasets_dir / name
    src_test = prepared_test_sets_dir / name
    train_df = pd.read_csv(src_train / "train.csv")
    test_df = pd.read_csv(src_test / "test.csv")
    target_col = infer_target(train_df)

    tasks_pkg = clone_dir / "biomlbench/tasks/agentomics"
    tasks_pkg.mkdir(parents=True, exist_ok=True)
    (tasks_pkg / "__init__.py").write_text("# Agentomics task package\n")

    task_dir = tasks_pkg / name
    task_dir.mkdir(parents=True, exist_ok=True)
    (task_dir / "__init__.py").write_text("")

    desc_src = src_train / "dataset_description.md"
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

    shutil.copy(src_train / "train.csv", raw_dir / "train.csv")
    shutil.copy(src_test / "test.csv", raw_dir / "test.csv")

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
    prepared_datasets_dir = competitors_dir.parent / "prepared_datasets"
    prepared_test_sets_dir = competitors_dir.parent / "prepared_test_sets"
    templates_dir = competitors_dir / "templates"

    # Prepare ALL datasets found in prepared_datasets directory
    dataset_names = [d.name for d in prepared_datasets_dir.iterdir() if d.is_dir()]

    for name in dataset_names:
        print(f"[setup_tasks] Preparing dataset: {name}")
        generate_task(clone_dir, prepared_datasets_dir, prepared_test_sets_dir, templates_dir, competitors_dir, name)

    print(f"[setup_tasks] Generated {len(dataset_names)} Agentomics tasks")


if __name__ == "__main__":
    main()
