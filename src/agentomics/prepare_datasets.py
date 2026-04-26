import argparse
from pathlib import Path
from rich.console import Console

from agentomics.utils.dataset_utils import prepare_dataset
from agentomics.utils.datasets_interactive_utils import prepare_all_datasets
from agentomics.utils.path_defaults import resolve_agentomics_paths

def parse_args():
    parser = argparse.ArgumentParser(description="Dataset preparation with auto-detection")
    parser.add_argument('--dataset-dir', type=Path, help='Single dataset directory to prepare')
    parser.add_argument('--datasets-dir', type=Path, help='Directory containing raw datasets. Defaults to DATASETS_DIR or ./datasets from the repo root/current working directory.')
    parser.add_argument('--prepared-datasets-dir', type=Path, help='Directory to write prepared datasets. Defaults to PREPARED_DATASETS_DIR or ./prepared_datasets from the repo root/current working directory.')
    parser.add_argument('--prepared-test-sets-dir', type=Path, help='Directory to write prepared test sets. Defaults to PREPARED_TEST_SETS_DIR/PREPARED_TESTS_DIR or ./prepared_test_sets from the repo root/current working directory.')
    parser.add_argument('--prepare-all', action='store_true', help='Prepare all datasets in datasets-dir and auto-detect their targets and tasks')
    parser.add_argument('--target-col', type=str, default=None, help='Target column name (auto-detected if not provided)')
    parser.add_argument('--task-type', choices=['classification', 'regression'], default=None, help='Task type (auto-detected if not provided)')
    parser.add_argument('--positive-class', help='Value used in the label column for a positive class (affects some binary classification metrics). If not provided, numeric labels are assigned based on the label appearance order in the train csv file.', default=None)
    parser.add_argument('--negative-class', help='Value used in the label column for a negative class (affects some binary classification metrics). If not provided, numeric labels are assigned based on the label appearance order in the train csv file.', default=None)
    return parser.parse_args()

def main():
    args = parse_args()
    console = Console()

    paths = resolve_agentomics_paths(
        datasets_dir=args.datasets_dir,
        prepared_datasets_dir=args.prepared_datasets_dir,
        prepared_test_sets_dir=args.prepared_test_sets_dir,
    )

    paths.prepared_datasets_dir.mkdir(parents=True, exist_ok=True)
    paths.prepared_test_sets_dir.mkdir(parents=True, exist_ok=True)

    if args.prepare_all or not args.dataset_dir:
        prepare_all_datasets(paths.datasets_dir, paths.prepared_datasets_dir, paths.prepared_test_sets_dir)
    else:
        console.print(f'[blue]Preparing dataset "{args.dataset_dir.name}"')# for {task_type} task with target column "{target_col}"[/blue]')
        try:
            prepare_dataset(
                dataset_dir=args.dataset_dir,
                target_col=args.target_col,
                positive_class=args.positive_class, #is auto-detected inside - do the same for target/task ?
                negative_class=args.negative_class,
                task_type=args.task_type,
                output_dir=paths.prepared_datasets_dir,
                test_sets_output_dir=paths.prepared_test_sets_dir
            )
            console.print(f"[green]Dataset '{args.dataset_dir.name}' prepared successfully![/green]")
        except Exception as e:
            console.print(f"[red]Dataset '{args.dataset_dir.name}' preparation failed! {e}[/red]")
        
if __name__ == "__main__":
    main()
